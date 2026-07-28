"""TIGER Trie 约束 Beam Search 推理与评估。

加载训练好的 TIGER checkpoint，根据用户历史自回归生成 semantic ID，
通过前缀约束保证生成路径对应真实 item，再计算 HR/NDCG 和无效码率。
"""

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiger_min.data.splits import load_splits
from tiger_min.tiger.dataset import TigerNextItemDataset, collate_tiger_batch
from tiger_min.tiger.model import TigerTransformer
from tiger_min.tiger.tokenizer import TigerTokenizer
from tiger_min.tiger.trie import SemanticIdTrie
from tiger_min.utils import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Trie-constrained TIGER beam inference."
    )
    parser.add_argument("--checkpoint", default="data/processed/beauty/tiger/tiger.pt")
    parser.add_argument("--splits", default="data/processed/beauty/splits.pt")
    parser.add_argument("--split", choices=["valid", "test"], default="valid")
    parser.add_argument("--output", default="data/processed/beauty/tiger/eval.json")
    parser.add_argument("--beam-size", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser


def load_tiger_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[TigerTransformer, TigerTokenizer, dict]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_args = checkpoint["model_args"]
    tokenizer_args = checkpoint["tokenizer_args"]

    tokenizer = TigerTokenizer.from_file(
        tokenizer_args["semantic_ids_path"],
        codebook_size=int(tokenizer_args["codebook_size"]),
        num_quantizer_layers=int(tokenizer_args["num_quantizer_layers"]),
    )
    model = TigerTransformer(**model_args)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model, tokenizer, checkpoint


def build_semantic_id_to_item(tokenizer: TigerTokenizer) -> dict[tuple[int, ...], int]:
    semantic_id_to_item: dict[tuple[int, ...], int] = {}
    for item_id, semantic_id in enumerate(tokenizer.semantic_ids.tolist()):
        key = tuple(int(code) for code in semantic_id)
        if key in semantic_id_to_item:
            raise ValueError(f"Duplicate semantic id found: {key}")
        # This table is the final bridge from generated semantic ids back to items.
        semantic_id_to_item[key] = item_id
    return semantic_id_to_item


def generated_tokens_to_item(
    token_row: list[int],
    tokenizer: TigerTokenizer,
    semantic_id_to_item: dict[tuple[int, ...], int],
) -> int:
    # Convert position-offset tokens back to raw semantic id codes before lookup.
    semantic_id = tokenizer.semantic_tokens_to_id(token_row)
    return semantic_id_to_item.get(semantic_id, -1)


@torch.no_grad()
def batch_beam_search_decode(
    model: TigerTransformer,
    tokenizer: TigerTokenizer,
    semantic_id_trie: SemanticIdTrie,
    encoder_input_ids: torch.LongTensor,
    encoder_attention_mask: torch.LongTensor,
    beam_size: int,
) -> tuple[torch.LongTensor, torch.Tensor]:
    if beam_size <= 0:
        raise ValueError("beam_size must be positive.")

    batch_size = int(encoder_input_ids.shape[0])
    device = encoder_input_ids.device
    beam_tokens = torch.full(
        (batch_size, 1, 1),
        fill_value=tokenizer.special_tokens.bos,
        dtype=torch.long,
        device=device,
    )
    beam_scores = torch.zeros(batch_size, 1, dtype=torch.float, device=device)

    # Encode user histories once; all beams for the same user reuse this memory.
    memory, memory_key_padding_mask = model.encode(
        encoder_input_ids=encoder_input_ids,
        encoder_attention_mask=encoder_attention_mask,
    )

    for position in range(tokenizer.semantic_id_length):
        current_beam_size = int(beam_tokens.shape[1])
        current_length = int(beam_tokens.shape[2])
        position_offset = tokenizer.position_offsets[position]
        position_vocab_size = tokenizer.position_vocab_sizes[position]

        # Flatten beams into the batch dimension for one vectorized decoder call.
        flat_decoder_input_ids = beam_tokens.reshape(
            batch_size * current_beam_size,
            current_length,
        )
        flat_memory = memory.repeat_interleave(
            current_beam_size,
            dim=0,
        )
        flat_memory_key_padding_mask = memory_key_padding_mask.repeat_interleave(
            current_beam_size,
            dim=0,
        )

        logits = model.decode(
            decoder_input_ids=flat_decoder_input_ids,
            memory=flat_memory,
            memory_key_padding_mask=flat_memory_key_padding_mask,
        )
        logits = logits[
            :,
            -1,
            position_offset : position_offset + position_vocab_size,
        ]

        # Mask paths that cannot be completed into any real item semantic ID.
        flat_prefixes = flat_decoder_input_ids[:, 1:].detach().cpu().tolist()
        valid_token_mask = torch.zeros(
            (len(flat_prefixes), position_vocab_size),
            dtype=torch.bool,
        )
        for row_index, prefix in enumerate(flat_prefixes):
            allowed_tokens = semantic_id_trie.allowed_next_tokens(tuple(prefix))
            if not allowed_tokens:
                raise RuntimeError(f"Trie contains no continuation for prefix: {prefix}")
            local_token_ids = [token - position_offset for token in allowed_tokens]
            valid_token_mask[row_index, local_token_ids] = True
        logits = logits.masked_fill(~valid_token_mask.to(device), float("-inf"))

        log_probs = torch.log_softmax(logits, dim=-1).reshape(
            batch_size,
            current_beam_size,
            position_vocab_size,
        )
        candidate_scores = beam_scores.unsqueeze(-1) + log_probs
        candidate_scores = candidate_scores.reshape(batch_size, -1)

        # Keep the best paths across all previous beams and next-token choices.
        min_valid_candidates = int(
            torch.isfinite(candidate_scores).sum(dim=1).min().item()
        )
        if min_valid_candidates == 0:
            raise RuntimeError("Trie-constrained decoding produced no valid candidate.")
        next_beam_size = min(beam_size, min_valid_candidates)
        top_scores, top_indices = torch.topk(
            candidate_scores,
            k=next_beam_size,
            dim=-1,
        )
        selected_beam_indices = top_indices // position_vocab_size
        selected_token_indices = top_indices % position_vocab_size
        selected_tokens = torch.gather(
            beam_tokens,
            dim=1,
            index=selected_beam_indices.unsqueeze(-1).expand(
                -1,
                -1,
                current_length,
            ),
        )
        next_tokens = (position_offset + selected_token_indices).unsqueeze(-1)
        beam_tokens = torch.cat([selected_tokens, next_tokens], dim=-1)
        beam_scores = top_scores

    return beam_tokens[:, :, 1:], beam_scores


def beam_tokens_to_ranked_items(
    beams: list[tuple[list[int], float]],
    tokenizer: TigerTokenizer,
    semantic_id_to_item: dict[tuple[int, ...], int],
    top_k: int,
) -> tuple[list[int], int]:
    ranked_items: list[int] = []
    seen_items: set[int] = set()
    invalid_semantic_id_count = 0

    for tokens, _ in beams:
        item_id = generated_tokens_to_item(
            token_row=tokens,
            tokenizer=tokenizer,
            semantic_id_to_item=semantic_id_to_item,
        )
        if item_id < 0:
            # Invalid ids are generated token paths that do not map to any item.
            invalid_semantic_id_count += 1
            continue
        if item_id in seen_items:
            continue
        seen_items.add(item_id)
        if len(ranked_items) < top_k:
            ranked_items.append(item_id)

    return ranked_items, invalid_semantic_id_count


def update_ranking_metrics(
    ranked_items: list[int],
    target_item: int,
    cutoffs: list[int],
    metric_sums: dict[str, float],
) -> None:
    for k in cutoffs:
        top_k_items = ranked_items[:k]
        if target_item in top_k_items:
            rank = top_k_items.index(target_item) + 1
            metric_sums[f"hr@{k}"] += 1.0
            metric_sums[f"ndcg@{k}"] += 1.0 / math.log2(rank + 1)


@torch.no_grad()
def evaluate_beam_ranking(
    model: TigerTransformer,
    tokenizer: TigerTokenizer,
    dataloader: DataLoader,
    device: torch.device,
    beam_size: int,
    top_k: int,
    max_batches: int | None,
) -> dict:
    if beam_size < top_k:
        raise ValueError("beam_size must be greater than or equal to top_k.")

    semantic_id_to_item = build_semantic_id_to_item(tokenizer)
    semantic_id_trie = SemanticIdTrie(tokenizer)
    cutoffs = [k for k in [1, 5, 10, 20] if k <= top_k]
    metric_sums = {f"hr@{k}": 0.0 for k in cutoffs}
    metric_sums.update({f"ndcg@{k}": 0.0 for k in cutoffs})
    total_examples = 0
    valid_prediction_total = 0
    generated_semantic_id_total = 0
    invalid_semantic_id_total = 0

    progress = tqdm(dataloader, desc="tiger beam eval")
    for step, batch in enumerate(progress, start=1):
        if max_batches is not None and step > max_batches:
            break

        encoder_input_ids = batch["encoder_input_ids"].to(device)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device)
        target_items = batch["target_items"].tolist()

        generated_tokens, beam_scores = batch_beam_search_decode(
            model=model,
            tokenizer=tokenizer,
            semantic_id_trie=semantic_id_trie,
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            beam_size=beam_size,
        )

        for row_index, target_item in enumerate(target_items):
            beams = [
                (
                    [int(token) for token in generated_tokens[row_index, beam_index].tolist()],
                    float(beam_scores[row_index, beam_index].item()),
                )
                for beam_index in range(generated_tokens.shape[1])
            ]
            ranked_items, invalid_semantic_id_count = beam_tokens_to_ranked_items(
                beams=beams,
                tokenizer=tokenizer,
                semantic_id_to_item=semantic_id_to_item,
                top_k=top_k,
            )
            total_examples += 1
            valid_prediction_total += len(ranked_items)
            generated_semantic_id_total += len(beams)
            invalid_semantic_id_total += invalid_semantic_id_count
            update_ranking_metrics(
                ranked_items=ranked_items,
                target_item=int(target_item),
                cutoffs=cutoffs,
                metric_sums=metric_sums,
            )

        display_k = cutoffs[-1]
        progress.set_postfix(
            **{
                f"hr@{display_k}": metric_sums[f"hr@{display_k}"]
                / max(1, total_examples),
                "invalid": invalid_semantic_id_total
                / max(1, generated_semantic_id_total),
            }
        )

    metrics = {
        "num_examples": total_examples,
        "beam_size": beam_size,
        "top_k": top_k,
        "avg_valid_predictions": valid_prediction_total / max(1, total_examples),
        "generated_semantic_ids": generated_semantic_id_total,
        "invalid_semantic_ids": invalid_semantic_id_total,
        "invalid_semantic_id_rate": invalid_semantic_id_total
        / max(1, generated_semantic_id_total),
    }
    for key, value in metric_sums.items():
        metrics[key] = value / max(1, total_examples)
    return metrics


def load_eval_dataset(
    splits_path: str | Path,
    split: str,
    tokenizer: TigerTokenizer,
) -> TigerNextItemDataset:
    splits = load_splits(splits_path)
    samples = getattr(splits, split)
    return TigerNextItemDataset(samples=samples, tokenizer=tokenizer)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tokenizer, checkpoint = load_tiger_checkpoint(args.checkpoint, device=device)
    dataset = load_eval_dataset(
        splits_path=args.splits,
        split=args.split,
        tokenizer=tokenizer,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_tiger_batch,
    )
    metrics = evaluate_beam_ranking(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        device=device,
        beam_size=args.beam_size,
        top_k=args.top_k,
        max_batches=args.max_batches,
    )
    result = {
        "checkpoint_path": str(args.checkpoint),
        "split": args.split,
        "decode": "trie_constrained_beam",
        "best_epoch": checkpoint.get("best_epoch"),
        "best_valid_loss": checkpoint.get("best_valid_loss"),
        "metrics": metrics,
    }
    save_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()
