import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from tiger_min.tiger.dataset import (
    TigerDatasets,
    collate_tiger_batch,
    load_tiger_datasets,
)
from tiger_min.tiger.model import TigerTransformer
from tiger_min.utils import ensure_dir, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train minimal TIGER Transformer.")
    parser.add_argument("--splits", default="data/processed/beauty/splits.pt")
    parser.add_argument("--semantic-ids", default="data/processed/beauty/semantic_ids.pt")
    parser.add_argument("--output-dir", default="data/processed/beauty/tiger")
    parser.add_argument("--codebook-size", type=int, default=256)
    parser.add_argument("--num-quantizer-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--num-heads", type=int, default=6)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-decoder-layers", type=int, default=3)
    parser.add_argument("--dim-feedforward", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--max-encoder-length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-valid-batches", type=int, default=None)
    return parser


def infer_max_encoder_length(datasets: TigerDatasets) -> int:
    tokenizer = datasets.train.tokenizer
    max_history_items = 0
    for dataset in (datasets.train, datasets.valid, datasets.test):
        max_history_items = max(
            max_history_items,
            max(len(sample.history) for sample in dataset.samples),
        )
    return max_history_items * tokenizer.semantic_id_length


def build_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_tiger_batch,
    )


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def train_one_epoch(
    model: TigerTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    epochs: int,
    max_batches: int | None,
) -> float:
    model.train()
    total_loss = 0.0
    total_examples = 0

    progress = tqdm(dataloader, desc=f"tiger train {epoch}/{epochs}")
    for step, batch in enumerate(progress, start=1):
        if max_batches is not None and step > max_batches:
            break

        batch = move_batch_to_device(batch, device)
        output = model(
            encoder_input_ids=batch["encoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
        )
        if output.loss is None:
            raise RuntimeError("Model did not return loss during training.")

        optimizer.zero_grad()
        output.loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        batch_size = int(batch["encoder_input_ids"].shape[0])
        total_loss += float(output.loss.item()) * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=total_loss / total_examples)

    return total_loss / max(1, total_examples)


@torch.no_grad()
def evaluate_loss(
    model: TigerTransformer,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    epoch: int,
    epochs: int,
) -> float:
    model.eval()
    total_loss = 0.0
    total_examples = 0

    progress = tqdm(dataloader, desc=f"tiger valid {epoch}/{epochs}")
    for step, batch in enumerate(progress, start=1):
        if max_batches is not None and step > max_batches:
            break

        batch = move_batch_to_device(batch, device)
        output = model(
            encoder_input_ids=batch["encoder_input_ids"],
            encoder_attention_mask=batch["encoder_attention_mask"],
            decoder_input_ids=batch["decoder_input_ids"],
            labels=batch["labels"],
        )
        if output.loss is None:
            raise RuntimeError("Model did not return loss during evaluation.")

        batch_size = int(batch["encoder_input_ids"].shape[0])
        total_loss += float(output.loss.item()) * batch_size
        total_examples += batch_size
        progress.set_postfix(loss=total_loss / total_examples)

    return total_loss / max(1, total_examples)


def save_checkpoint(
    model: TigerTransformer,
    output_dir: str | Path,
    model_args: dict,
    tokenizer_args: dict,
    best_epoch: int,
    best_valid_loss: float,
) -> str:
    output_path = ensure_dir(output_dir)
    checkpoint_path = output_path / "tiger.pt"
    torch.save(
        {
            "model_args": model_args,
            "tokenizer_args": tokenizer_args,
            "best_epoch": best_epoch,
            "best_valid_loss": best_valid_loss,
            "state_dict": {
                name: tensor.detach().cpu()
                for name, tensor in model.state_dict().items()
            },
        },
        checkpoint_path,
    )
    return str(checkpoint_path)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    datasets = load_tiger_datasets(
        splits_path=args.splits,
        semantic_ids_path=args.semantic_ids,
        codebook_size=args.codebook_size,
        num_quantizer_layers=args.num_quantizer_layers,
    )
    tokenizer = datasets.train.tokenizer
    max_encoder_length = args.max_encoder_length or infer_max_encoder_length(datasets)
    max_decoder_length = tokenizer.semantic_id_length + 1

    train_loader = build_dataloader(
        datasets.train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    valid_loader = build_dataloader(
        datasets.valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_args = {
        "vocab_size": tokenizer.vocab_size,
        "max_encoder_length": max_encoder_length,
        "max_decoder_length": max_decoder_length,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_encoder_layers": args.num_encoder_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "dim_feedforward": args.dim_feedforward,
        "dropout": args.dropout,
        "pad_token_id": tokenizer.special_tokens.pad,
    }
    tokenizer_args = {
        "semantic_ids_path": str(args.semantic_ids),
        "codebook_size": args.codebook_size,
        "num_quantizer_layers": args.num_quantizer_layers,
        "vocab_size": tokenizer.vocab_size,
        "semantic_id_length": tokenizer.semantic_id_length,
        "position_vocab_sizes": tokenizer.position_vocab_sizes,
        "position_offsets": tokenizer.position_offsets,
    }

    model = TigerTransformer(**model_args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    epoch_metrics: list[dict[str, float]] = []
    best_epoch = -1
    best_valid_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] = {}

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
            epoch=epoch,
            epochs=args.epochs,
            max_batches=args.max_train_batches,
        )
        valid_loss = evaluate_loss(
            model=model,
            dataloader=valid_loader,
            device=device,
            max_batches=args.max_valid_batches,
            epoch=epoch,
            epochs=args.epochs,
        )
        metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
        }
        epoch_metrics.append(metrics)
        print(metrics)

        if valid_loss < best_valid_loss:
            best_epoch = epoch
            best_valid_loss = valid_loss
            best_state_dict = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

    if best_state_dict:
        model.load_state_dict(best_state_dict)

    checkpoint_path = save_checkpoint(
        model=model,
        output_dir=args.output_dir,
        model_args=model_args,
        tokenizer_args=tokenizer_args,
        best_epoch=best_epoch,
        best_valid_loss=best_valid_loss,
    )
    train_meta = {
        "splits_path": str(args.splits),
        "semantic_ids_path": str(args.semantic_ids),
        "output_dir": str(args.output_dir),
        "checkpoint_path": checkpoint_path,
        "device": str(device),
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "num_train_samples": len(datasets.train),
        "num_valid_samples": len(datasets.valid),
        "model_args": model_args,
        "tokenizer_args": tokenizer_args,
        "epoch_metrics": epoch_metrics,
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
    }
    save_json(train_meta, Path(args.output_dir) / "tiger_train_meta.json")
    print(train_meta)


if __name__ == "__main__":
    main()
