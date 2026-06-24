import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from tiger_min.embedding.dataset import (
    Item2VecDataset,
    Pair,
    build_positive_pairs,
)
from tiger_min.embedding.model import SkipGramNegSampling
from tiger_min.utils import ensure_dir, load_json, save_json, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train minimal item2vec embeddings.")
    parser.add_argument("--sequences", default="data/processed/beauty_5k/item2vec_sequences.json")
    parser.add_argument("--output", default="data/processed/beauty_5k/item_embeddings.pt")
    parser.add_argument("--data-meta", default=None)
    parser.add_argument("--num-items", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=2)
    parser.add_argument("--num-negatives", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


def load_sequences(path: str | Path) -> list[list[int]]:
    sequences = load_json(path)
    if not isinstance(sequences, list):
        raise ValueError("sequences must be a list of item-id lists.")
    return [[int(item) for item in sequence] for sequence in sequences]


def infer_num_items(sequences: list[list[int]]) -> int:
    max_item = max((max(sequence) for sequence in sequences if sequence), default=-1)
    if max_item < 0:
        raise ValueError("Cannot infer num_items from empty sequences.")
    return max_item + 1


def load_num_items_from_meta(path: str | Path) -> int:
    meta = load_json(path)
    if not isinstance(meta, dict) or "num_items" not in meta:
        raise ValueError("data_meta must contain `num_items`.")
    num_items = int(meta["num_items"])
    if num_items <= 0:
        raise ValueError("num_items in data_meta must be positive.")
    return num_items


def resolve_num_items(
    sequences_path: str | Path,
    sequences: list[list[int]],
    num_items: int | None,
    data_meta: str | Path | None,
) -> int:
    if num_items is not None:
        resolved_num_items = num_items
    else:
        meta_path = (
            Path(data_meta)
            if data_meta is not None
            else Path(sequences_path).with_name("data_meta.json")
        )
        if meta_path.exists():
            resolved_num_items = load_num_items_from_meta(meta_path)
        else:
            resolved_num_items = infer_num_items(sequences)

    inferred_minimum = infer_num_items(sequences)
    if resolved_num_items < inferred_minimum:
        raise ValueError(
            f"num_items={resolved_num_items} is smaller than max item id + 1 "
            f"from sequences ({inferred_minimum})."
        )
    return resolved_num_items


def build_positive_contexts_by_center(pairs: list[Pair]) -> dict[int, set[int]]:
    contexts: dict[int, set[int]] = {}
    for center_item, context_item in pairs:
        contexts.setdefault(center_item, set()).add(context_item)
    return contexts


def train_item2vec(
    sequences: list[list[int]],
    num_items: int,
    embedding_dim: int,
    window_size: int,
    num_negatives: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, list[float], int]:
    set_seed(seed)

    positive_pairs = build_positive_pairs(sequences, window_size=window_size)
    if not positive_pairs:
        raise ValueError("No positive pairs were generated.")

    positive_contexts_by_center = build_positive_contexts_by_center(positive_pairs)
    dataset = Item2VecDataset(
        positive_pairs=positive_pairs,
        positive_contexts_by_center=positive_contexts_by_center,
        num_items=num_items,
        num_negatives=num_negatives,
        seed=seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    model = SkipGramNegSampling(num_items=num_items, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        running_loss = 0.0
        step_count = 0

        progress = tqdm(
            dataloader,
            desc=f"item2vec epoch {epoch + 1}/{epochs}",
        )
        for batch in progress:
            center_items = batch["center"].to(device)
            positive_contexts = batch["positive"].to(device)
            negative_items = batch["negatives"].to(device)

            loss = model(center_items, positive_contexts, negative_items)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            step_count += 1
            progress.set_postfix(loss=running_loss / step_count)

        epoch_losses.append(running_loss / max(1, step_count))

    return model.item_embeddings().cpu(), epoch_losses, len(positive_pairs)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sequences = load_sequences(args.sequences)
    num_items = resolve_num_items(
        sequences_path=args.sequences,
        sequences=sequences,
        num_items=args.num_items,
        data_meta=args.data_meta,
    )
    item_embeddings, epoch_losses, num_positive_pairs = train_item2vec(
        sequences=sequences,
        num_items=num_items,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        num_negatives=args.num_negatives,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device=device,
    )

    output = Path(args.output)
    ensure_dir(output.parent)
    torch.save(item_embeddings, output)

    meta = {
        "sequences_path": str(args.sequences),
        "output_path": str(args.output),
        "num_items": int(item_embeddings.shape[0]),
        "embedding_dim": int(item_embeddings.shape[1]),
        "data_meta": str(args.data_meta) if args.data_meta is not None else None,
        "window_size": args.window_size,
        "num_negatives": args.num_negatives,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "device": str(device),
        "num_positive_pairs": num_positive_pairs,
        "epoch_losses": epoch_losses,
    }
    save_json(meta, output.with_name("item_embedding_meta.json"))
    print(meta)


if __name__ == "__main__":
    main()
