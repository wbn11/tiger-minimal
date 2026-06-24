import argparse
from pathlib import Path

import torch
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
    parser.add_argument("--sequences", default="data/processed/toy/sequences.json")
    parser.add_argument("--output", default="data/processed/toy/item_embeddings.pt")
    parser.add_argument("--embedding-dim", type=int, default=16)
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


def build_positive_contexts_by_center(pairs: list[Pair]) -> dict[int, set[int]]:
    contexts: dict[int, set[int]] = {}
    for center_item, context_item in pairs:
        contexts.setdefault(center_item, set()).add(context_item)
    return contexts


def train_item2vec(
    sequences: list[list[int]],
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

    num_items = infer_num_items(sequences)
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
    for _ in range(epochs):
        running_loss = 0.0
        step_count = 0

        for batch in dataloader:
            center_items = batch["center"].to(device)
            positive_contexts = batch["positive"].to(device)
            negative_items = batch["negatives"].to(device)

            loss = model(center_items, positive_contexts, negative_items)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())
            step_count += 1

        epoch_losses.append(running_loss / max(1, step_count))

    return model.item_embeddings().cpu(), epoch_losses, len(positive_pairs)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sequences = load_sequences(args.sequences)
    item_embeddings, epoch_losses, num_positive_pairs = train_item2vec(
        sequences=sequences,
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
