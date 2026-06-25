import argparse
from pathlib import Path

from tiger_min.data.splits import SequenceSplits, load_splits
from tiger_min.utils import load_json, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report dataset and target coverage stats.")
    parser.add_argument("--processed-dir", default="data/processed/beauty")
    parser.add_argument("--output", default=None)
    return parser


def load_item2vec_sequences(processed_dir: str | Path) -> list[list[int]]:
    path = Path(processed_dir) / "item2vec_sequences.json"
    sequences = load_json(path)
    if not isinstance(sequences, list):
        raise ValueError("item2vec_sequences.json must contain a list.")
    return [[int(item) for item in sequence] for sequence in sequences]


def target_coverage(samples, train_visible_items: set[int]) -> float:
    if not samples:
        return 0.0
    covered = sum(int(sample.target) in train_visible_items for sample in samples)
    return covered / len(samples)


def build_dataset_stats(
    processed_dir: str | Path,
    splits: SequenceSplits,
    item2vec_sequences: list[list[int]],
) -> dict:
    train_visible_items = {
        int(item)
        for sequence in item2vec_sequences
        for item in sequence
    }
    item2vec_interactions = sum(len(sequence) for sequence in item2vec_sequences)
    valid_target_coverage = target_coverage(splits.valid, train_visible_items)
    test_target_coverage = target_coverage(splits.test, train_visible_items)

    return {
        "processed_dir": str(processed_dir),
        "num_users": splits.num_users,
        "num_items": splits.num_items,
        "num_item2vec_sequences": len(item2vec_sequences),
        "num_item2vec_interactions": item2vec_interactions,
        "num_train_samples": len(splits.train),
        "num_valid_samples": len(splits.valid),
        "num_test_samples": len(splits.test),
        "num_item2vec_unique_items": len(train_visible_items),
        "item2vec_item_coverage": len(train_visible_items) / splits.num_items,
        "valid_target_coverage": valid_target_coverage,
        "test_target_coverage": test_target_coverage,
        "valid_cold_start_target_ratio": 1.0 - valid_target_coverage,
        "test_cold_start_target_ratio": 1.0 - test_target_coverage,
    }


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    splits = load_splits(processed_dir / "splits.pt")
    item2vec_sequences = load_item2vec_sequences(processed_dir)
    stats = build_dataset_stats(
        processed_dir=processed_dir,
        splits=splits,
        item2vec_sequences=item2vec_sequences,
    )

    output = (
        Path(args.output)
        if args.output is not None
        else processed_dir / "dataset_stats.json"
    )
    save_json(stats, output)
    print(stats)


if __name__ == "__main__":
    main()
