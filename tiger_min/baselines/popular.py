import argparse
import math
from collections import Counter
from pathlib import Path

from tiger_min.data.splits import NextItemSample, load_splits
from tiger_min.utils import load_json, save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a global popular-item baseline.")
    parser.add_argument("--processed-dir", default="data/processed/beauty")
    parser.add_argument("--sequences", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--split", choices=["valid", "test", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", default=None)
    return parser


def load_item2vec_sequences(path: str | Path) -> list[list[int]]:
    sequences = load_json(path)
    if not isinstance(sequences, list):
        raise ValueError("item2vec sequences must be a list.")
    return [[int(item) for item in sequence] for sequence in sequences]


def build_popular_ranking(
    sequences: list[list[int]],
    top_k: int,
) -> list[int]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    counter = Counter(
        item
        for sequence in sequences
        for item in sequence
    )
    if not counter:
        raise ValueError("Cannot build popular baseline from empty sequences.")
    return [item for item, _ in counter.most_common(top_k)]


def evaluate_ranking(
    ranked_items: list[int],
    samples: list[NextItemSample],
    top_k: int,
) -> dict:
    cutoffs = [k for k in [1, 5, 10, 20] if k <= top_k]
    metric_sums = {f"hr@{k}": 0.0 for k in cutoffs}
    metric_sums.update({f"ndcg@{k}": 0.0 for k in cutoffs})

    for sample in samples:
        target = int(sample.target)
        for k in cutoffs:
            top_items = ranked_items[:k]
            if target in top_items:
                rank = top_items.index(target) + 1
                metric_sums[f"hr@{k}"] += 1.0
                metric_sums[f"ndcg@{k}"] += 1.0 / math.log2(rank + 1)

    num_examples = len(samples)
    metrics = {
        "num_examples": num_examples,
        "top_k": top_k,
        "avg_valid_predictions": len(ranked_items[:top_k]),
        "valid_prediction_rate": len(ranked_items[:top_k]) / top_k,
    }
    for key, value in metric_sums.items():
        metrics[key] = value / max(1, num_examples)
    return metrics


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    sequences_path = (
        Path(args.sequences)
        if args.sequences is not None
        else processed_dir / "item2vec_sequences.json"
    )
    splits_path = (
        Path(args.splits)
        if args.splits is not None
        else processed_dir / "splits.pt"
    )
    output_path = (
        Path(args.output)
        if args.output is not None
        else processed_dir / "popular_baseline.json"
    )

    sequences = load_item2vec_sequences(sequences_path)
    splits = load_splits(splits_path)
    ranked_items = build_popular_ranking(sequences, top_k=args.top_k)

    split_names = ["valid", "test"] if args.split == "all" else [args.split]
    result = {
        "processed_dir": str(processed_dir),
        "sequences_path": str(sequences_path),
        "splits_path": str(splits_path),
        "ranking_rule": "global top-K from train-visible item2vec sequences",
        "top_items": ranked_items,
        "metrics": {},
    }
    for split_name in split_names:
        samples = getattr(splits, split_name)
        result["metrics"][split_name] = evaluate_ranking(
            ranked_items=ranked_items,
            samples=samples,
            top_k=args.top_k,
        )

    save_json(result, output_path)
    print(result)


if __name__ == "__main__":
    main()
