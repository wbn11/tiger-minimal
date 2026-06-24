import argparse
from pathlib import Path

from tiger_min.data.adapters import ProcessedSequenceAdapter, RawAmazonAdapter, SequenceCorpus
from tiger_min.data.splits import build_leave_one_out_splits, save_splits
from tiger_min.utils import save_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build contiguous item-id sequences and leave-one-out splits."
    )
    parser.add_argument("--source", choices=["processed", "raw_amazon"], default="raw_amazon")
    parser.add_argument("--input", default="data/raw/reviews_Beauty_5.json.gz")
    parser.add_argument("--output", default="data/processed/beauty_5k")
    parser.add_argument("--min-sequence-length", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=5000)
    parser.add_argument("--max-history-length", type=int, default=20)
    return parser


def build_item2vec_sequences(sequences: list[list[int]]) -> list[list[int]]:
    """Keep only training-visible history for item2vec to avoid valid/test leakage."""
    return [seq[:-2] for seq in sequences if len(seq[:-2]) >= 2]


def save_corpus(corpus: SequenceCorpus, processed_dir: str | Path) -> None:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    item2vec_sequences = build_item2vec_sequences(corpus.sequences)
    save_json(item2vec_sequences, processed / "item2vec_sequences.json")
    save_json(corpus.user2id, processed / "user2id.json")
    save_json(corpus.item2id, processed / "item2id.json")
    save_json(
        {
            "source": corpus.source,
            "num_users": corpus.num_users,
            "num_items": corpus.num_items,
            "num_interactions": sum(len(seq) for seq in corpus.sequences),
            "item2vec_sequences_path": str(processed / "item2vec_sequences.json"),
            "num_item2vec_sequences": len(item2vec_sequences),
            "num_item2vec_interactions": sum(len(seq) for seq in item2vec_sequences),
        },
        processed / "data_meta.json",
    )


def build_corpus(
    source: str,
    input_path: str | Path,
    min_sequence_length: int,
    max_users: int | None,
) -> SequenceCorpus:
    if source == "raw_amazon":
        adapter = RawAmazonAdapter(
            raw_path=input_path,
            min_sequence_length=min_sequence_length,
            max_users=max_users,
        )
    elif source == "processed":
        adapter = ProcessedSequenceAdapter(
            sequences_path=input_path,
            min_sequence_length=min_sequence_length,
        )
    else:
        raise ValueError(f"Unknown source: {source}")

    return adapter.load_corpus()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    processed_dir = Path(args.output)
    corpus = build_corpus(
        source=args.source,
        input_path=args.input,
        min_sequence_length=args.min_sequence_length,
        max_users=args.max_users,
    )
    save_corpus(corpus, processed_dir)

    splits = build_leave_one_out_splits(
        corpus=corpus,
        min_sequence_length=args.min_sequence_length,
        train_all_prefixes=True,
        max_history_length=args.max_history_length,
    )
    save_splits(splits, processed_dir / "splits.pt")

    item2vec_sequences = build_item2vec_sequences(corpus.sequences)
    summary = {
        "num_users": corpus.num_users,
        "num_items": corpus.num_items,
        "num_item2vec_sequences": len(item2vec_sequences),
        "num_item2vec_interactions": sum(len(seq) for seq in item2vec_sequences),
        "num_train_samples": len(splits.train),
        "num_valid_samples": len(splits.valid),
        "num_test_samples": len(splits.test),
    }
    save_json(summary, processed_dir / "split_meta.json")
    print(summary)


if __name__ == "__main__":
    main()
