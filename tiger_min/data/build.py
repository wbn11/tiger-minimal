import argparse
from pathlib import Path

from tiger_min.data.adapters import ProcessedSequenceAdapter, RawAmazonAdapter, SequenceCorpus
from tiger_min.data.splits import build_leave_one_out_splits, save_splits
from tiger_min.utils import save_json


def save_corpus(corpus: SequenceCorpus, processed_dir: str | Path) -> None:
    processed = Path(processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    save_json(corpus.sequences, processed / "sequences.json")
    save_json(corpus.user2id, processed / "user2id.json")
    save_json(corpus.item2id, processed / "item2id.json")
    save_json(
        {
            "source": corpus.source,
            "num_users": corpus.num_users,
            "num_items": corpus.num_items,
            "num_interactions": sum(len(seq) for seq in corpus.sequences),
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


def run(
    source: str,
    input_path: str | Path,
    output_dir: str | Path,
    min_sequence_length: int = 5,
    max_users: int | None = None,
    max_history_length: int | None = 20,
) -> dict:
    processed_dir = Path(output_dir)
    corpus = build_corpus(
        source=source,
        input_path=input_path,
        min_sequence_length=min_sequence_length,
        max_users=max_users,
    )
    save_corpus(corpus, processed_dir)

    splits = build_leave_one_out_splits(
        corpus=corpus,
        min_sequence_length=min_sequence_length,
        train_all_prefixes=True,
        max_history_length=max_history_length,
    )
    save_splits(splits, processed_dir / "splits.pt")

    summary = {
        "num_users": corpus.num_users,
        "num_items": corpus.num_items,
        "num_train_samples": len(splits.train),
        "num_valid_samples": len(splits.valid),
        "num_test_samples": len(splits.test),
    }
    save_json(summary, processed_dir / "split_meta.json")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build contiguous item-id sequences and leave-one-out splits."
    )
    parser.add_argument("--source", choices=["processed", "raw_amazon"], default="processed")
    parser.add_argument("--input", default="data/raw/toy_sequences.json")
    parser.add_argument("--output", default="data/processed/toy")
    parser.add_argument("--min-sequence-length", type=int, default=5)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--max-history-length", type=int, default=20)
    args = parser.parse_args()

    summary = run(
        source=args.source,
        input_path=args.input,
        output_dir=args.output,
        min_sequence_length=args.min_sequence_length,
        max_users=args.max_users,
        max_history_length=args.max_history_length,
    )
    print(summary)


if __name__ == "__main__":
    main()
