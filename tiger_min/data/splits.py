from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tiger_min.data.adapters import SequenceCorpus


@dataclass
class NextItemSample:
    user_id: int
    history: list[int]
    target: int


@dataclass
class SequenceSplits:
    train: list[NextItemSample]
    valid: list[NextItemSample]
    test: list[NextItemSample]
    num_users: int
    num_items: int

    def to_dict(self) -> dict:
        return {
            "train": [asdict(x) for x in self.train],
            "valid": [asdict(x) for x in self.valid],
            "test": [asdict(x) for x in self.test],
            "num_users": self.num_users,
            "num_items": self.num_items,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SequenceSplits":
        return cls(
            train=[NextItemSample(**x) for x in data["train"]],
            valid=[NextItemSample(**x) for x in data["valid"]],
            test=[NextItemSample(**x) for x in data["test"]],
            num_users=int(data["num_users"]),
            num_items=int(data["num_items"]),
        )


def truncate_history(history: list[int], max_history_length: int | None) -> list[int]:
    if max_history_length is None:
        return history
    return history[-max_history_length:]


def build_leave_one_out_splits(
    corpus: SequenceCorpus,
    min_sequence_length: int = 5,
    train_all_prefixes: bool = True,
    max_history_length: int | None = None,
) -> SequenceSplits:
    train: list[NextItemSample] = []
    valid: list[NextItemSample] = []
    test: list[NextItemSample] = []

    for user_id, seq in enumerate(corpus.sequences):
        if len(seq) < min_sequence_length:
            continue

        valid.append(
            NextItemSample(
                user_id=user_id,
                history=truncate_history(seq[:-2], max_history_length),
                target=seq[-2],
            )
        )
        test.append(
            NextItemSample(
                user_id=user_id,
                history=truncate_history(seq[:-1], max_history_length),
                target=seq[-1],
            )
        )

        train_end = len(seq) - 2
        if train_all_prefixes:
            target_positions = range(1, train_end)
        else:
            target_positions = [train_end - 1]

        for target_pos in target_positions:
            train.append(
                NextItemSample(
                    user_id=user_id,
                    history=truncate_history(seq[:target_pos], max_history_length),
                    target=seq[target_pos],
                )
            )

    return SequenceSplits(
        train=train,
        valid=valid,
        test=test,
        num_users=corpus.num_users,
        num_items=corpus.num_items,
    )


def save_splits(splits: SequenceSplits, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(splits.to_dict(), target)


def load_splits(path: str | Path) -> SequenceSplits:
    data = torch.load(path, map_location="cpu", weights_only=False)
    return SequenceSplits.from_dict(data)
