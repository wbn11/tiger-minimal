import gzip
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class SequenceCorpus:
    sequences: list[list[int]]
    user2id: dict[str, int]
    item2id: dict[str, int]
    source: str

    @property
    def num_users(self) -> int:
        return len(self.sequences)

    @property
    def num_items(self) -> int:
        return len(self.item2id)


class SequenceAdapter(ABC):
    @abstractmethod
    def load_corpus(self) -> SequenceCorpus:
        raise NotImplementedError

    def load_sequences(self) -> list[list[int]]:
        return self.load_corpus().sequences


class RawAmazonAdapter(SequenceAdapter):
    """Load Amazon review JSON lines and normalize them into integer sequences."""

    def __init__(
        self,
        raw_path: str | Path,
        min_sequence_length: int = 5,
        max_users: int | None = None,
        field_user: str = "reviewerID",
        field_item: str = "asin",
        field_time: str = "unixReviewTime",
    ) -> None:
        self.raw_path = Path(raw_path)
        self.min_sequence_length = min_sequence_length
        self.max_users = max_users
        self.field_user = field_user
        self.field_item = field_item
        self.field_time = field_time

    def load_corpus(self) -> SequenceCorpus:
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Raw Amazon file not found: {self.raw_path}")

        events_by_user: dict[str, list[tuple[int, str]]] = {}
        with gzip.open(self.raw_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                user_key = str(row[self.field_user])
                item_key = str(row[self.field_item])
                timestamp = int(row.get(self.field_time, 0))
                events_by_user.setdefault(user_key, []).append((timestamp, item_key))

        user2id: dict[str, int] = {}
        item2id: dict[str, int] = {}
        sequences: list[list[int]] = []

        for user_key in sorted(events_by_user.keys()):
            events = sorted(events_by_user[user_key], key=lambda x: x[0])
            item_keys = [item_key for _, item_key in events]
            if len(item_keys) < self.min_sequence_length:
                continue

            user2id[user_key] = len(user2id)
            sequence: list[int] = []
            for item_key in item_keys:
                if item_key not in item2id:
                    item2id[item_key] = len(item2id)
                sequence.append(item2id[item_key])
            sequences.append(sequence)

            if self.max_users is not None and len(sequences) >= self.max_users:
                break

        return SequenceCorpus(
            sequences=sequences,
            user2id=user2id,
            item2id=item2id,
            source=str(self.raw_path),
        )


class ProcessedSequenceAdapter(SequenceAdapter):
    """Load processed sequences from common JSON formats and remap ids."""

    def __init__(
        self,
        sequences_path: str | Path,
        min_sequence_length: int = 5,
    ) -> None:
        self.sequences_path = Path(sequences_path)
        self.min_sequence_length = min_sequence_length

    def load_corpus(self) -> SequenceCorpus:
        with open(self.sequences_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        external_sequences = self._extract_sequences(raw)
        user2id: dict[str, int] = {}
        item2id: dict[str, int] = {}
        sequences: list[list[int]] = []

        for external_user, external_items in external_sequences:
            if len(external_items) < self.min_sequence_length:
                continue
            user2id[str(external_user)] = len(user2id)
            sequence: list[int] = []
            for item in external_items:
                item_key = str(item)
                if item_key not in item2id:
                    item2id[item_key] = len(item2id)
                sequence.append(item2id[item_key])
            sequences.append(sequence)

        return SequenceCorpus(
            sequences=sequences,
            user2id=user2id,
            item2id=item2id,
            source=str(self.sequences_path),
        )

    def _extract_sequences(self, raw: Any) -> list[tuple[str, list[Any]]]:
        if isinstance(raw, dict):
            return [(str(user), items) for user, items in raw.items()]

        if isinstance(raw, list) and all(isinstance(x, list) for x in raw):
            return [(str(i), items) for i, items in enumerate(raw)]

        if isinstance(raw, list) and all(isinstance(x, dict) for x in raw):
            result = []
            for i, row in enumerate(raw):
                user = row.get("user_id", row.get("user", i))
                items = row.get("items", row.get("sequence"))
                if items is None:
                    raise ValueError("Processed row must contain `items` or `sequence`.")
                result.append((str(user), items))
            return result

        raise ValueError("Unsupported processed sequence JSON format.")


class SemanticIdAdapter(ABC):
    @abstractmethod
    def load_semantic_ids(self) -> torch.LongTensor:
        raise NotImplementedError


class PrecomputedSemanticIdAdapter(SemanticIdAdapter):
    """Load semantic ids from .pt, list JSON, or dict JSON."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load_semantic_ids(self) -> torch.LongTensor:
        if self.path.suffix == ".pt":
            ids = torch.load(self.path, map_location="cpu")
        else:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                ids = [raw[str(i)] for i in range(len(raw))]
            else:
                ids = raw

        tensor = torch.as_tensor(ids, dtype=torch.long)
        if tensor.ndim != 2:
            raise ValueError("semantic_ids must be a 2D tensor: [num_items, sem_id_len].")
        return tensor

