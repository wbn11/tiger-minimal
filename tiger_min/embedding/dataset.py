"""item2vec 训练数据集。

从用户物品序列中构造窗口共现正样本，并为每个正样本随机采样负物品，
供 Skip-Gram Negative Sampling 模型训练 item embedding。
"""

import random
from collections.abc import Iterable

import torch
from torch.utils.data import Dataset


Pair = tuple[int, int]


def iter_positive_pairs(sequence: list[int], window_size: int) -> Iterable[Pair]:
    """Yield directed (center, context) pairs from one item sequence."""
    if window_size <= 0:
        raise ValueError("window_size must be positive.")

    for center_pos, center_item in enumerate(sequence):
        left = max(0, center_pos - window_size)
        right = min(len(sequence), center_pos + window_size + 1)

        for context_pos in range(left, right):
            if context_pos == center_pos:
                continue

            context_item = sequence[context_pos]
            if context_item == center_item:
                continue

            yield center_item, context_item


def build_positive_pairs(sequences: list[list[int]], window_size: int) -> list[Pair]:
    """Build all directed positive pairs from multiple user sequences."""
    pairs: list[Pair] = []
    for sequence in sequences:
        pairs.extend(iter_positive_pairs(sequence, window_size))
    return pairs


def sample_negative_items(
    center_item: int,
    positive_contexts: set[int],
    num_items: int,
    num_negatives: int,
    rng: random.Random,
) -> list[int]:
    """Sample negative item ids excluding the center item and positive contexts."""
    if num_items <= 0:
        raise ValueError("num_items must be positive.")
    if num_negatives < 0:
        raise ValueError("num_negatives cannot be negative.")

    excluded = set(positive_contexts)
    excluded.add(center_item)

    # Exclude known positive contexts so sampled negatives are not local positives.
    candidates = [item_id for item_id in range(num_items) if item_id not in excluded]
    if len(candidates) < num_negatives:
        raise ValueError("Not enough candidate items to sample negatives.")

    return rng.sample(candidates, k=num_negatives)


class Item2VecDataset(Dataset):
    """Dataset that turns positive pairs into item2vec training samples."""

    def __init__(
        self,
        positive_pairs: list[Pair],
        positive_contexts_by_center: dict[int, set[int]],
        num_items: int,
        num_negatives: int,
        seed: int = 42,
    ) -> None:
        if not positive_pairs:
            raise ValueError("positive_pairs cannot be empty.")
        if num_items <= 0:
            raise ValueError("num_items must be positive.")
        if num_negatives < 0:
            raise ValueError("num_negatives cannot be negative.")

        self.positive_pairs = positive_pairs
        self.positive_contexts_by_center = positive_contexts_by_center
        self.num_items = num_items
        self.num_negatives = num_negatives
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.positive_pairs)

    def __getitem__(self, index: int) -> dict[str, torch.LongTensor]:
        center_item, positive_context = self.positive_pairs[index]
        # Negatives are sampled lazily so each epoch can see fresh random examples.
        negative_items = sample_negative_items(
            center_item=center_item,
            positive_contexts=self.positive_contexts_by_center[center_item],
            num_items=self.num_items,
            num_negatives=self.num_negatives,
            rng=self.rng,
        )

        return {
            "center": torch.tensor(center_item, dtype=torch.long),
            "positive": torch.tensor(positive_context, dtype=torch.long),
            "negatives": torch.tensor(negative_items, dtype=torch.long),
        }
