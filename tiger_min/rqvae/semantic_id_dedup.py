from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SemanticIdDedupResult:
    semantic_ids: torch.LongTensor
    meta: dict[str, Any]


def _validate_semantic_ids(base_semantic_ids: torch.Tensor) -> None:
    if base_semantic_ids.ndim != 2:
        raise ValueError("base_semantic_ids must be a 2D tensor.")
    if base_semantic_ids.numel() == 0:
        raise ValueError("base_semantic_ids cannot be empty.")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if base_semantic_ids.dtype not in integer_dtypes:
        raise TypeError("base_semantic_ids must contain integer ids.")


def semantic_id_rows_to_tuples(semantic_ids: torch.Tensor) -> list[tuple[int, ...]]:
    _validate_semantic_ids(semantic_ids)
    return [tuple(int(x) for x in row.tolist()) for row in semantic_ids.cpu()]


def deduplicate_semantic_ids(base_semantic_ids: torch.LongTensor) -> SemanticIdDedupResult:
    """Append a suffix code so every item has a unique semantic id."""
    _validate_semantic_ids(base_semantic_ids)

    base_tuples = semantic_id_rows_to_tuples(base_semantic_ids)
    next_suffix_by_base_id: dict[tuple[int, ...], int] = {}
    group_size_by_base_id: dict[tuple[int, ...], int] = {}
    suffixes: list[int] = []

    for base_id in base_tuples:
        suffix = next_suffix_by_base_id.get(base_id, 0)
        suffixes.append(suffix)
        next_suffix_by_base_id[base_id] = suffix + 1
        group_size_by_base_id[base_id] = group_size_by_base_id.get(base_id, 0) + 1

    suffix_tensor = torch.tensor(
        suffixes,
        dtype=torch.long,
        device=base_semantic_ids.device,
    ).unsqueeze(1)
    semantic_ids = torch.cat([base_semantic_ids.long(), suffix_tensor], dim=1)

    collided_group_sizes = [
        group_size for group_size in group_size_by_base_id.values() if group_size > 1
    ]
    num_items = int(base_semantic_ids.shape[0])
    num_collided_items = int(sum(collided_group_sizes))

    meta = {
        "num_items": num_items,
        "base_semantic_id_length": int(base_semantic_ids.shape[1]),
        "semantic_id_length": int(semantic_ids.shape[1]),
        "num_unique_base_semantic_ids": len(group_size_by_base_id),
        "num_collided_base_semantic_ids": len(collided_group_sizes),
        "num_collided_items": num_collided_items,
        "max_collision_group_size": max(collided_group_sizes, default=1),
        "collision_rate": num_collided_items / num_items,
        "deduplication": "append_suffix",
    }
    return SemanticIdDedupResult(semantic_ids=semantic_ids, meta=meta)
