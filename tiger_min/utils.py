import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def project_path(path: str | Path) -> Path:
    return Path(path)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | Path) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_contiguous_zero_based(ids: list[int]) -> bool:
    if not ids:
        return True
    return sorted(ids) == list(range(max(ids) + 1))

