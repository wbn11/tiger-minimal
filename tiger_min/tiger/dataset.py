"""TIGER 训练数据集。

把 next-item 样本转换成 Encoder-Decoder Transformer 需要的 token：
encoder 输入用户历史，decoder 输入 BOS+目标前缀，labels 是目标 token+EOS。
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tiger_min.data.splits import NextItemSample, load_splits
from tiger_min.tiger.tokenizer import TigerTokenizer


@dataclass
class TigerDatasets:
    train: "TigerNextItemDataset"
    valid: "TigerNextItemDataset"
    test: "TigerNextItemDataset"


class TigerNextItemDataset(Dataset):
    """Turn next-item samples into TIGER encoder-decoder token samples."""

    def __init__(self, samples: list[NextItemSample], tokenizer: TigerTokenizer) -> None:
        if not samples:
            raise ValueError("samples cannot be empty.")
        self.samples = samples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        encoded = self.tokenizer.encode_sample(
            history=sample.history,
            target_item=sample.target,
        )
        return {
            "user_id": sample.user_id,
            "target_item": sample.target,
            "encoder_input": encoded["encoder_input"],
            "decoder_input": encoded["decoder_input"],
            "decoder_labels": encoded["decoder_labels"],
        }


def load_tiger_datasets(
    splits_path: str | Path,
    semantic_ids_path: str | Path,
    codebook_size: int,
    num_quantizer_layers: int,
) -> TigerDatasets:
    splits = load_splits(splits_path)
    tokenizer = TigerTokenizer.from_file(
        semantic_ids_path,
        codebook_size=codebook_size,
        num_quantizer_layers=num_quantizer_layers,
    )
    return TigerDatasets(
        train=TigerNextItemDataset(splits.train, tokenizer),
        valid=TigerNextItemDataset(splits.valid, tokenizer),
        test=TigerNextItemDataset(splits.test, tokenizer),
    )


def collate_tiger_batch(batch: list[dict]) -> dict[str, torch.LongTensor]:
    if not batch:
        raise ValueError("batch cannot be empty.")

    pad_token_id = 0

    # Encoder histories have variable length, so they are padded within a batch.
    encoder_input_ids = _pad_sequences(
        [example["encoder_input"] for example in batch],
        pad_value=pad_token_id,
    )
    # Decoder sequences have fixed semantic_id_length + 1, so direct tensorization is safe.
    decoder_input_ids = torch.tensor(
        [example["decoder_input"] for example in batch],
        dtype=torch.long,
    )
    labels = torch.tensor(
        [example["decoder_labels"] for example in batch],
        dtype=torch.long,
    )

    return {
        "user_ids": torch.tensor([example["user_id"] for example in batch], dtype=torch.long),
        "target_items": torch.tensor(
            [example["target_item"] for example in batch],
            dtype=torch.long,
        ),
        "encoder_input_ids": encoder_input_ids,
        # 1 means real token, 0 means padding for the Transformer key padding mask.
        "encoder_attention_mask": (encoder_input_ids != pad_token_id).long(),
        "decoder_input_ids": decoder_input_ids,
        "labels": labels,
    }


def _pad_sequences(sequences: list[list[int]], pad_value: int) -> torch.LongTensor:
    max_length = max(len(sequence) for sequence in sequences)
    padded = [
        sequence + [pad_value] * (max_length - len(sequence))
        for sequence in sequences
    ]
    return torch.tensor(padded, dtype=torch.long)
