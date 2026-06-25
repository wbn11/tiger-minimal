from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class TigerSpecialTokens:
    pad: int = 0
    bos: int = 1
    eos: int = 2

    @property
    def offset(self) -> int:
        return 3


class TigerTokenizer:
    """Convert item ids into TIGER semantic token sequences."""

    def __init__(
        self,
        semantic_ids: torch.LongTensor,
        codebook_size: int,
        num_quantizer_layers: int,
        special_tokens: TigerSpecialTokens | None = None,
    ) -> None:
        if semantic_ids.ndim != 2:
            raise ValueError("semantic_ids must be a 2D tensor: [num_items, sem_id_len].")
        if semantic_ids.numel() == 0:
            raise ValueError("semantic_ids cannot be empty.")
        if semantic_ids.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError("semantic_ids must contain integer ids.")

        self.semantic_ids = semantic_ids.long().cpu()
        self.special_tokens = special_tokens or TigerSpecialTokens()
        self.num_items = int(self.semantic_ids.shape[0])
        self.semantic_id_length = int(self.semantic_ids.shape[1])
        if torch.any(self.semantic_ids < 0).item():
            raise ValueError("semantic_ids cannot contain negative ids.")
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive.")
        if num_quantizer_layers <= 0:
            raise ValueError("num_quantizer_layers must be positive.")
        if self.semantic_id_length != num_quantizer_layers + 1:
            raise ValueError(
                "semantic_id_length must equal num_quantizer_layers + 1 "
                "because the last position is the dedup suffix."
            )

        self.codebook_size = codebook_size
        self.num_quantizer_layers = num_quantizer_layers
        self.position_vocab_sizes = self._build_position_vocab_sizes()
        self.position_offsets = self._build_position_offsets()
        self.vocab_size = self.position_offsets[-1] + self.position_vocab_sizes[-1]

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        codebook_size: int,
        num_quantizer_layers: int,
    ) -> "TigerTokenizer":
        semantic_ids = torch.load(path, map_location="cpu")
        if not isinstance(semantic_ids, torch.Tensor):
            raise TypeError("semantic_ids file must contain a torch.Tensor.")
        return cls(
            semantic_ids=semantic_ids,
            codebook_size=codebook_size,
            num_quantizer_layers=num_quantizer_layers,
        )

    def item_to_semantic_tokens(self, item_id: int) -> list[int]:
        self._validate_item_id(item_id)
        return [
            self._code_to_token(position, int(code))
            for position, code in enumerate(self.semantic_ids[item_id].tolist())
        ]

    def history_to_tokens(self, history: list[int]) -> list[int]:
        tokens: list[int] = []
        for item_id in history:
            tokens.extend(self.item_to_semantic_tokens(item_id))
        return tokens

    def target_to_decoder_input(self, target_item: int) -> list[int]:
        return [self.special_tokens.bos] + self.item_to_semantic_tokens(target_item)

    def target_to_decoder_labels(self, target_item: int) -> list[int]:
        return self.item_to_semantic_tokens(target_item) + [self.special_tokens.eos]

    def encode_sample(self, history: list[int], target_item: int) -> dict[str, list[int]]:
        return {
            "encoder_input": self.history_to_tokens(history),
            "decoder_input": self.target_to_decoder_input(target_item),
            "decoder_labels": self.target_to_decoder_labels(target_item),
        }

    def semantic_tokens_to_id(self, tokens: list[int]) -> tuple[int, ...]:
        if len(tokens) != self.semantic_id_length:
            raise ValueError(
                f"Expected {self.semantic_id_length} semantic tokens, got {len(tokens)}."
            )
        return tuple(
            self.token_to_code(position, token)
            for position, token in enumerate(tokens)
        )

    def token_to_code(self, position: int, token: int) -> int:
        if position < 0 or position >= self.semantic_id_length:
            raise IndexError(
                f"position {position} is outside [0, {self.semantic_id_length})."
            )

        offset = self.position_offsets[position]
        vocab_size = self.position_vocab_sizes[position]
        if token < offset or token >= offset + vocab_size:
            raise ValueError(
                f"token {token} is outside position {position} token range "
                f"[{offset}, {offset + vocab_size})."
            )
        return token - offset

    def _validate_item_id(self, item_id: int) -> None:
        if item_id < 0 or item_id >= self.num_items:
            raise IndexError(f"item_id {item_id} is outside [0, {self.num_items}).")

    def _build_position_vocab_sizes(self) -> list[int]:
        rq_codes = self.semantic_ids[:, : self.num_quantizer_layers]
        if torch.any(rq_codes >= self.codebook_size).item():
            raise ValueError("RQ code ids must be smaller than codebook_size.")

        suffix_codes = self.semantic_ids[:, self.num_quantizer_layers]
        suffix_vocab_size = int(torch.max(suffix_codes).item()) + 1
        return [self.codebook_size] * self.num_quantizer_layers + [suffix_vocab_size]

    def _build_position_offsets(self) -> list[int]:
        offsets: list[int] = []
        next_offset = self.special_tokens.offset
        for vocab_size in self.position_vocab_sizes:
            offsets.append(next_offset)
            next_offset += vocab_size
        return offsets

    def _code_to_token(self, position: int, code: int) -> int:
        if position < 0 or position >= self.semantic_id_length:
            raise IndexError(
                f"position {position} is outside [0, {self.semantic_id_length})."
            )
        if code < 0 or code >= self.position_vocab_sizes[position]:
            raise ValueError(
                f"code {code} is outside position {position} vocab size "
                f"{self.position_vocab_sizes[position]}."
            )
        return self.position_offsets[position] + code
