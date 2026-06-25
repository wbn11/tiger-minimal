from dataclasses import dataclass
import warnings

import torch
from torch import nn
from torch.nn import functional as F

warnings.filterwarnings(
    "ignore",
    message="The PyTorch API of nested tensors is in prototype stage.*",
    category=UserWarning,
)


@dataclass
class TigerModelOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TigerTransformer(nn.Module):
    """Minimal encoder-decoder Transformer for TIGER semantic token generation."""

    def __init__(
        self,
        vocab_size: int,
        max_encoder_length: int,
        max_decoder_length: int,
        d_model: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if max_encoder_length <= 0:
            raise ValueError("max_encoder_length must be positive.")
        if max_decoder_length <= 0:
            raise ValueError("max_decoder_length must be positive.")
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.vocab_size = vocab_size
        self.max_encoder_length = max_encoder_length
        self.max_decoder_length = max_decoder_length
        self.d_model = d_model
        self.pad_token_id = pad_token_id

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model,
            padding_idx=pad_token_id,
        )
        self.encoder_position_embedding = nn.Embedding(max_encoder_length, d_model)
        self.decoder_position_embedding = nn.Embedding(max_decoder_length, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        encoder_input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
        labels: torch.LongTensor | None = None,
    ) -> TigerModelOutput:
        self._validate_inputs(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_input_ids=decoder_input_ids,
        )

        memory, memory_key_padding_mask = self.encode(
            encoder_input_ids=encoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
        )
        logits = self.decode(
            decoder_input_ids=decoder_input_ids,
            memory=memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        loss = None
        if labels is not None:
            if labels.shape != decoder_input_ids.shape:
                raise ValueError("labels must have the same shape as decoder_input_ids.")
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
            )

        return TigerModelOutput(logits=logits, loss=loss)

    def encode(
        self,
        encoder_input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if encoder_input_ids.ndim != 2:
            raise ValueError("encoder_input_ids must be a 2D tensor.")
        if encoder_attention_mask.shape != encoder_input_ids.shape:
            raise ValueError(
                "encoder_attention_mask must have the same shape as encoder_input_ids."
            )

        src = self._embed_tokens(
            input_ids=encoder_input_ids,
            position_embedding=self.encoder_position_embedding,
            max_length=self.max_encoder_length,
        )
        memory_key_padding_mask = encoder_attention_mask == 0
        memory = self.transformer.encoder(
            src,
            src_key_padding_mask=memory_key_padding_mask,
        )
        return memory, memory_key_padding_mask

    def decode(
        self,
        decoder_input_ids: torch.LongTensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        if decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must be a 2D tensor.")

        tgt = self._embed_tokens(
            input_ids=decoder_input_ids,
            position_embedding=self.decoder_position_embedding,
            max_length=self.max_decoder_length,
        )

        decoder_length = decoder_input_ids.shape[1]
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_length,
            device=decoder_input_ids.device,
        )
        hidden = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_projection(hidden)

    def _embed_tokens(
        self,
        input_ids: torch.LongTensor,
        position_embedding: nn.Embedding,
        max_length: int,
    ) -> torch.Tensor:
        sequence_length = input_ids.shape[1]
        if sequence_length > max_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds max length {max_length}."
            )

        positions = torch.arange(sequence_length, device=input_ids.device).unsqueeze(0)
        return self.token_embedding(input_ids) + position_embedding(positions)

    def _validate_inputs(
        self,
        encoder_input_ids: torch.LongTensor,
        encoder_attention_mask: torch.LongTensor,
        decoder_input_ids: torch.LongTensor,
    ) -> None:
        if encoder_input_ids.ndim != 2:
            raise ValueError("encoder_input_ids must be a 2D tensor.")
        if encoder_attention_mask.shape != encoder_input_ids.shape:
            raise ValueError(
                "encoder_attention_mask must have the same shape as encoder_input_ids."
            )
        if decoder_input_ids.ndim != 2:
            raise ValueError("decoder_input_ids must be a 2D tensor.")
        if torch.max(encoder_input_ids).item() >= self.vocab_size:
            raise ValueError("encoder_input_ids contains token id outside vocab_size.")
        if torch.max(decoder_input_ids).item() >= self.vocab_size:
            raise ValueError("decoder_input_ids contains token id outside vocab_size.")
