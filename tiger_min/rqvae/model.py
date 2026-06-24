from dataclasses import dataclass
import os
import warnings

import torch
from sklearn.cluster import KMeans
from torch import nn
from torch.nn import functional as F


@dataclass
class KMeansCodebook:
    centroids: torch.Tensor
    assignments: torch.LongTensor


@dataclass
class QuantizerOutput:
    quantized: torch.Tensor
    code_ids: torch.LongTensor
    loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor


@dataclass
class ResidualQuantizerOutput:
    quantized: torch.Tensor
    code_ids: torch.LongTensor
    loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor


@dataclass
class RqvaeOutput:
    reconstructed: torch.Tensor
    latents: torch.Tensor
    quantized: torch.Tensor
    code_ids: torch.LongTensor
    loss: torch.Tensor
    reconstruction_loss: torch.Tensor
    quantizer_loss: torch.Tensor
    codebook_loss: torch.Tensor
    commitment_loss: torch.Tensor


def fit_kmeans_codebook(
    vectors: torch.Tensor,
    codebook_size: int,
    seed: int = 42,
    max_iter: int = 100,
    n_init: int = 10,
) -> KMeansCodebook:
    """Fit KMeans on vectors and return centroids on the original torch device."""
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D tensor.")
    if codebook_size <= 0:
        raise ValueError("codebook_size must be positive.")
    if codebook_size > vectors.shape[0]:
        raise ValueError("codebook_size cannot be larger than number of vectors.")

    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    vectors_np = vectors.detach().cpu().numpy()
    kmeans = KMeans(
        n_clusters=codebook_size,
        init="k-means++",
        n_init=n_init,
        max_iter=max_iter,
        random_state=seed,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Could not find the number of physical cores.*",
            category=UserWarning,
        )
        kmeans.fit(vectors_np)

    centroids = torch.tensor(
        kmeans.cluster_centers_,
        dtype=vectors.dtype,
        device=vectors.device,
    )
    assignments = torch.tensor(
        kmeans.labels_,
        dtype=torch.long,
        device=vectors.device,
    )
    return KMeansCodebook(centroids=centroids, assignments=assignments)


class RqvaeEncoder(nn.Module):
    """Encode item embeddings into the latent space used by quantization."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        if hidden_dim is None:
            self.network = nn.Linear(input_dim, latent_dim)
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, latent_dim),
            )

    def forward(self, item_embeddings: torch.Tensor) -> torch.Tensor:
        if item_embeddings.ndim != 2:
            raise ValueError("item_embeddings must be a 2D tensor.")
        if item_embeddings.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, "
                f"got {item_embeddings.shape[1]}."
            )

        return self.network(item_embeddings)


class VectorQuantizer(nn.Module):
    """Map continuous latent vectors to nearest codebook entries."""

    def __init__(
        self,
        codebook_size: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if codebook_size <= 0:
            raise ValueError("codebook_size must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if commitment_weight < 0:
            raise ValueError("commitment_weight cannot be negative.")

        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.commitment_weight = commitment_weight
        self.codebook = nn.Embedding(codebook_size, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    @torch.no_grad()
    def initialize_codebook_with_kmeans(
        self,
        vectors: torch.Tensor,
        seed: int = 42,
        max_iter: int = 100,
        n_init: int = 10,
    ) -> torch.LongTensor:
        if vectors.ndim != 2:
            raise ValueError("vectors must be a 2D tensor.")
        if vectors.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected vector dimension {self.embedding_dim}, got {vectors.shape[1]}."
            )

        result = fit_kmeans_codebook(
            vectors=vectors,
            codebook_size=self.codebook_size,
            seed=seed,
            max_iter=max_iter,
            n_init=n_init,
        )
        self.codebook.weight.copy_(result.centroids)
        return result.assignments

    def forward(self, latents: torch.Tensor) -> QuantizerOutput:
        if latents.ndim != 2:
            raise ValueError("latents must be a 2D tensor.")
        if latents.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected latent dimension {self.embedding_dim}, got {latents.shape[1]}."
            )

        distances = self._squared_l2_distance(latents, self.codebook.weight)
        code_ids = torch.argmin(distances, dim=1)
        quantized = self.codebook(code_ids)

        codebook_loss = F.mse_loss(quantized, latents.detach())
        commitment_loss = self.commitment_weight * F.mse_loss(
            latents,
            quantized.detach(),
        )
        loss = codebook_loss + commitment_loss

        quantized_st = latents + (quantized - latents).detach()
        return QuantizerOutput(
            quantized=quantized_st,
            code_ids=code_ids,
            loss=loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )

    @staticmethod
    def _squared_l2_distance(vectors: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
        vectors_norm = torch.sum(vectors * vectors, dim=1, keepdim=True)
        codebook_norm = torch.sum(codebook * codebook, dim=1).unsqueeze(0)
        return vectors_norm - 2 * vectors @ codebook.t() + codebook_norm


class ResidualQuantizer(nn.Module):
    """Stack multiple vector quantizers to produce multi-code semantic ids."""

    def __init__(
        self,
        num_layers: int,
        codebook_size: int,
        embedding_dim: int,
        commitment_weight: float = 0.25,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.num_layers = num_layers
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.layers = nn.ModuleList(
            [
                VectorQuantizer(
                    codebook_size=codebook_size,
                    embedding_dim=embedding_dim,
                    commitment_weight=commitment_weight,
                )
                for _ in range(num_layers)
            ]
        )

    @torch.no_grad()
    def initialize_codebooks_with_kmeans(
        self,
        latents: torch.Tensor,
        seed: int = 42,
        max_iter: int = 100,
        n_init: int = 10,
    ) -> torch.LongTensor:
        self._validate_latents(latents)

        residual = latents
        code_ids_by_layer: list[torch.LongTensor] = []

        for layer_index, layer in enumerate(self.layers):
            layer.initialize_codebook_with_kmeans(
                residual,
                seed=seed + layer_index,
                max_iter=max_iter,
                n_init=n_init,
            )
            output = layer(residual)
            code_ids_by_layer.append(output.code_ids)
            residual = residual - output.quantized

        return torch.stack(code_ids_by_layer, dim=1)

    def forward(self, latents: torch.Tensor) -> ResidualQuantizerOutput:
        self._validate_latents(latents)

        residual = latents
        quantized = torch.zeros_like(latents)
        code_ids_by_layer: list[torch.LongTensor] = []
        total_loss = latents.new_tensor(0.0)
        total_codebook_loss = latents.new_tensor(0.0)
        total_commitment_loss = latents.new_tensor(0.0)

        for layer in self.layers:
            output = layer(residual)
            quantized = quantized + output.quantized
            code_ids_by_layer.append(output.code_ids)
            total_loss = total_loss + output.loss
            total_codebook_loss = total_codebook_loss + output.codebook_loss
            total_commitment_loss = total_commitment_loss + output.commitment_loss
            residual = residual - output.quantized.detach()

        return ResidualQuantizerOutput(
            quantized=quantized,
            code_ids=torch.stack(code_ids_by_layer, dim=1),
            loss=total_loss,
            codebook_loss=total_codebook_loss,
            commitment_loss=total_commitment_loss,
        )

    def _validate_latents(self, latents: torch.Tensor) -> None:
        if latents.ndim != 2:
            raise ValueError("latents must be a 2D tensor.")
        if latents.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Expected latent dimension {self.embedding_dim}, got {latents.shape[1]}."
            )


class RqvaeDecoder(nn.Module):
    """Decode quantized latent vectors back to item embedding space."""

    def __init__(
        self,
        latent_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive.")
        if output_dim <= 0:
            raise ValueError("output_dim must be positive.")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive when provided.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")

        self.latent_dim = latent_dim
        self.output_dim = output_dim

        if hidden_dim is None:
            self.network = nn.Linear(latent_dim, output_dim)
        else:
            self.network = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward(self, quantized_latents: torch.Tensor) -> torch.Tensor:
        if quantized_latents.ndim != 2:
            raise ValueError("quantized_latents must be a 2D tensor.")
        if quantized_latents.shape[1] != self.latent_dim:
            raise ValueError(
                f"Expected latent dimension {self.latent_dim}, "
                f"got {quantized_latents.shape[1]}."
            )

        return self.network(quantized_latents)


class RqvaeModel(nn.Module):
    """Minimal RQ-VAE tokenizer model for item semantic id learning."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        num_quantizer_layers: int,
        codebook_size: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        commitment_weight: float = 0.25,
        reconstruction_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight cannot be negative.")

        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_quantizer_layers = num_quantizer_layers
        self.codebook_size = codebook_size
        self.reconstruction_weight = reconstruction_weight

        self.encoder = RqvaeEncoder(
            input_dim=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.quantizer = ResidualQuantizer(
            num_layers=num_quantizer_layers,
            codebook_size=codebook_size,
            embedding_dim=latent_dim,
            commitment_weight=commitment_weight,
        )
        self.decoder = RqvaeDecoder(
            latent_dim=latent_dim,
            output_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

    @torch.no_grad()
    def initialize_codebooks_with_kmeans(
        self,
        item_embeddings: torch.Tensor,
        seed: int = 42,
        max_iter: int = 100,
        n_init: int = 10,
    ) -> torch.LongTensor:
        was_training = self.training
        self.eval()

        latents = self.encoder(item_embeddings)
        code_ids = self.quantizer.initialize_codebooks_with_kmeans(
            latents=latents,
            seed=seed,
            max_iter=max_iter,
            n_init=n_init,
        )

        if was_training:
            self.train()
        return code_ids

    def forward(self, item_embeddings: torch.Tensor) -> RqvaeOutput:
        latents = self.encoder(item_embeddings)
        quantizer_output = self.quantizer(latents)
        reconstructed = self.decoder(quantizer_output.quantized)

        reconstruction_loss = self.reconstruction_weight * F.mse_loss(
            reconstructed,
            item_embeddings,
        )
        loss = reconstruction_loss + quantizer_output.loss

        return RqvaeOutput(
            reconstructed=reconstructed,
            latents=latents,
            quantized=quantizer_output.quantized,
            code_ids=quantizer_output.code_ids,
            loss=loss,
            reconstruction_loss=reconstruction_loss,
            quantizer_loss=quantizer_output.loss,
            codebook_loss=quantizer_output.codebook_loss,
            commitment_loss=quantizer_output.commitment_loss,
        )
