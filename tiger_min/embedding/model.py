import torch
from torch import nn
from torch.nn import functional as F


class SkipGramNegSampling(nn.Module):
    """Minimal item2vec skip-gram model with negative sampling."""

    def __init__(self, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        if num_items <= 0:
            raise ValueError("num_items must be positive.")
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")

        self.center_embeddings = nn.Embedding(num_items, embedding_dim)
        self.context_embeddings = nn.Embedding(num_items, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.center_embeddings.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.context_embeddings.weight, mean=0.0, std=0.01)

    def forward(
        self,
        center_items: torch.LongTensor,
        positive_contexts: torch.LongTensor,
        negative_items: torch.LongTensor,
    ) -> torch.Tensor:
        center_vec = self.center_embeddings(center_items)
        positive_vec = self.context_embeddings(positive_contexts)
        negative_vecs = self.context_embeddings(negative_items)

        positive_scores = torch.sum(center_vec * positive_vec, dim=-1)
        positive_loss = -F.logsigmoid(positive_scores)

        negative_scores = torch.bmm(negative_vecs, center_vec.unsqueeze(-1)).squeeze(-1)
        negative_loss = -F.logsigmoid(-negative_scores).sum(dim=-1)

        return (positive_loss + negative_loss).mean()

    def item_embeddings(self) -> torch.Tensor:
        return self.center_embeddings.weight.detach().clone()

