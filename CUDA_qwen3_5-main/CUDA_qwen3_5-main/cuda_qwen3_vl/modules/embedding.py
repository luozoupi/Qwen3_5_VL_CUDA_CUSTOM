from __future__ import annotations

import torch
from torch import nn

from cuda_qwen3_vl.kernels import embedding


class CudaEmbedding(nn.Module):
    """Drop-in replacement for nn.Embedding using the custom CUDA embedding kernel."""

    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int | None = None) -> None:
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].zero_()

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return embedding(ids, self.weight, self.padding_idx)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}, padding_idx={self.padding_idx}"
