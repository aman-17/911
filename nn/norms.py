import torch
import torch.nn as nn
from nn.utils import autocast_precision

class LayerNorm(nn.Module):
    def __init__(self, emb_dim, dtype):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim, dtype=autocast_precision(dtype)))
        self.shift = nn.Parameter(torch.zeros(emb_dim, dtype=autocast_precision(dtype)))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class RMSNorm(nn.Module):
    def __init__(self, dim: int, dtype, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=autocast_precision(dtype)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
