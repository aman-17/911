import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.activations import GELU
from nn.utils import autocast_precision


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"] if "hidden_dim" in cfg else 4 * self.emb_dim
        self.dtype = autocast_precision(cfg["dtype"])
        self.w1 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)
        self.w2 = nn.Linear(self.hidden_dim, self.emb_dim, dtype=self.dtype)
        self.w3 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)
        # self.layers = nn.Sequential(
        #     nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        #     GELU(),
        #     nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        # )

    def forward(self, x):
        x = x.to(self.dtype)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Qwen3FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"] if "hidden_dim" in cfg else 3 * self.emb_dim
        self.dtype = autocast_precision(cfg["dtype"])
        self.w1 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)
        self.w2 = nn.Linear(self.hidden_dim, self.emb_dim, dtype=self.dtype)
        self.w3 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)

    def forward(self, x):
        x = x.to(self.dtype)
        x_fc1 = self.w1(x)
        x_fc3 = self.w3(x)
        x = F.silu(x_fc1) * x_fc3
        return self.w2(x)


class NormalizedFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"] if "hidden_dim" in cfg else 4 * self.emb_dim
        self.dtype = autocast_precision(cfg["dtype"])
        self.w1 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)
        self.w2 = nn.Linear(self.hidden_dim, self.emb_dim, dtype=self.dtype)
        self.w3 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)
        # self.layers = nn.Sequential(
        #     nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        #     GELU(),
        #     nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        # )
        self.sw_init_value = 1.0
        self.sw_init_scaling = 1.0
        self.sw1 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        self.sw3 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        self.sqrt_d_model = math.sqrt(self.emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sw1)
        nn.init.ones_(self.sw3)
        with torch.no_grad():
            self.sw1.mul_(self.sw_init_scaling)
            self.sw3.mul_(self.sw_init_scaling)

    def forward(self, x):
        x = x.to(self.dtype)
        sw1 = self.sw1 * (
            (self.sw_init_value / self.sw_init_scaling) * self.sqrt_d_model
        )
        sw3 = self.sw3 * (self.sw_init_value / self.sw_init_scaling)
        return self.w2(F.silu(sw1 * self.w1(x)) * (sw3 * self.w3(x)))

    @torch.no_grad()
    def normalize_matrices(self):
        self._normalize_matrix(self.w1.weight)
        self._normalize_matrix(self.w2.weight, dim=0)
        self._normalize_matrix(self.w3.weight)

    @staticmethod
    def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).type_as(x)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(self.l2_normalize(w, dim=dim))


class nanoGPTFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"] if "hidden_dim" in cfg else 4 * self.emb_dim
        self.dtype = autocast_precision(cfg["dtype"])
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], self.hidden_dim),
            GELU(),
            nn.Linear(self.hidden_dim, cfg["emb_dim"]),
        )

    def forward(self, x):
        x = x.to(self.dtype)
        return self.layers(x)
