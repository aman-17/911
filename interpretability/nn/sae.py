from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SAEConfig:
    d_model: int = 2048
    dict_size: int = 32768
    k: int = 32


@dataclass
class SAEOutput:
    recon: torch.Tensor
    features: torch.Tensor
    pre_acts: torch.Tensor


class SparseAutoencoder(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg

        self.encoder = nn.Linear(cfg.d_model, cfg.dict_size, bias=True)
        self.decoder = nn.Linear(cfg.dict_size, cfg.d_model, bias=True)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)
        with torch.no_grad():
            self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_acts = self.encoder(x)
        topk_vals, topk_idx = torch.topk(pre_acts, k=self.cfg.k, dim=-1)
        features = torch.zeros_like(pre_acts).scatter_(-1, topk_idx, topk_vals.clamp(min=0))
        return features, pre_acts

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> SAEOutput:
        orig_shape = x.shape
        x_flat = x.view(-1, self.cfg.d_model)

        features, pre_acts = self.encode(x_flat)
        recon = self.decode(features)

        return SAEOutput(
            recon=recon.view(*orig_shape),
            features=features.view(*orig_shape[:-1], self.cfg.dict_size),
            pre_acts=pre_acts.view(*orig_shape[:-1], self.cfg.dict_size),
        )

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.data = F.normalize(self.decoder.weight.data, dim=0)

    def loss(self, x: torch.Tensor, recon: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(recon, x)
