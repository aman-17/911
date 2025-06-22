import math

import torch
import torch.nn as nn

from nn.norms import LayerNorm
from nn.transfomer.block.gpt_transformer_block import GPTTransformerBlock
from nn.transfomer.block.nanoGPT_transformer_block import nanoGPTTransformerBlock
from nn.utils import autocast_precision


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["max_seq_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[GPTTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(
            cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"])
        )
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], dtype=autocast_precision(cfg["dtype"])
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.cfg["n_layers"]) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class nGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["max_seq_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[GPTTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(
            cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"])
        )
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.sz_init_value = 1.0
        self.sz_init_scaling = 1.0 / math.sqrt(cfg["emb_dim"])
        self.sz = nn.Parameter(
            torch.empty(cfg["vocab_size"], dtype=autocast_precision(cfg["dtype"]))
        )
        self.apply(self._init_weights)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sz)
        with torch.no_grad():
            self.sz.mul_(self.sz_init_scaling)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.cfg["n_layers"]) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        tok_embeds = self.l2_normalize(tok_embeds, dim=-1)
        pos_embeds = self.l2_normalize(pos_embeds, dim=-1)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
        logits = sz * self.out_head(x)
        return logits

    @torch.no_grad()
    def normalize_matrices(self):
        self._normalize_matrix(self.out_head.weight)
        self._normalize_matrix(self.tok_emb.weight, dim=-1)
        self._normalize_matrix(self.pos_emb.weight, dim=-1)

    @staticmethod
    def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).type_as(x)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(self.l2_normalize(w, dim=dim))


class nanoGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["max_seq_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[nanoGPTTransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(
            cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"])
        )
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        assert (
            seq_len <= self.cfg["emb_dim"]
        ), f"Cannot forward sequence of length {seq_len}, block size is only {self.cfg['emb_dim']}"
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(0, seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        logits = self.out_head(x[:, [-1], :])
        return logits
