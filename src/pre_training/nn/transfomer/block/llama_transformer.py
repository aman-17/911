from typing import Optional

import torch.nn as nn

from pre_training.nn.attention.factory import build_attention
from pre_training.nn.attention.minmax_attention import MinMaxAttention
from pre_training.nn.attention.multihead_latent_attention import (
    MultiHeadLatentAttention,
)
from pre_training.nn.ffn import FeedForward
from pre_training.nn.norms import RMSNorm
from pre_training.nn.utils import autocast_precision


class LlamaTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.block_idx = cfg["block_idx"]
        self.att = build_attention(cfg)
        self.dropout = nn.Dropout(cfg["drop_rate"]) if cfg["drop_rate"] > 0.0 else nn.Identity()
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"]))
        self.norm2 = RMSNorm(cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"]))
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(
        self,
        x,
        start_pos: int = 0,
        freqs_cis: Optional[object] = None,
        mask: Optional[object] = None,
    ):
        if isinstance(self.att, MultiHeadLatentAttention):
            h = x + self.dropout(self.att(self.norm1(x), start_pos, freqs_cis, mask))
        elif isinstance(self.att, MinMaxAttention):
            h = x + self.dropout(self.att(self.norm1(x), attn_mask=mask))
        else:
            h = x + self.dropout(self.att(self.norm1(x)))
        return h + self.dropout(self.ff(self.norm2(h)))
