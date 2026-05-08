import torch.nn as nn
from pre_training.nn.attention.factory import build_attention
from pre_training.nn.attention.minmax_attention import MinMaxAttention
from pre_training.nn.ffn import FeedForward
from pre_training.nn.norms import LayerNorm
from pre_training.nn.utils import autocast_precision


class GPTTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = build_attention(cfg)
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"]))
        self.norm2 = LayerNorm(cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"]))
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        if isinstance(self.att, MinMaxAttention):
            x = self.att(x, use_cache=use_cache)
        else:
            x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x
