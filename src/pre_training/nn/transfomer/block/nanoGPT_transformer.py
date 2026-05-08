import torch.nn as nn
from pre_training.nn.attention.factory import build_attention
from pre_training.nn.ffn import nanoGPTFeedForward
from pre_training.nn.norms import LayerNorm
from pre_training.nn.utils import autocast_precision


class nanoGPTTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = build_attention(cfg)
        self.ff = nanoGPTFeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"]))
        self.norm2 = LayerNorm(cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"]))
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x
