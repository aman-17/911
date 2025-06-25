import torch.nn as nn

from nn.norms import Qwen3RMSNorm
from nn.rope import RotaryPositionalEmbeddings
from nn.transfomer.block.qwen3_transformer import Qwen3TransformerBlock
from nn.utils import autocast_precision


class Qwen3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.trf_blocks = nn.ModuleList(
            [Qwen3TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = Qwen3RMSNorm(cfg["emb_dim"])
        self.max_seq_len = cfg["max_seq_length"]
        self.head_dim = cfg["emb_dim"] // cfg["n_heads"]
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.dtype = autocast_precision(cfg["dtype"])
        self.rope = RotaryPositionalEmbeddings(
            dim=self.head_dim, max_seq_len=self.max_seq_len
        )
        cos, sin = self.rope.compute_rope_params(
            dim=self.head_dim,
            max_seq_len=self.max_seq_len,
            base=cfg["rope_theta"],
            dtype=self.dtype,
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def forward(self, in_idx, use_cache=False):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        for block in self.trf_blocks:
            x = block(x, self.cos, self.sin, use_cache)
        x = self.final_norm(x)
        logits = self.out_head(x.to(self.dtype))
        return logits

    def reset_kv_cache(self):
        for blk in self.trf_blocks:
            blk.att.reset_cache()
        self.ptr_current_pos = 0
