import torch
import torch.nn as nn
from typing import Optional, Union

from nn.attention.multihead_attention import MultiHeadAttention
from nn.attention.native_sparse_attention import NativeSparseAttention
from nn.attention.multihead_latent_attention import MultiHeadLatentAttention
from nn.ffn import FeedForward
from nn.norms import RMSNorm
from nn.utils import autocast_precision


class LlamaTransformerBlock(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        n_heads = cfg["n_heads"]
        self.block_idx = cfg["block_idx"]
        n_kv_heads = cfg.get("n_kv_heads", n_heads)
        if n_heads % n_kv_heads != 0:
            raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")
        if cfg.get("attention", "mha") == "nsa":
            self.att = NativeSparseAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                max_seq_len=cfg["max_seq_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                dtype=autocast_precision(cfg["dtype"]),
                n_kv_heads=n_kv_heads,
                qkv_bias=cfg["qkv_bias"],
                use_rope=cfg["rope"],
                compression_block_size=cfg.get("compression_block_size", 16),
                compression_stride=cfg.get("compression_stride", 16),
                selection_block_size=cfg.get("selection_block_size", 8),
                selection_top_k=cfg.get("selection_top_k", 2),
                window_size=cfg.get("window_size", 256),
            )
        elif cfg.get("attention", "mha") == "mla":
            self.att = MultiHeadLatentAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                max_seq_len=cfg["max_seq_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                dtype=autocast_precision(cfg["dtype"]),
                n_kv_heads=n_kv_heads,
                qkv_bias=cfg["qkv_bias"],
                use_rope=cfg["rope"],
                q_lora_rank=cfg.get("q_lora_rank", None),
                kv_lora_rank=cfg.get("kv_lora_rank", cfg["emb_dim"] // 2),
                qk_rope_head_dim=cfg.get("qk_rope_head_dim", 64),
                qk_nope_head_dim=cfg.get("qk_nope_head_dim", None),
                v_head_dim=cfg.get("v_head_dim", cfg["emb_dim"] // cfg["n_heads"]),
                rope_theta=cfg.get("rope_theta", 10000.0),
                softcap=cfg.get("softcap", None),
            )
        else:
            self.att = MultiHeadAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                max_seq_len=cfg["max_seq_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                dtype=autocast_precision(cfg["dtype"]),
                qkv_bias=cfg["qkv_bias"],
                use_rope=cfg["rope"],
                use_flash_attn=cfg.get("use_flash_attn", True),
            )
        self.dropout = nn.Dropout(cfg["drop_rate"]) if cfg["drop_rate"] > 0.0 else nn.Identity()
        self.ff = FeedForward(cfg)
        self.norm1 = RMSNorm(cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"]))
        self.norm2 = RMSNorm(cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"]))
        
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(
        self,
        x,
    ) -> torch.Tensor:
        h = x + self.dropout(self.att(self.norm1(x)))
        return h + self.dropout(self.ff(self.norm2(h)))