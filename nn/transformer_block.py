import torch.nn as nn

from nn.attention.multihead_attention import MultiHeadAttention
from nn.attention.nsa import NativeSparseAttention
from nn.attention.multihead_latent_attention import MultiHeadLatentAttention
from nn.ffn import FeedForward
from nn.norms import LayerNorm


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        if cfg.get("attention", "mha") == "nsa":
            self.att = NativeSparseAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                max_seq_len=cfg["max_seq_length"],
                num_heads=cfg["n_heads"],
                dropout=cfg["drop_rate"],
                n_kv_heads=cfg.get("n_kv_heads", cfg["n_heads"]),
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
                n_kv_heads=cfg.get("n_kv_heads", cfg["n_heads"]),
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
                qkv_bias=cfg["qkv_bias"],
                use_rope=cfg["rope"],
            )

        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x
