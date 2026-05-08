import torch.nn as nn
from pre_training.nn.attention.groupquery_attention import GroupedQueryAttention
from pre_training.nn.attention.minmax_attention import MinMaxAttention
from pre_training.nn.attention.multihead_attention import MultiHeadAttention
from pre_training.nn.attention.multihead_latent_attention import MultiHeadLatentAttention
from pre_training.nn.attention.native_sparse_attention import NativeSparseAttention
from pre_training.nn.utils import autocast_precision


def build_attention(cfg) -> nn.Module:
    """Return the attention module specified by cfg["attention"]."""
    n_heads = cfg["n_heads"]
    n_kv_heads = cfg.get("n_kv_heads", n_heads)
    if n_heads % n_kv_heads != 0:
        raise ValueError(f"n_heads ({n_heads}) must be divisible by n_kv_heads ({n_kv_heads})")

    kind = cfg.get("attention", "mha")
    dtype = autocast_precision(cfg["dtype"])

    if kind == "minmax":
        return MinMaxAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=n_heads,
            max_seq_len=cfg["max_seq_length"],
            dropout=cfg["drop_rate"],
            dtype=dtype,
            qkv_bias=cfg["qkv_bias"],
            activation=cfg.get("minmax_activation", "silu"),
            block_size=cfg.get("minmax_block_size", 256),
        )

    if kind == "nsa":
        return NativeSparseAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            max_seq_len=cfg["max_seq_length"],
            num_heads=n_heads,
            dropout=cfg["drop_rate"],
            dtype=dtype,
            n_kv_heads=n_kv_heads,
            qkv_bias=cfg["qkv_bias"],
            use_rope=cfg["rope"],
            compression_block_size=cfg.get("compression_block_size", 16),
            compression_stride=cfg.get("compression_stride", 16),
            selection_block_size=cfg.get("selection_block_size", 8),
            selection_top_k=cfg.get("selection_top_k", 2),
            window_size=cfg.get("window_size", 256),
        )

    if kind == "mla":
        return MultiHeadLatentAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            max_seq_len=cfg["max_seq_length"],
            original_seq_len=cfg.get("original_seq_len", cfg["max_seq_length"]),
            num_heads=n_heads,
            dtype=dtype,
            n_kv_heads=n_kv_heads,
            qkv_bias=cfg["qkv_bias"],
            use_rope=cfg["rope"],
            q_lora_rank=cfg.get("q_lora_rank", None),
            kv_lora_rank=cfg.get("kv_lora_rank", cfg["emb_dim"] // 2),
            qk_rope_head_dim=cfg.get("qk_rope_head_dim", 64),
            qk_nope_head_dim=cfg.get("qk_nope_head_dim", 128),
            v_head_dim=cfg.get("v_head_dim", cfg["emb_dim"] // n_heads),
            rope_theta=cfg.get("rope_theta", 10000.0),
            softcap=cfg.get("softcap", None),
            attn_impl=cfg.get("attn_impl", "absorb"),
            mscale=cfg.get("mscale", 1.0),
            rope_factor=cfg.get("rope_factor", 1.0),
        )

    if kind == "gqa":
        return GroupedQueryAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            num_heads=n_heads,
            num_kv_groups=n_kv_heads,
            dtype=dtype,
            max_seq_len=cfg["max_seq_length"],
            window_size=cfg.get("window_size", cfg["max_seq_length"]),
            use_rope=cfg["rope"],
            qk_norm=cfg.get("qk_norm", False),
            use_flash_attn=cfg.get("use_flash_attn", True),
        )

    return MultiHeadAttention(
        d_in=cfg["emb_dim"],
        d_out=cfg["emb_dim"],
        max_seq_len=cfg["max_seq_length"],
        num_heads=n_heads,
        dropout=cfg["drop_rate"],
        dtype=dtype,
        qkv_bias=cfg["qkv_bias"],
        use_rope=cfg["rope"],
        use_flash_attn=cfg.get("use_flash_attn", True),
        use_cache=cfg.get("use_cache", False),
    )
