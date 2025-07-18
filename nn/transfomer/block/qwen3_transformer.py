import torch.nn as nn

from nn.attention.groupquery_attention import GroupedQueryAttention
from nn.attention.multihead_latent_attention import MultiHeadLatentAttention
from nn.attention.native_sparse_attention import NativeSparseAttention
from nn.ffn import Qwen3FeedForward
from nn.norms import Qwen3RMSNorm
from nn.utils import autocast_precision

from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Shard
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module

from nn.distributed.utils import get_tp_wrappers
from nn.distributed.parallel.tensor_parallel import SequenceParallel


class Qwen3TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        n_heads = cfg["n_heads"]
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
                original_seq_len=cfg.get("original_seq_len", cfg["max_seq_length"]),
                num_heads=cfg["n_heads"],
                dtype=autocast_precision(cfg["dtype"]),
                n_kv_heads=n_kv_heads,
                qkv_bias=cfg["qkv_bias"],
                use_rope=cfg["rope"],
                q_lora_rank=cfg.get("q_lora_rank", None),
                kv_lora_rank=cfg.get("kv_lora_rank", cfg["emb_dim"] // 2),
                qk_rope_head_dim=cfg.get("qk_rope_head_dim", 64),
                qk_nope_head_dim=cfg.get("qk_nope_head_dim", 128),
                v_head_dim=cfg.get("v_head_dim", cfg["emb_dim"] // cfg["n_heads"]),
                rope_theta=cfg.get("rope_theta", 10000.0),
                softcap=cfg.get("softcap", None),
                attn_impl=cfg.get("attn_impl", "absorb"),
                mscale=cfg.get("mscale", 1.0),
                batch_size=cfg.get("batch_size", 1),
            )
        else:
            self.att = GroupedQueryAttention(
                d_in=cfg["emb_dim"],
                d_out=cfg["emb_dim"],
                num_heads=cfg["n_heads"],
                num_kv_groups=cfg["n_kv_heads"],
                dtype=autocast_precision(cfg["dtype"]),
                max_seq_len=cfg["max_seq_length"],
                window_size=cfg.get("window_size", cfg["max_seq_length"]),
                use_rope=cfg["rope"],
                qk_norm=cfg["qk_norm"],
                use_flash_attn=cfg.get("use_flash_attn", True),
            )
        self.ff = Qwen3FeedForward(cfg)
        self.norm1 = Qwen3RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = Qwen3RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, cos, sin, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x, cos, sin, use_cache)  # Shape [batch_size, num_tokens, emb_size]
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + shortcut

        return x

    def apply_tp(self, tp_mesh: DeviceMesh, *, input_layout: Placement, float8_enabled: bool = False):
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=(input_layout,),
                desired_input_layouts=(Shard(1),),
            ),
        )

        # parallelize_module(
        #     self.attention_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        # )

        self.att.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=False,
        )

        # parallelize_module(
        #     self.feed_forward_norm, device_mesh=tp_mesh, parallelize_plan=SequenceParallel()
        # )

        self.ff.apply_tp(
            tp_mesh,
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=False,
        )

        # parallelize_module(self.dropout, device_mesh=tp_mesh, parallelize_plan=SequenceParallel())

    def apply_cp(self, cp_mesh: DeviceMesh):
        self.att.apply_cp(cp_mesh)