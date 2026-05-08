import torch.nn as nn
from pre_training.nn.attention.factory import build_attention
from pre_training.nn.attention.groupquery_attention import GroupedQueryAttention
from pre_training.nn.attention.minmax_attention import MinMaxAttention
from pre_training.nn.ffn import Qwen3FeedForward
from pre_training.nn.norms import Qwen3RMSNorm
from pre_training.nn.utils import autocast_precision
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Shard
from torch.distributed.tensor.parallel import PrepareModuleInput, parallelize_module


class Qwen3TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = build_attention(cfg)
        self.ff = Qwen3FeedForward(cfg)
        self.norm1 = Qwen3RMSNorm(cfg["emb_dim"], eps=1e-6)
        self.norm2 = Qwen3RMSNorm(cfg["emb_dim"], eps=1e-6)

    def forward(self, x, cos, sin, use_cache=False):
        shortcut = x
        x = self.norm1(x)
        if isinstance(self.att, MinMaxAttention):
            x = self.att(x, use_cache=use_cache)
        elif isinstance(self.att, GroupedQueryAttention):
            x = self.att(x, cos, sin, use_cache)
        else:
            x = self.att(x)
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
        self.att.apply_tp(
            tp_mesh,
            input_layout=Shard(1),
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=False,
        )
        self.ff.apply_tp(
            tp_mesh,
            output_layout=Shard(1),
            use_local_output=False,
            float8_enabled=False,
        )

    def apply_cp(self, cp_mesh: DeviceMesh):
        self.att.apply_cp(cp_mesh)
