from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module

from nn.distributed.utils import get_tp_wrappers
from nn.distributed.parallel.tensor_parallel import SequenceParallel
from nn.norms import Qwen3RMSNorm
from nn.rope import RotaryPositionalEmbeddings


class GroupedQueryAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        num_kv_groups: int,
        dtype: torch.dtype,
        max_seq_len: int,
        window_size: Optional[int] = None,
        use_rope: bool = True,
        qk_norm: bool = False,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        assert num_heads % num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.num_heads = num_heads
        self.num_kv_groups = num_kv_groups
        self.group_size = num_heads // num_kv_groups
        self.head_dim = d_in // num_heads
        self.d_out = self.num_heads * self.head_dim
        self.max_seq_len = max_seq_len
        self.w_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.w_key = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.w_value = nn.Linear(d_in, num_kv_groups * self.head_dim, bias=False, dtype=dtype)
        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)
        if qk_norm:
            self.q_norm = Qwen3RMSNorm(self.head_dim, eps=1e-6)
            self.k_norm = Qwen3RMSNorm(self.head_dim, eps=1e-6)
        else:
            self.q_norm = self.k_norm = None
        self.use_rope = use_rope
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=self.max_seq_len)
        self.window_size = window_size or self.max_seq_len
        self.use_flash_attn = use_flash_attn
        self.register_buffer("cache_k", None, persistent=False)
        self.register_buffer("cache_v", None, persistent=False)
        self.cache_initialized = False
        self.ptr = 0
        self._cp_pg: Optional[dist.ProcessGroup] = None
        self._cp_enabled = False

    @property
    def cp_enabled(self) -> bool:
        return self._cp_enabled

    def forward(self, x, cos, sin, use_cache=False):
        b, num_tokens, _ = x.shape
        queries = self.w_query(x)  # (b, num_tokens, num_heads * head_dim)
        keys_new = self.w_key(x)  # (b, num_tokens, num_kv_groups * head_dim)
        values_new = self.w_value(x)  # (b, num_tokens, num_kv_groups * head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys_new = keys_new.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values_new = values_new.view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        if self.q_norm:
            queries = self.q_norm(queries)
        if self.k_norm:
            keys_new = self.k_norm(keys_new)
        pos_start = self.ptr
        pos_end = pos_start + num_tokens
        cos_slice = cos[pos_start:pos_end]
        sin_slice = sin[pos_start:pos_end]

        if self.use_rope:
            keys_new = self.rope.qwen3_rope(keys_new, cos_slice, sin_slice)
            queries = self.rope.qwen3_rope(queries, cos_slice, sin_slice)

        keys_new = keys_new.repeat_interleave(self.group_size, dim=1)
        values_new = values_new.repeat_interleave(self.group_size, dim=1)

        if use_cache:
            if not self.cache_initialized:
                self.cache_k = torch.zeros(
                    b,
                    self.num_heads,
                    self.max_seq_len,
                    self.head_dim,
                    device=x.device,
                    dtype=keys_new.dtype,
                )
                self.cache_v = torch.zeros(
                    b,
                    self.num_heads,
                    self.max_seq_len,
                    self.head_dim,
                    device=x.device,
                    dtype=values_new.dtype,
                )
                self.ptr = 0
                self.cache_initialized = True

            end = self.ptr + num_tokens
            self.cache_k[:, :, self.ptr : end].copy_(keys_new)
            self.cache_v[:, :, self.ptr : end].copy_(values_new)

            keys = self.cache_k[:, :, max(0, end - self.window_size) : end]
            values = self.cache_v[:, :, max(0, end - self.window_size) : end]
            self.ptr = end
        else:
            keys, values = keys_new, values_new

        if self.use_flash_attn:
            context_vec = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=True,
            )
            context_vec = context_vec.transpose(1, 2).reshape(b, num_tokens, self.d_out)
        else:
            attn_scores = queries @ keys.transpose(2, 3)
            T_q = queries.shape[-2]
            T_k = keys.shape[-2]
            if not use_cache or T_q > 1:
                causal_mask = torch.triu(
                    torch.ones((T_q, T_k), device=x.device, dtype=torch.bool),
                    diagonal=1,
                )
                attn_scores = attn_scores.masked_fill(causal_mask, -torch.inf)
            attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
            context_vec = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(float8_enabled=float8_enabled)

        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        plan = {
            "w_q": colwise_parallel(
                output_layouts=None if self.q_norm is None else Shard(1),
                use_local_output=self.q_norm is None,
            ),
            "w_k": colwise_parallel(
                output_layouts=None if self.k_norm is None else Shard(1),
                use_local_output=self.k_norm is None,
            ),
            "w_v": colwise_parallel(),
            "w_out": rowwise_parallel(output_layouts=output_layout, use_local_output=use_local_output),
        }
        if self.q_norm is not None:
            plan["q_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(-1))
        if self.k_norm is not None:
            plan["k_norm"] = SequenceParallel(use_local_output=True, output_layouts=Shard(-1))

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=plan,
        )

    def apply_cp(self, cp_mesh: DeviceMesh):
        """
        Prepare the module for context-parallelism (ring attention).

        .. important::
            This requires flash-attn and ring-flash-attn (``use_flash=True``).

        :param cp_mesh: The context parallel device sub-mesh.
        :param load_balancer: The load balancer type.
        """
        self._cp_pg = cp_mesh.get_group()
        self._cp_enabled = True

    def reset_cache(self):
        if self.cache_k is not None:
            self.cache_k.zero_()
            self.cache_v.zero_()
        self.ptr = 0
