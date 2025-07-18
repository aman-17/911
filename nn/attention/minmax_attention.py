import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.activations import SiLU
from nn.norms import T5LayerNorm


class MinMaxAttention(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        num_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        dtype: torch.dtype = torch.float32,
        qkv_bias: bool = False,
        block_size: int = 256,
        **kwargs,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.block_size = block_size

        self.qkv_proj = nn.Linear(d_in, 3 * self.head_dim * num_heads, bias=qkv_bias)
        self.output_gate = nn.Linear(d_in, self.head_dim * num_heads, bias=qkv_bias)
        self.out_proj = nn.Linear(self.head_dim * num_heads, d_out, bias=qkv_bias)

        self.act = SiLU()
        self.norm = T5LayerNorm(self.head_dim * num_heads)

        self.offset = 0
        self._setup_slopes()

    def _setup_slopes(self):
        """Setup slope rates for attention decay"""

        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = 2 ** (-(2 ** -(math.log2(n) - 3)))
                ratio = start
                return [start * ratio**i for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2 ** math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]

        slopes = torch.tensor(get_slopes(self.num_heads), dtype=torch.float32)
        self.register_buffer("slopes", slopes.reshape(self.num_heads, 1, 1))

    def reset_cache(self):
        """Reset cache for new sequence"""
        self.offset = 0

    def forward(
        self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, past_key_value: Optional[Tuple[torch.Tensor]] = None, use_cache: bool = False, **kwargs
    ) -> torch.Tensor:
        """
        Forward pass of MinMax attention

        Args:
            x: Input tensor of shape (batch, seq_len, d_in)
            attn_mask: Optional attention mask
            past_key_value: Optional cached key-value states
            use_cache: Whether to use caching for inference

        Returns:
            Output tensor of shape (batch, seq_len, d_out)
        """
        batch_size, seq_len, _ = x.shape

        # Project to Q, K, V
        qkv = self.act(self.qkv_proj(x))
        new_shape = qkv.size()[:-1] + (self.num_heads, -1)
        qkv = qkv.view(*new_shape)
        q, k, v = torch.split(qkv, [self.head_dim] * 3, dim=3)

        # Reshape for attention computation
        q = q.transpose(1, 2)  # (batch, num_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Compute MinMax attention
        if past_key_value is None:
            self.offset = q.shape[-2]
        else:
            self.offset += 1

        # Get slope rates
        slope_rate = self.slopes.to(q.device)
        ratio = torch.exp(-slope_rate)

        if past_key_value is None:
            # Full sequence computation
            output = self._compute_full_attention(q, k, v, slope_rate, attn_mask)
            kv_cache = None
        else:
            # Incremental computation with cache
            output, kv_cache = self._compute_incremental_attention(q, k, v, ratio, past_key_value)

        # Reshape output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)

        # Apply normalization and gating
        output = self.norm(output)
        output = F.sigmoid(self.output_gate(x)) * output
        output = self.out_proj(output)

        if use_cache:
            return output
        return output

    def _compute_full_attention(self, q, k, v, slope_rate, attn_mask=None):
        """Compute attention for full sequence"""
        b, h, n, d = q.shape
        e = v.shape[-1]

        # Apply mask to values if provided
        if attn_mask is not None:
            mask_expanded = (1 - attn_mask).unsqueeze(1).unsqueeze(-1).to(torch.bool)
            v = v.masked_fill(mask_expanded, 0)

        NUM_BLOCK = (n + self.block_size - 1) // self.block_size

        # Precompute decay patterns
        array = torch.arange(self.block_size, device=q.device) + 1
        q_decay = torch.exp(-slope_rate * array.reshape(-1, 1))
        k_decay = torch.exp(-slope_rate * (self.block_size - array.reshape(-1, 1)))

        index = array[:, None] - array[None, :]
        s_index = (
            slope_rate
            * index[
                None,
                None,
            ]
        )
        s_index = torch.where(index >= 0, -s_index, float("-inf"))
        diag_decay = torch.exp(s_index)

        # Initialize accumulator and output
        kv = torch.zeros(b, h, d, e, dtype=torch.float32, device=q.device)
        output = torch.empty((b, h, n, e), dtype=q.dtype, device=q.device)

        # Process blocks
        for i in range(NUM_BLOCK):
            si = i * self.block_size
            ei = min(si + self.block_size, n)
            m = ei - si

            qi = q[:, :, si:ei].contiguous()
            ki = k[:, :, si:ei].contiguous()
            vi = v[:, :, si:ei].contiguous()

            # Non-diagonal contribution
            qkv_none_diag = torch.matmul(qi * q_decay[:, :m], kv).to(torch.float32)

            # Diagonal contribution
            qk = torch.matmul(qi, ki.transpose(-1, -2)).to(torch.float32) * diag_decay[:, :, :m, :m]
            qkv_diag = torch.matmul(qk, vi.to(torch.float32))

            # Update output
            output[:, :, si:ei] = qkv_none_diag + qkv_diag

            # Update accumulator
            block_decay = torch.exp(-slope_rate * m)
            kv = block_decay * kv + torch.matmul((ki * k_decay[:, -m:]).transpose(-1, -2).to(vi.dtype), vi)

        return output

    def _compute_incremental_attention(self, q, k, v, ratio, past_kv):
        """Compute attention incrementally with cache"""
        kv = past_kv
        output = []

        for i in range(q.shape[-2]):
            kv = ratio * kv + torch.einsum(
                "... n d, ... n e -> ... d e",
                k[:, :, i : i + 1],
                v[:, :, i : i + 1],
            )
            qkv = torch.einsum("... n e, ... e d -> ... n d", q[:, :, i : i + 1], kv.to(q.dtype))
            output.append(qkv)

        output = torch.concat(output, dim=-2)
        return output, kv
