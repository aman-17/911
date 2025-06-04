import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.rope import RotaryPositionalEmbeddings


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        max_seq_len: int,
        num_heads: int,
        dropout: float,
        dtype: torch.dtype,
        n_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        use_rope: bool = True,
        use_flash_attn: bool = True,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dtype = dtype
        self.w_query = nn.Linear(d_in, d_out, dtype=dtype, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, dtype=dtype, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, dtype=dtype, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        self.use_flash_attn = use_flash_attn
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.head_dim, max_seq_len=self.max_seq_len
            )
        self.register_buffer(
            "mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        )

    def forward(self, x):
        x = x.to(self.dtype)
        batch_size, num_tokens, d_in = x.shape
        keys = self.w_key(x)  # shape (2, 6, 4)
        queries = self.w_query(x)  # shape (2, 6, 4)
        values = self.w_value(x)  # shape (2, 6, 4)

        # shape: (batch_size, seq_len, n_heads, head_dim)
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)

        # This transposes the tensors to bring the num_heads dimension before num_tokens: `SDPA expects this.`
        keys, queries, values = (
            keys.transpose(1, 2),
            queries.transpose(1, 2),
            values.transpose(1, 2),
        )  # New shape: (2, 2, 6, 2)

        if self.use_rope:
            queries = self.rope(queries)
            keys = self.rope(keys)

        if self.use_flash_attn:
            context_vec = F.scaled_dot_product_attention(
                queries, keys, values,
                attn_mask=None,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Scaled dot product between the query and key vectors. keys.transpose(2, 3) changes the shape of keys to (2, 2, 2, 6).
            attn_scores = queries @ keys.transpose(2, 3)  # attn_scores shape: (2, 2, 6, 6).
            # Creates a boolean mask to prevent attending to future tokens
            mask_bool = self.mask.to(torch.bool)[
                :num_tokens, :num_tokens
            ]  # mask_bool shape: (6, 6)

            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)
            # attn_weights: (2, 2, 6, 6), values: (2, 2, 6, 2)
            context_vec = (attn_weights @ values).transpose(
                1, 2
            )  # context_vec: (2, 6, 2, 2)
        context_vec = context_vec.contiguous().view(
            batch_size, num_tokens, self.d_out
        )  # context_vec shape: (2, 6, 4)
        context_vec = self.out_proj(context_vec)

        return context_vec


class NormalizedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        max_seq_len: int,
        num_heads: int,
        dropout: float,
        dtype: torch.dtype,
        n_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.dtype = dtype
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias, dtype=dtype)
        self.out_proj = nn.Linear(d_out, d_out, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.head_dim, max_seq_len=self.max_seq_len
            )
        self.register_buffer(
            "mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        )
        self.sq_init_value = 1.0
        self.sq_init_scaling = 1.0 / math.sqrt(d_in)
        self.sq = nn.Parameter(torch.empty(self.head_dim * self.num_heads, dtype=dtype))
        self.sk_init_value = 1.0
        self.sk_init_scaling = 1.0 / math.sqrt(d_in)
        self.sk = nn.Parameter(torch.empty(self.head_dim * self.n_kv_heads, dtype=dtype))

        self.sqrt_head_dim = math.sqrt(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sq)
        nn.init.ones_(self.sk)
        with torch.no_grad():
            self.sq.mul_(self.sq_init_scaling)
            self.sk.mul_(self.sk_init_scaling)

    def forward(self, x):
        x = x.to(self.dtype)
        batch_size, num_tokens, d_in = x.shape
        keys = self.w_key(x)  # shape (2, 6, 4)
        queries = self.w_query(x)  # shape (2, 6, 4)
        values = self.w_value(x)  # shape (2, 6, 4)

        sq = (self.sq * (self.sq_init_value / self.sq_init_scaling)).view(1, 1, -1)
        queries = sq * queries
        sk = (self.sk * (self.sk_init_value / self.sk_init_scaling)).view(1, 1, -1)
        keys = sk * keys

        # shape: (batch_size, seq_len, n_heads, head_dim)
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)

        # This transposes the tensors to bring the num_heads dimension before num_tokens: `SDPA expects this.`
        keys, queries, values = (
            keys.transpose(1, 2),
            queries.transpose(1, 2),
            values.transpose(1, 2),
        )  # New shape: (2, 2, 6, 2)

        if self.use_rope:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Scaled dot product between the query and key vectors. keys.transpose(2, 3) changes the shape of keys to (2, 2, 2, 6).
        attn_scores = queries @ keys.transpose(2, 3)  # attn_scores shape: (2, 2, 6, 6).
        # Creates a boolean mask to prevent attending to future tokens
        mask_bool = self.mask.to(torch.bool)[
            :num_tokens, :num_tokens
        ]  # mask_bool shape: (6, 6)

        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        # attn_weights: (2, 2, 6, 6), values: (2, 2, 6, 2)
        context_vec = (attn_weights @ values).transpose(
            1, 2
        )  # context_vec: (2, 6, 2, 2)
        context_vec = context_vec.contiguous().view(
            batch_size, num_tokens, self.d_out
        )  # context_vec shape: (2, 6, 4)
        context_vec = self.out_proj(context_vec)

        return context_vec

    @torch.no_grad()
    def normalize_matrices(self):
        self._normalize_matrix(self.w_query.weight)
        self._normalize_matrix(self.w_key.weight)
        self._normalize_matrix(self.w_value.weight)
        self._normalize_matrix(self.out_proj.weight, dim=0)

    @staticmethod
    def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).type_as(x)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(self.l2_normalize(w, dim=dim))
