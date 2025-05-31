# from: https://github.com/dhcode-cpp/NSA-pytorch/blob/main/native-sparse-attention-pytorch.ipynb

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.rope import RotaryPositionalEmbeddings


class NativeSparseAttention(nn.Module):

    def __init__(
        self,
        d_in: int,
        d_out: int,
        max_seq_len: int,
        num_heads: int,
        dropout: float,
        n_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        use_rope: bool = True,
        compression_block_size: int = 16,
        compression_stride: int = 16,
        selection_block_size: int = 8,
        selection_top_k: int = 2,
        window_size: int = 256,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.n_kv_heads
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, self.n_kv_heads * self.head_dim, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.compression_block_size = compression_block_size
        self.compression_stride = compression_stride
        self.selection_block_size = selection_block_size
        self.selection_top_k = selection_top_k
        self.window_size = window_size

        self.w_k_compress = nn.Parameter(torch.randn(compression_block_size, 1))
        self.w_v_compress = nn.Parameter(torch.randn(compression_block_size, 1))
        self.w_pe_compress = nn.Parameter(
            torch.randn(compression_block_size, self.n_kv_heads * self.head_dim)
        )

        # Gating weights for combining different attention outputs
        self.w_gate = nn.Linear(d_in, 3)  # 3 gates= compress,select,window
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.head_dim, max_seq_len=self.max_seq_len
            )
            pass

        self.register_buffer(
            "causal_mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        )

    def _create_sliding_window_mask(self, seq_len: int) -> torch.Tensor:
        mask = torch.ones(seq_len, seq_len, device=self.causal_mask.device)
        mask = torch.tril(mask)
        for i in range(seq_len):
            if i > self.window_size:
                mask[i, : i - self.window_size] = 0

        return mask

    def _compress_tokens(
        self, keys: torch.Tensor, values: torch.Tensor, batch_size: int, seq_len: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= self.compression_block_size:
            return (
                keys,
                values,
            )  # if sequence is too short, just return the original keys/values

        num_blocks = max(
            1, (seq_len - self.compression_block_size) // self.compression_stride + 1
        )

        compressed_keys = []
        compressed_values = []

        for i in range(num_blocks):
            start_idx = i * self.compression_stride
            end_idx = start_idx + self.compression_block_size

            if end_idx > seq_len:
                end_idx = seq_len

            if start_idx >= seq_len:
                break

            block_k = keys[:, start_idx:end_idx, :]
            block_v = values[:, start_idx:end_idx, :]
            actual_size = end_idx - start_idx
            if actual_size < self.compression_block_size:
                padding = self.compression_block_size - actual_size
                block_k = F.pad(block_k, (0, 0, 0, padding))
                block_v = F.pad(block_v, (0, 0, 0, padding))

            pe = (
                self.w_pe_compress[:actual_size].unsqueeze(0).to(keys.device)
            )  # pe to actual tokens
            if actual_size > 0:
                block_k[:, :actual_size, :] = block_k[:, :actual_size, :] + pe
                block_v[:, :actual_size, :] = block_v[:, :actual_size, :] + pe

            # [batch, block_size, dim] -> [batch, 1, dim]
            w_k_compress = self.w_k_compress.to(keys.device)
            w_v_compress = self.w_v_compress.to(keys.device)
            compressed_k = (block_k.transpose(1, 2) @ w_k_compress).transpose(1, 2)
            compressed_v = (block_v.transpose(1, 2) @ w_v_compress).transpose(1, 2)

            compressed_keys.append(compressed_k)
            compressed_values.append(compressed_v)

        if not compressed_keys:
            compressed_keys = [keys[:, :1, :]]
            compressed_values = [values[:, :1, :]]

        compressed_keys = torch.cat(compressed_keys, dim=1)
        compressed_values = torch.cat(compressed_values, dim=1)

        return compressed_keys, compressed_values

    def _select_tokens(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        compressed_keys: torch.Tensor,
        compressed_values: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        num_compressed = compressed_keys.shape[1]

        q_compress = queries.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k_compress = compressed_keys.view(
            batch_size, num_compressed, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        # KV heads for GQA
        if self.n_kv_heads < self.num_heads:
            k_compress = k_compress.repeat_interleave(self.num_kv_groups, dim=1)

        scores = q_compress @ k_compress.transpose(2, 3) / math.sqrt(self.head_dim)
        token_importance = scores.sum(dim=1)  # [batch, seq_len, num_compressed]
        _, top_indices = torch.topk(
            token_importance, k=min(self.selection_top_k, num_compressed), dim=-1
        )
        selected_keys = []
        selected_values = []

        for i in range(seq_len):
            token_keys = []
            token_values = []

            for j in range(self.selection_top_k):
                if j < top_indices.shape[-1]:
                    block_idx = top_indices[:, i, j]
                    start_idx = block_idx * self.compression_stride
                    end_idx = torch.min(
                        start_idx + self.selection_block_size, torch.tensor(seq_len)
                    )
                    for b in range(batch_size):
                        s, e = start_idx[b].item(), end_idx[b].item()
                        token_keys.append(keys[b : b + 1, s:e, :])
                        token_values.append(values[b : b + 1, s:e, :])

            if token_keys:
                selected_keys.append(torch.cat(token_keys, dim=1))
                selected_values.append(torch.cat(token_values, dim=1))

        return selected_keys, selected_values

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_in = x.shape
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        if self.use_rope:
            queries = self.rope(queries)
            keys = self.rope(keys)
            pass

        compressed_keys, compressed_values = self._compress_tokens(
            keys, values, batch_size, seq_len
        )
        q_comp = queries.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k_comp = compressed_keys.view(
            batch_size, -1, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        v_comp = compressed_values.view(
            batch_size, -1, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        if self.n_kv_heads < self.num_heads:
            k_comp = k_comp.repeat_interleave(self.num_kv_groups, dim=1)
            v_comp = v_comp.repeat_interleave(self.num_kv_groups, dim=1)

        scores_comp = q_comp @ k_comp.transpose(2, 3) / math.sqrt(self.head_dim)
        attn_comp = F.softmax(scores_comp, dim=-1)
        attn_comp = self.dropout(attn_comp)
        out_comp = (
            (attn_comp @ v_comp)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_out)
        )
        selected_keys, selected_values = self._select_tokens(
            queries,
            keys,
            values,
            compressed_keys,
            compressed_values,
            batch_size,
            seq_len,
        )
        out_select = torch.zeros(batch_size, seq_len, self.d_out, device=x.device)
        q_sel = queries.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        for i in range(seq_len):
            if i < len(selected_keys):
                k_sel = (
                    selected_keys[i]
                    .view(batch_size, -1, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )
                v_sel = (
                    selected_values[i]
                    .view(batch_size, -1, self.n_kv_heads, self.head_dim)
                    .transpose(1, 2)
                )

                if self.n_kv_heads < self.num_heads:
                    k_sel = k_sel.repeat_interleave(self.num_kv_groups, dim=1)
                    v_sel = v_sel.repeat_interleave(self.num_kv_groups, dim=1)

                scores_sel = (
                    q_sel[:, :, i : i + 1, :]
                    @ k_sel.transpose(2, 3)
                    / math.sqrt(self.head_dim)
                )
                attn_sel = F.softmax(scores_sel, dim=-1)
                attn_sel = self.dropout(attn_sel)

                out_i = (
                    (attn_sel @ v_sel)
                    .squeeze(2)
                    .transpose(1, 2)
                    .contiguous()
                    .view(batch_size, self.d_out)
                )
                out_select[:, i, :] = out_i

        window_mask = self._create_sliding_window_mask(seq_len)
        q_win = queries.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        k_win = keys.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)
        v_win = values.view(
            batch_size, seq_len, self.n_kv_heads, self.head_dim
        ).transpose(1, 2)

        if self.n_kv_heads < self.num_heads:
            k_win = k_win.repeat_interleave(self.num_kv_groups, dim=1)
            v_win = v_win.repeat_interleave(self.num_kv_groups, dim=1)

        scores_win = q_win @ k_win.transpose(2, 3) / math.sqrt(self.head_dim)
        scores_win = scores_win.masked_fill(
            window_mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf")
        )
        attn_win = F.softmax(scores_win, dim=-1)
        attn_win = self.dropout(attn_win)
        out_win = (
            (attn_win @ v_win)
            .transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.d_out)
        )
        gates = torch.sigmoid(self.w_gate(x))  # [batch, seq_len, 3]
        out = (
            gates[:, :, 0:1] * out_comp
            + gates[:, :, 1:2] * out_select
            + gates[:, :, 2:3] * out_win
        )
        out = self.out_proj(out)

        return out
