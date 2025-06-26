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
        dtype: torch.dtype,
        n_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        use_rope: bool = True,
        use_causal: bool = True,
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
        self.w_query = nn.Linear(d_in, d_out, dtype=dtype, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, self.n_kv_heads * self.head_dim, dtype=dtype, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, self.n_kv_heads * self.head_dim, dtype=dtype, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.compression_block_size = compression_block_size
        self.compression_stride = compression_stride
        self.selection_block_size = selection_block_size
        self.selection_top_k = selection_top_k
        self.window_size = window_size
        self.use_causal = use_causal

        self.w_k_compress = nn.Parameter(torch.randn(compression_block_size, 1, dtype=dtype))
        self.w_v_compress = nn.Parameter(torch.randn(compression_block_size, 1, dtype=dtype))
        self.w_pe_compress = nn.Parameter(torch.randn(compression_block_size, self.n_kv_heads * self.head_dim, dtype=dtype))
        self.w_gate = nn.Linear(d_in, 3, dtype=dtype)  # 3 gates= compress,select,window
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(dim=self.head_dim, max_seq_len=self.max_seq_len)

        self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1))

    def _create_sliding_window_mask(self, seq_len: int) -> torch.Tensor:
        if not hasattr(self, "_cached_window_mask") or self._cached_window_mask.shape[0] < seq_len:
            max_size = max(seq_len, self.max_seq_len)
            device = self.causal_mask.device
            rows = torch.arange(max_size, device=device).unsqueeze(1)
            cols = torch.arange(max_size, device=device).unsqueeze(0)
            mask = (cols <= rows) & (cols > rows - self.window_size)

            self._cached_window_mask = mask

        return self._cached_window_mask[:seq_len, :seq_len]

    def _create_compressed_causal_mask(self, q_len: int, kv_len: int) -> torch.Tensor:
        mask = torch.ones(q_len, kv_len, device=self.causal_mask.device)
        for i in range(q_len):
            for j in range(kv_len):
                block_end = (j + 1) * self.compression_stride
                if block_end > i:
                    mask[i, j] = 0
        return mask

    def _compress_tokens(self, keys: torch.Tensor, values: torch.Tensor, batch_size: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if seq_len <= self.compression_block_size:
            return (
                keys,
                values,
            )

        num_blocks = max(1, (seq_len - self.compression_block_size) // self.compression_stride + 1)

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

            pe = self.w_pe_compress[:actual_size].unsqueeze(0).to(keys.device)  # pe to actual tokens
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_compressed = compressed_keys.shape[1]

        q_compress = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_compress = compressed_keys.view(batch_size, num_compressed, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_kv_heads < self.num_heads:
            k_compress = k_compress.repeat_interleave(self.num_kv_groups, dim=1)

        scores = q_compress @ k_compress.transpose(2, 3) / math.sqrt(self.head_dim)
        token_importance = scores.mean(dim=(1, 2))  # [batch, num_compressed]

        _, top_indices = torch.topk(token_importance, k=min(self.selection_top_k, num_compressed), dim=-1)

        max_selected_tokens = self.selection_top_k * self.selection_block_size
        selected_keys = torch.zeros(batch_size, max_selected_tokens, keys.shape[-1], device=keys.device)
        selected_values = torch.zeros(batch_size, max_selected_tokens, values.shape[-1], device=values.device)
        selected_mask = torch.zeros(batch_size, max_selected_tokens, dtype=torch.bool, device=keys.device)
        selected_positions = torch.zeros(batch_size, max_selected_tokens, dtype=torch.long, device=keys.device)

        # Vectorized gathering
        for b in range(batch_size):
            token_idx = 0
            for k in range(min(self.selection_top_k, top_indices.shape[-1])):
                block_idx = top_indices[b, k].item()
                start_idx = block_idx * self.compression_stride
                end_idx = min(start_idx + self.selection_block_size, seq_len)
                block_len = end_idx - start_idx

                if block_len > 0:
                    selected_keys[b, token_idx : token_idx + block_len] = keys[b, start_idx:end_idx]
                    selected_values[b, token_idx : token_idx + block_len] = values[b, start_idx:end_idx]
                    selected_mask[b, token_idx : token_idx + block_len] = True
                    positions = torch.arange(start_idx, end_idx, device=keys.device)
                    selected_positions[b, token_idx : token_idx + block_len] = positions
                    token_idx += block_len

        return selected_keys, selected_values, selected_mask, selected_positions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_in = x.shape
        queries = self.w_query(x)
        keys = self.w_key(x)
        values = self.w_value(x)

        if self.use_rope:
            queries = self.rope(queries)
            keys = self.rope(keys)

        compressed_keys, compressed_values = self._compress_tokens(keys, values, batch_size, seq_len)
        q_comp = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_comp = compressed_keys.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_comp = compressed_values.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_kv_heads < self.num_heads:
            k_comp = k_comp.repeat_interleave(self.num_kv_groups, dim=1)
            v_comp = v_comp.repeat_interleave(self.num_kv_groups, dim=1)

        scores_comp = q_comp @ k_comp.transpose(2, 3) / math.sqrt(self.head_dim)

        if self.use_causal:
            comp_mask = self._create_compressed_causal_mask(seq_len, compressed_keys.shape[1])
            scores_comp = scores_comp.masked_fill(comp_mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))

        attn_comp = F.softmax(scores_comp, dim=-1)
        attn_comp = self.dropout(attn_comp)
        out_comp = (attn_comp @ v_comp).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        (
            selected_keys,
            selected_values,
            selected_mask,
            selected_positions,
        ) = self._select_tokens(
            queries,
            keys,
            values,
            compressed_keys,
            compressed_values,
            batch_size,
            seq_len,
        )

        q_sel = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_sel = selected_keys.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_sel = selected_values.view(batch_size, -1, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_kv_heads < self.num_heads:
            k_sel = k_sel.repeat_interleave(self.num_kv_groups, dim=1)
            v_sel = v_sel.repeat_interleave(self.num_kv_groups, dim=1)

        scores_sel = q_sel @ k_sel.transpose(2, 3) / math.sqrt(self.head_dim)

        scores_sel = scores_sel.masked_fill(~selected_mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        if self.use_causal:
            q_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).unsqueeze(-1)
            causal_mask_sel = q_positions >= selected_positions.unsqueeze(1)
            causal_mask_sel = causal_mask_sel.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores_sel = scores_sel.masked_fill(~causal_mask_sel, float("-inf"))

        attn_sel = F.softmax(scores_sel, dim=-1)
        attn_sel = self.dropout(attn_sel)
        out_select = (attn_sel @ v_sel).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        window_mask = self._create_sliding_window_mask(seq_len)
        q_win = queries.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_win = keys.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v_win = values.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.n_kv_heads < self.num_heads:
            k_win = k_win.repeat_interleave(self.num_kv_groups, dim=1)
            v_win = v_win.repeat_interleave(self.num_kv_groups, dim=1)

        scores_win = q_win @ k_win.transpose(2, 3) / math.sqrt(self.head_dim)
        scores_win = scores_win.masked_fill(window_mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
        attn_win = F.softmax(scores_win, dim=-1)
        attn_win = self.dropout(attn_win)
        out_win = (attn_win @ v_win).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_out)

        gates = torch.sigmoid(self.w_gate(x))  # [batch, seq_len, 3]
        out = gates[:, :, 0:1] * out_comp + gates[:, :, 1:2] * out_select + gates[:, :, 2:3] * out_win
        out = self.out_proj(out)

        return out
