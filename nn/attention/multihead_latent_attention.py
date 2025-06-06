import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from nn.norms import RMSNorm
from nn.rope import RotaryPositionalEmbeddings


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates half the hidden dims of the input for RoPE."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(
    x: torch.Tensor, 
    freqs_cis: torch.Tensor,
    dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Apply rotary embeddings to input tensor using complex number multiplication.
    
    Args:
        x: Input tensor of shape [batch_size, seq_len, n_heads, head_dim]
        freqs_cis: Precomputed frequency tensor (complex)
        dtype: Output dtype
        
    Returns:
        Tensor with rotary embeddings applied
    """
    x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshape)
    
    freqs_cis = freqs_cis.unsqueeze(2) if x.dim() == 4 else freqs_cis
    x_rotated = x_complex * freqs_cis
    
    x_out = torch.view_as_real(x_rotated)
    x_out = x_out.reshape(*x.shape)
    
    return x_out.to(dtype if dtype is not None else x.dtype)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    rope_factor: float = 1.0
) -> torch.Tensor:
    """
    Precompute the frequency tensor for rotary embeddings.
    
    Args:
        dim: Dimension of the embeddings
        max_seq_len: Maximum sequence length
        theta: Base for the frequency calculation
        rope_factor: Scaling factor for RoPE
        
    Returns:
        Complex tensor with precomputed frequencies
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    
    if rope_factor != 1.0:
        t = t / rope_factor
        
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key/value heads for multi-query attention.
    From (batch, n_kv_heads, seq_len, head_dim) to (batch, n_heads, seq_len, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    batch, seq_len, n_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(
        batch, seq_len, n_kv_heads, n_rep, head_dim
    )
    return hidden_states.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)


class MultiHeadLatentAttention(nn.Module):    
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
        q_lora_rank: Optional[int] = None,
        kv_lora_rank: Optional[int] = None,
        qk_rope_head_dim: int = 64,
        qk_nope_head_dim: Optional[int] = None,
        v_head_dim: Optional[int] = None,
        rope_theta: float = 10000.0,
        softcap: Optional[float] = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.n_kv_groups = self.num_heads // self.n_kv_heads
        self.head_dim = d_out // num_heads
        self.v_head_dim = v_head_dim if v_head_dim is not None else self.head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim if qk_nope_head_dim is not None else (self.head_dim - qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank if q_lora_rank else 0
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else d_in // 2

        if self.q_lora_rank == 0:
            self.w_query = nn.Linear(d_in, self.num_heads * self.qk_head_dim, dtype=dtype, bias=qkv_bias)
        else:
            self.wq_a = nn.Linear(d_in, self.q_lora_rank, dtype=dtype, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, dtype=dtype)
            self.w_query = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, dtype=dtype, bias=qkv_bias)

        self.wkv_a = nn.Linear(d_in, self.kv_lora_rank + self.qk_rope_head_dim, dtype=dtype, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, dtype=dtype)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_kv_heads * (self.qk_nope_head_dim + self.v_head_dim),
            dtype=dtype, bias=qkv_bias
        )
        self.out_proj = nn.Linear(self.num_heads * self.v_head_dim, d_out, dtype=dtype, bias=False)
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.softcap = softcap
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.qk_rope_head_dim, max_seq_len=max_seq_len
            )
        self.register_buffer(
            "causal_mask", 
            torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if self.q_lora_rank == 0:
            queries = self.w_query(x)
        else:
            queries = self.w_query(self.q_norm(self.wq_a(x)))

        queries = queries.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        kv = self.wkv_a(x)
        kv_compressed, k_pe = torch.split(
            kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1
        )
        kv_decompressed = self.wkv_b(self.kv_norm(kv_compressed))
        kv_decompressed = kv_decompressed.view(
            batch_size, seq_len, self.n_kv_heads, 
            self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, values = torch.split(
            kv_decompressed,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1
        )
        
        k_pe = k_pe.unsqueeze(2).expand(-1, -1, self.n_kv_heads, -1)
    
        if self.qk_nope_head_dim > 0:
            q_nope, q_pe = torch.split(
                queries, 
                [self.qk_nope_head_dim, self.qk_rope_head_dim], 
                dim=-1
            )
            if self.use_rope:
                q_pe = self.rope(q_pe)
                k_pe = self.rope(k_pe)

            queries = torch.cat([q_nope, q_pe], dim=-1)
            keys = torch.cat([k_nope, k_pe], dim=-1)
        else:
            q_pe = queries 
            
            if self.use_rope:
                q_pe = self.rope(q_pe)
                k_pe = self.rope(k_pe)
            
            queries = q_pe
            keys = k_pe
        
        queries = queries.transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        keys = keys.transpose(1, 2)        # [batch, n_kv_heads, seq_len, head_dim]
        values = values.transpose(1, 2)    # [batch, n_kv_heads, seq_len, v_head_dim]
        
        if self.n_kv_groups > 1:
            keys = keys.repeat_interleave(self.n_kv_groups, dim=1)
            values = values.repeat_interleave(self.n_kv_groups, dim=1)
        
        scores = torch.matmul(queries, keys.transpose(2, 3)) * self.softmax_scale
        
        if self.softcap is not None:
            scores = scores / self.softcap
            scores = torch.tanh(scores)
            scores = scores * self.softcap
        
        mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 1, float("-inf"))

        attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(x.dtype)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, values)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.num_heads * self.v_head_dim)
        output = self.out_proj(attn_output)
        
        return output
