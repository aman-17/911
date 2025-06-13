import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

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
    seq_len:int,
    beta_fast: int,
    beta_slow: int,
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
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))
    
    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)
    
    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func
    
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if seq_len > max_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, theta, max_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seq_len)
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
        original_seq_len: int,
        num_heads: int,
        batch_size: int,
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
        mscale: float = 1,
        rope_factor: float = 40,
        softcap: Optional[float] = None,
        attn_impl: Optional[str] = "naive",
    ):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.num_heads = num_heads
        self.n_local_heads = num_heads // torch.distributed.get_world_size()
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.n_kv_groups = self.num_heads // self.n_kv_heads
        self.head_dim = d_out // num_heads
        self.v_head_dim = v_head_dim if v_head_dim is not None else self.head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim if qk_nope_head_dim is not None else (self.head_dim - qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.q_lora_rank = q_lora_rank if q_lora_rank else 0
        self.kv_lora_rank = kv_lora_rank if kv_lora_rank is not None else d_in // 2
        self.mscale = mscale
        self.rope_factor = rope_factor
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.attn_impl = attn_impl

        if self.q_lora_rank == 0:
            self.wq = nn.Linear(d_in, self.num_heads * self.qk_head_dim, dtype=dtype, bias=qkv_bias)
        else:
            self.wq_a = nn.Linear(d_in, self.q_lora_rank, dtype=dtype, bias=False)
            self.q_norm = RMSNorm(self.q_lora_rank, dtype=dtype)
            self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, dtype=dtype, bias=qkv_bias)
        self.wkv_a = nn.Linear(d_in, self.kv_lora_rank + self.qk_rope_head_dim, dtype=dtype, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, dtype=dtype)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_kv_heads * (self.qk_nope_head_dim + self.v_head_dim),
            dtype=dtype, bias=qkv_bias
        )
        self.wo = nn.Linear(self.num_heads * self.v_head_dim, d_out, dtype=dtype, bias=False)
        self.softmax_scale = self.qk_head_dim ** -0.5
        self.softcap = softcap
        if max_seq_len > original_seq_len:
            mscale = 0.1 * self.mscale * math.log(self.rope_factor) + 1.0
            self.softmax_scale = self.softmax_scale * mscale * mscale
        
        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(self.batch_size, self.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(self.batch_size, self.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            self.register_buffer("kv_cache", torch.zeros(self.batch_size, self.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(self.batch_size, self.max_seq_len, self.qk_rope_head_dim), persistent=False)

        self.use_rope = use_rope
        
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.qk_rope_head_dim, max_seq_len=self.max_seq_len
            )
        self.register_buffer(
            "causal_mask", 
            torch.triu(torch.ones(self.max_seq_len, self.max_seq_len), diagonal=1)
        )
        
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        end_pos = start_pos + seq_len
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            q = self.wq(self.q_norm(self.wq_a(x)))

        q = q.view(batch_size, seq_len, self.n_local_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        kv = self.wkv_a(x)
        kv, k_pe = torch.split(kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        
        if self.attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(batch_size, seq_len, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
        
            self.k_cache[:batch_size, start_pos:end_pos] = k
            self.v_cache[:batch_size, start_pos:end_pos] = v
            
            q_reshaped = q.transpose(1, 2)  # [batch_size, n_heads, seq_len, head_dim]
            k_cached = self.k_cache[:batch_size, :end_pos].transpose(1, 2)  # [batch_size, n_heads, end_pos, head_dim]
            scores = torch.matmul(q_reshaped, k_cached.transpose(-2, -1))  # [batch_size, n_heads, seq_len, end_pos]
            scores = scores.transpose(1, 2) * self.softmax_scale  # [batch_size, seq_len, n_heads, end_pos]
            
        else:
            # wkv_b = self.wkv_b.weight if self.wkv_b.scale is None else weight_dequant(self.wkv_b.weight, self.wkv_b.scale, block_size)
            wkv_b = self.wkv_b.weight
            wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
            
            q_nope_reshaped = q_nope.reshape(batch_size * seq_len, self.n_local_heads, self.qk_nope_head_dim)
            wkv_b_q = wkv_b[:, :self.qk_nope_head_dim]  # [n_heads, qk_nope_head_dim, kv_lora_rank]
            
            q_nope_proj = []
            for h in range(self.n_local_heads):
                # [batch_size*seq_len, qk_nope_head_dim] @ [qk_nope_head_dim, kv_lora_rank]
                proj = torch.matmul(q_nope_reshaped[:, h], wkv_b_q[h])
                q_nope_proj.append(proj)
            q_nope = torch.stack(q_nope_proj, dim=1).reshape(batch_size, seq_len, self.n_local_heads, self.kv_lora_rank)
            
            self.kv_cache[:batch_size, start_pos:end_pos] = self.kv_norm(kv)
            self.pe_cache[:batch_size, start_pos:end_pos] = k_pe.squeeze(2)
            
            q_nope_reshaped = q_nope.transpose(1, 2)  # [batch_size, n_heads, seq_len, kv_lora_rank]
            kv_cached = self.kv_cache[:batch_size, :end_pos]  # [batch_size, end_pos, kv_lora_rank]
            kv_cached = kv_cached.unsqueeze(1).expand(-1, self.n_local_heads, -1, -1)  # [batch_size, n_heads, end_pos, kv_lora_rank]
            scores1 = torch.matmul(q_nope_reshaped, kv_cached.transpose(-2, -1))  # [batch_size, n_heads, seq_len, end_pos]
            q_pe_reshaped = q_pe.transpose(1, 2)  # [batch_size, n_heads, seq_len, rope_head_dim]
            pe_cached = self.pe_cache[:batch_size, :end_pos]  # [batch_size, end_pos, rope_head_dim]
            pe_cached = pe_cached.unsqueeze(1).expand(-1, self.n_local_heads, -1, -1)  # [batch_size, n_heads, end_pos, rope_head_dim]
            scores2 = torch.matmul(q_pe_reshaped, pe_cached.transpose(-2, -1))  # [batch_size, n_heads, seq_len, end_pos]
            scores = (scores1 + scores2).transpose(1, 2) * self.softmax_scale  # [batch_size, seq_len, n_heads, end_pos]
        
        if mask is not None:
            scores += mask.unsqueeze(1)
        scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)

        if self.attn_impl == "naive":
            scores_reshaped = scores.transpose(1, 2)  # [batch_size, n_heads, seq_len, end_pos]
            v_cached = self.v_cache[:batch_size, :end_pos].transpose(1, 2)  # [batch_size, n_heads, end_pos, v_head_dim]
            x = torch.matmul(scores_reshaped, v_cached)  # [batch_size, n_heads, seq_len, v_head_dim]
            x = x.transpose(1, 2)  # [batch_size, seq_len, n_heads, v_head_dim]
            
        else:
            scores_reshaped = scores.transpose(1, 2)  # [batch_size, n_heads, seq_len, end_pos]
            kv_cached = self.kv_cache[:batch_size, :end_pos]  # [batch_size, end_pos, kv_lora_rank]
            kv_cached = kv_cached.unsqueeze(1).expand(-1, self.n_local_heads, -1, -1)  # [batch_size, n_heads, end_pos, kv_lora_rank]
            x = torch.matmul(scores_reshaped, kv_cached)  # [batch_size, n_heads, seq_len, kv_lora_rank]
            x = x.transpose(1, 2)  # [batch_size, seq_len, n_heads, kv_lora_rank]
            x_reshaped = x.reshape(batch_size * seq_len, self.n_local_heads, self.kv_lora_rank)
            wkv_b_v = wkv_b[:, -self.v_head_dim:]  # [n_heads, v_head_dim, kv_lora_rank]
            
            x_proj = []
            for h in range(self.n_local_heads):
                # [batch_size*seq_len, kv_lora_rank] @ [kv_lora_rank, v_head_dim]
                proj = torch.matmul(x_reshaped[:, h], wkv_b_v[h].transpose(0, 1))
                x_proj.append(proj)
            x = torch.stack(x_proj, dim=1).reshape(batch_size, seq_len, self.n_local_heads, self.v_head_dim)
        
        x = self.wo(x.flatten(2))
        return x
