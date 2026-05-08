import math

import torch


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    if x.size(-1) % 2 != 0:
        raise ValueError(f"Last dimension must be even, got {x.size(-1)}")

    x = torch.view_as_complex(x.float().view(*x.shape[:-1], -1, 2))

    if len(freqs_cis.shape) == 2:  # [seq_len, dim//2]
        freqs_cis = freqs_cis.view(1, freqs_cis.size(0), 1, freqs_cis.size(1))
    elif len(freqs_cis.shape) == 4:
        pass
    else:
        raise ValueError(f"Unexpected freqs_cis shape: {freqs_cis.shape}")

    if freqs_cis.size(-1) != x.size(-1):
        freqs_cis = freqs_cis[..., : x.size(-1)]

    y = torch.view_as_real(x * freqs_cis).flatten(3)
    return y.to(dtype)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    dtype = q.dtype
    rot_dim = cos.shape[-1]
    q_, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q_ * cos) + (rotate_half(q_) * sin)
    k_embed = (k_ * cos) + (rotate_half(k_) * sin)
    return torch.cat((q_embed, q_pass), dim=-1).to(dtype), torch.cat((k_embed, k_pass), dim=-1).to(dtype)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, seq_len, n_kv_heads, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, :, None, :].expand(batch, seq_len, n_kv_heads, n_rep, head_dim)
    return hidden_states.reshape(batch, seq_len, n_kv_heads * n_rep, head_dim)


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    seq_len: int,
    beta_fast: int,
    beta_slow: int,
    theta: float = 10000.0,
    rope_factor: float = 1.0,
) -> torch.Tensor:
    """
    Precompute the frequency tensor for rotary embeddings.

    Args:
        dim: Full dimension of the rope embeddings (not dim//2)
        max_seq_len: Maximum sequence length
        seq_len: Current sequence length
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
        return max(low, 0), min(high, dim - 1)

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

    freq_dim = dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, freq_dim * 2, 2, dtype=torch.float32) / dim))

    if seq_len > max_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, theta, max_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, freq_dim)
        freqs = freqs / rope_factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seq_len)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis
