import math
from typing import Dict, Optional, Tuple

import torch
from torch import nn


class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 2048) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(
        self, x: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor with shape
                ``[b, s, n_h, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            torch.Tensor: output tensor with shape ``[b, s, n_h, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        # input tensor has shape [b, s, n_h, h_d]
        seq_len = x.size(1)

        # extract the values based on whether input_pos is set or not
        rope_cache = (
            self.cache[:seq_len] if input_pos is None else self.cache[input_pos]
        )

        # reshape input; the last dimension is used for computing the output.
        # Cast to float to match the reference implementation
        # tensor has shape [b, s, n_h, h_d // 2, 2]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

        # reshape the cache for broadcasting
        # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
        # otherwise has shape [1, s, 1, h_d // 2, 2]
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

        # tensor has shape [b, s, n_h, h_d // 2, 2]
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )

        # tensor has shape [b, s, n_h, h_d]
        x_out = x_out.flatten(3)
        # print(x_out)
        return x_out.type_as(x)

    def compute_rope_params(self, dim: int, base: int, max_seq_len: int, dtype=torch.float32):
        assert dim % 2 == 0, "Embedding dimension must be even"
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=dtype)[: (dim // 2)].float() / dim))
        positions = torch.arange(max_seq_len, dtype=dtype)
        angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)
        angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)
        cos = torch.cos(angles)
        sin = torch.sin(angles)
        return cos, sin


class ComplexRotaryEmbedding(nn.Module):
    """
    An implementation of `RoPE <https://arxiv.org/abs/2104.09864>`_ as a rotation in complex space.

    :param head_size: The dimensionality of the attention heads.
    :param theta: The theta base value to use.
    :param full_precision: Always apply RoPE in full precision regardless of the input data type.
    """

    def __init__(
        self,
        *,
        head_size: int,
        theta: int = 500_000,
        full_precision: bool = True,
    ):
        super().__init__()
        self.dim = head_size
        self.theta = theta
        self.full_precision = full_precision
        self._cache: Dict[str, torch.Tensor] = {}

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_rotary_embedding(max_seq_len, device)

    def get_buffers(self, max_seq_len: int, device: torch.device):
        freqs_cis = self._get_rotary_embedding(max_seq_len, device)
        return freqs_cis

    def _get_rotary_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (
            freqs_cis := self._cache.get("rope_freqs_cis")
        ) is not None and freqs_cis.shape[-2] >= seq_len:
            if freqs_cis.device != device:
                freqs_cis = freqs_cis.to(device)
                self._cache["rope_freqs_cis"] = freqs_cis
            return freqs_cis[:seq_len, :]

        with torch.autocast(device.type, enabled=False):
            inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(0, self.dim, 2, device=device, dtype=torch.float)[
                        : (self.dim // 2)
                    ]
                    / self.dim
                )
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self._cache["rope_freqs_cis"] = freqs_cis
        return freqs_cis

    def _apply_rotary_pos_emb(
        self, freqs_cis: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return torch.view_as_real(x * freqs_cis).flatten(3)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        head_first: bool = True,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query (``q``) and key (``k``) matrices.

        :param q: The query matrix of shape ``(batch_size, num_heads, seq_len, head_size)``
            if ``head_first`` (the default) otherwise ``(batch_size, seq_len, num_heads, head_size)``.
        :param k: The key matrix of shape ``(batch_size, num_kv_heads, seq_len, head_size)``
            if ``head_first`` (the default) otherwise
            ``(batch_size, seq_len, num_kv_heads, head_size)``.
        :param head_first: If the head dim comes before the sequence dim.

        :returns: The query and key matrices after RoPE has been applied.
        """
        if pos_sin is not None or pos_cos is not None:
            raise RuntimeError(
                f"'pos_sin' and 'pos_cos' are invalid for {self.__class__.__name__}"
            )

        if head_first:
            q_len = q.size(2)
            k_len = k.size(2)
        else:
            q_len = q.size(1)
            k_len = k.size(1)

        if self.full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        # shape (complex64):
        #  (B, nh, T, hs // 2), (B, n_kv_h, T, hs // 2) if `head_first`, else
        #  (B, T, nh, hs // 2), (B, T, n_kv_h, hs // 2)
        q_ = torch.view_as_complex(q_.reshape(*q_.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k_.reshape(*k_.shape[:-1], -1, 2))

        with torch.autocast(q.device.type, enabled=False):
            # shape: (T, hs // 2)
            if freqs_cis is None:
                freqs_cis = self._get_rotary_embedding(k_len, q_.device)
            if head_first:
                # shape: (1, 1, T, hs // 2)
                q_ = self._apply_rotary_pos_emb(
                    freqs_cis[None, None, k_len - q_len : k_len, :],
                    q_,
                )
                k_ = self._apply_rotary_pos_emb(freqs_cis[None, None, :, :], k_)
            else:
                # shape: (1, T, 1, hs // 2)
                q_ = self._apply_rotary_pos_emb(
                    freqs_cis[None, k_len - q_len : k_len, None, :],
                    q_,
                )
                k_ = self._apply_rotary_pos_emb(freqs_cis[None, :, None, :], k_)

        return q_.type_as(q), k_.type_as(k)


class YaRNRotaryEmbedding(nn.Module):
    """
    An implementation of `RoPE <https://openreview.net/forum?id=wHBfxhZu1u>`_ as a rotation in complex space.
    This version includes YaRN-style improvements for handling longer sequences.

    :param head_size: The dimensionality of the attention heads.
    :param theta: The theta base value to use.
    :param full_precision: Always apply RoPE in full precision regardless of the input data type.
    :param max_seq_len: Maximum sequence length for computing corrections.
    :param rope_factor: Scaling factor for RoPE.
    :param beta_fast: Fast beta value for YaRN corrections.
    :param beta_slow: Slow beta value for YaRN corrections.
    """

    def __init__(
        self,
        *,
        head_size: int,
        theta: float = 10000.0,
        full_precision: bool = True,
        max_seq_len: int = 4096,
        rope_factor: float = 1.0,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ):
        super().__init__()
        self.dim = head_size
        self.theta = theta
        self.full_precision = full_precision
        self.max_seq_len = max_seq_len
        self.rope_factor = rope_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self._cache: Dict[str, torch.Tensor] = {}

    def warmup_cache(self, max_seq_len: int, device: torch.device):
        self._get_rotary_embedding(max_seq_len, device)

    def get_buffers(self, max_seq_len: int, device: torch.device):
        freqs_cis = self._get_rotary_embedding(max_seq_len, device)
        return freqs_cis

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input for RoPE."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _find_correction_dim(
        self, num_rotations: float, dim: int, base: float, max_seq_len: int
    ) -> float:
        """
        Computes the correction dimension for a given number of rotations.
        """
        return (
            dim
            * math.log(max_seq_len / (num_rotations * 2 * math.pi))
            / (2 * math.log(base))
        )

    def _find_correction_range(
        self, low_rot: float, high_rot: float, dim: int, base: float, max_seq_len: int
    ) -> Tuple[int, int]:
        """
        Computes the range of correction dimensions for rotary positional embeddings.
        """
        low = math.floor(self._find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(self._find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def _linear_ramp_factor(
        self, min_val: float, max_val: float, dim: int, device: torch.device
    ) -> torch.Tensor:
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.
        """
        if min_val == max_val:
            max_val += 0.001
        linear_func = (
            torch.arange(dim, dtype=torch.float32, device=device) - min_val
        ) / (max_val - min_val)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    def _get_rotary_embedding(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (
            freqs_cis := self._cache.get("rope_freqs_cis")
        ) is not None and freqs_cis.shape[-2] >= seq_len:
            if freqs_cis.device != device:
                freqs_cis = freqs_cis.to(device)
                self._cache["rope_freqs_cis"] = freqs_cis
            return freqs_cis[:seq_len, :]

        with torch.autocast(device.type, enabled=False):
            inv_freq = 1.0 / (
                self.theta
                ** (
                    torch.arange(0, self.dim, 2, device=device, dtype=torch.float)[
                        : (self.dim // 2)
                    ]
                    / self.dim
                )
            )

            if seq_len > self.max_seq_len:
                low, high = self._find_correction_range(
                    self.beta_fast,
                    self.beta_slow,
                    self.dim,
                    self.theta,
                    self.max_seq_len,
                )
                smooth = 1 - self._linear_ramp_factor(low, high, self.dim // 2, device)
                inv_freq = (
                    inv_freq * self.rope_factor * (1 - smooth)
                    + inv_freq / self.rope_factor * smooth
                )

            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.einsum("i , j -> i j", seq, inv_freq)
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        self._cache["rope_freqs_cis"] = freqs_cis
        return freqs_cis

    def _apply_rotary_pos_emb(
        self, freqs_cis: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        return torch.view_as_real(x * freqs_cis).flatten(3)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        head_first: bool = True,
        pos_sin: Optional[torch.Tensor] = None,
        pos_cos: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply RoPE to query (``q``) and key (``k``) matrices.

        :param q: The query matrix of shape ``(batch_size, num_heads, seq_len, head_size)``
            if ``head_first`` (the default) otherwise ``(batch_size, seq_len, num_heads, head_size)``.
        :param k: The key matrix of shape ``(batch_size, num_kv_heads, seq_len, head_size)``
            if ``head_first`` (the default) otherwise
            ``(batch_size, seq_len, num_kv_heads, head_size)``.
        :param head_first: If the head dim comes before the sequence dim.

        :returns: The query and key matrices after RoPE has been applied.
        """
        if pos_sin is not None or pos_cos is not None:
            raise RuntimeError(
                f"'pos_sin' and 'pos_cos' are invalid for {self.__class__.__name__}"
            )

        if head_first:
            q_len = q.size(2)
            k_len = k.size(2)
        else:
            q_len = q.size(1)
            k_len = k.size(1)

        if self.full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        #  (B, nh, T, hs // 2), (B, n_kv_h, T, hs // 2) if `head_first`, else (B, T, nh, hs // 2), (B, T, n_kv_h, hs // 2)
        q_ = torch.view_as_complex(q_.reshape(*q_.shape[:-1], -1, 2))
        k_ = torch.view_as_complex(k_.reshape(*k_.shape[:-1], -1, 2))

        with torch.autocast(q.device.type, enabled=False):
            # shape: (T, hs // 2)
            if freqs_cis is None:
                freqs_cis = self._get_rotary_embedding(k_len, q_.device)
            if head_first:
                # shape: (1, 1, T, hs // 2)
                q_ = self._apply_rotary_pos_emb(
                    freqs_cis[None, None, k_len - q_len : k_len, :],
                    q_,
                )
                k_ = self._apply_rotary_pos_emb(freqs_cis[None, None, :, :], k_)
            else:
                # shape: (1, T, 1, hs // 2)
                q_ = self._apply_rotary_pos_emb(
                    freqs_cis[None, k_len - q_len : k_len, None, :],
                    q_,
                )
                k_ = self._apply_rotary_pos_emb(freqs_cis[None, :, None, :], k_)

        return q_.type_as(q), k_.type_as(k)
