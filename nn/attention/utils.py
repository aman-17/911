import math
from collections import defaultdict
from typing import Dict, Tuple, Type

import torch
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
)


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


def _get_custom_checkpoint_policy(meta: Dict[str, int]):
    # Adapted from
    # https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py
    from torch.utils.checkpoint import CheckpointPolicy

    _save_list = {
        torch.ops.aten.mm.default,  # type: ignore
        torch.ops.aten._scaled_dot_product_efficient_attention.default,  # type: ignore
        torch.ops.aten._scaled_dot_product_flash_attention.default,  # type: ignore
        torch.ops._c10d_functional.reduce_scatter_tensor.default,  # type: ignore
        # for low precision training, it's useful to always save
        # the result of max(abs(tensor))
        torch.ops.aten.abs.default,  # type: ignore
        torch.ops.aten.max.default,  # type: ignore
    }

    def _custom_policy(ctx, func, *args, **kwargs):
        del args, kwargs
        mode = "recompute" if ctx.is_recompute else "forward"
        mm_count_key = f"{mode}_mm_count"
        if func == torch.ops.aten.mm.default:  # type: ignore
            meta[mm_count_key] += 1
        # Saves output of all compute ops, except every second mm
        to_save = func in _save_list and not (func == torch.ops.aten.mm.default and meta[mm_count_key] % 2 == 0)  # type: ignore
        return CheckpointPolicy.MUST_SAVE if to_save else CheckpointPolicy.PREFER_RECOMPUTE

    return _custom_policy


def selective_checkpointing_context_fn():
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    meta: Dict[str, int] = defaultdict(int)
    return create_selective_checkpoint_contexts(_get_custom_checkpoint_policy(meta))


def get_tp_wrappers(
    float8_enabled: bool,
) -> Tuple[Type[RowwiseParallel], Type[ColwiseParallel], Type[PrepareModuleInput]]:
    if not float8_enabled:
        return (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )
    else:
        # TODO (epwalsh): once float8 configuration supports delayed scaling,
        # add a check here to enforce supported float8 all-gather configurations.
        from torchao.float8.float8_tensor_parallel import (  # type: ignore
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        return (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
