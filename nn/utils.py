import math
from typing import Optional
import torch
import torch.nn as nn


def ensure_multiple_of(x: int, of: int) -> int:
    return of * math.ceil(x / of)


def autocast_precision(precision) -> torch.dtype:
    if isinstance(precision, torch.dtype):
        return precision

    if precision == "bf16":
        return torch.bfloat16
    elif precision == "fp16":
        return torch.float16
    elif precision == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unexpected precision type '{precision}'")


def generate_text_simple(
    model: nn.Module, idx: torch.Tensor, max_new_tokens: int, context_size: int, use_cache: Optional[bool] = False
) -> list[int]:
    """
    Generate text using a simple greedy sampling strategy.

    Args:
        model: The language model
        idx: Input token indices
        max_new_tokens: Number of tokens to generate
        context_size: Size of the context window

    Returns:
        torch.Tensor: Generated token indices
    """

    if not isinstance(idx, torch.Tensor):
        idx = torch.tensor(idx, dtype=torch.long)

    device = next(model.parameters()).device
    idx = idx.to(device).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            # if use_cache:
            #     model.reset_kv_cache()
            #     logits = model(idx_cond, use_cache=True)
            # else:
            #     logits = model(idx_cond, use_cache=False)
            logits = model(idx_cond, use_cache=use_cache)
            logits = logits[:, -1, :]
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            idx = torch.cat((idx, idx_next), dim=1)
    return idx[0].cpu().tolist()
