import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedMultiHeadAttention(nn.Module):
    """
    Gated multi-head attention from https://arxiv.org/pdf/2505.06708.

    - headwise_gate: scalar gate per head (query-dependent, low param overhead).
    - elementwise_gate: element-wise gate across full head dim (higher capacity).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        headwise_gate: bool = False,
        elementwise_gate: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        if headwise_gate and elementwise_gate:
            raise ValueError("Cannot enable both headwise_gate and elementwise_gate simultaneously.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else embed_dim // num_heads

        if self.head_dim * num_heads != self.embed_dim:
            raise ValueError(f"embed_dim must be divisible by num_heads (got {embed_dim} / {num_heads} = {self.head_dim})")

        self.headwise_gate = headwise_gate
        self.elementwise_gate = elementwise_gate

        if self.headwise_gate:
            self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim + num_heads, bias=bias)
        elif self.elementwise_gate:
            self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim * 2, bias=bias)
        else:
            self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)

        self.k_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=bias)

        self.inv_sqrt_head_dim = 1.0 / math.sqrt(self.head_dim)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        gate_logits: Optional[torch.Tensor] = None
        if self.headwise_gate:
            query_states, gate_logits = torch.split(query_states, [self.num_heads * self.head_dim, self.num_heads], dim=-1)
            gate_logits = gate_logits.unsqueeze(-1)
        elif self.elementwise_gate:
            query_states, gate_logits = torch.split(query_states, [self.num_heads * self.head_dim, self.num_heads * self.head_dim], dim=-1)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query_states, key_states.transpose(2, 3)) * self.inv_sqrt_head_dim
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        if gate_logits is not None:
            gate = torch.sigmoid(gate_logits.view(bsz, seq_len, self.num_heads, -1).transpose(1, 2))
            attn_output = attn_output * gate

        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)
        return self.o_proj(attn_output)
