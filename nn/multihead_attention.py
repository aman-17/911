import torch
import torch.nn as nn

from nn.rope import RotaryPositionalEmbeddings


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        max_seq_len: int,
        num_heads: int,
        dropout: float,
        qkv_bias: bool = False,
        use_rope: bool = True,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.head_dim, max_seq_len=self.max_seq_len
            )
        self.register_buffer(
            "mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        keys = self.W_key(x)  # shape (2, 6, 4)
        queries = self.W_query(x)  # shape (2, 6, 4)
        values = self.W_value(x)  # shape (2, 6, 4)

        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        values = values.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        queries = queries.view(
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
