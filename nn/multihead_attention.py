import torch
import torch.nn as nn
import math
from typing import Optional
from nn.rope import RotaryPositionalEmbeddings


class MultiHeadAttention(nn.Module):
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
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
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
        keys = self.w_key(x)  # shape (2, 6, 4)
        queries = self.w_query(x)  # shape (2, 6, 4)
        values = self.w_value(x)  # shape (2, 6, 4)

        # shape: (batch_size, seq_len, n_heads, head_dim)
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        values = values.view(
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


class NormalizedMultiHeadAttention(nn.Module):
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
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.head_dim, max_seq_len=self.max_seq_len
            )
        self.register_buffer(
            "mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        )
        self.sq_init_value = 1.0
        self.sq_init_scaling = 1.0 / math.sqrt(d_in)
        self.sq = nn.Parameter(
            torch.empty(self.head_dim * self.num_heads)
        )
        self.sk_init_value = 1.0
        self.sk_init_scaling = 1.0 / math.sqrt(d_in)
        self.sk = nn.Parameter(
            torch.empty(self.head_dim * self.n_kv_heads)
        )

        self.sqrt_head_dim = math.sqrt(self.head_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sq)
        nn.init.ones_(self.sk)
        with torch.no_grad():
            self.sq.mul_(self.sq_init_scaling)
            self.sk.mul_(self.sk_init_scaling)

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        keys = self.w_key(x)  # shape (2, 6, 4)
        queries = self.w_query(x)  # shape (2, 6, 4)
        values = self.w_value(x)  # shape (2, 6, 4)

        sq = (self.sq * (self.sq_init_value / self.sq_init_scaling)).view(1, 1, -1)
        queries = sq * queries
        sk = (self.sk * (self.sk_init_value / self.sk_init_scaling)).view(1, 1, -1)
        keys = sk * keys

        # shape: (batch_size, seq_len, n_heads, head_dim)
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        values = values.view(
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

    @torch.no_grad()
    def normalize_matrices(self):
        self._normalize_matrix(self.w_query.weight)
        self._normalize_matrix(self.w_key.weight)
        self._normalize_matrix(self.w_value.weight)
        self._normalize_matrix(self.out_proj.weight, dim=0)
    
    @staticmethod
    def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).type_as(x)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(self.l2_normalize(w, dim=dim))


class muPMultiHeadAttention(nn.Module):
    """
    1. Q and K matrices have O(1/√d) initialization and learning rate
    2. V and O matrices have O(1/√d) initialization but O(1) learning rate  
    3. Attention logits are scaled by 1/√d_head
    """
    
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
        mup_base_d_model: Optional[int] = None,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.w_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else num_heads
        self.use_rope = use_rope
        self.max_seq_len = max_seq_len
        if use_rope:
            self.rope = RotaryPositionalEmbeddings(
                dim=self.head_dim, max_seq_len=self.max_seq_len
            )
        
        self.register_buffer(
            "mask", torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1)
        )       
        self.mup_base_d_model = mup_base_d_model or d_in
        self.mup_scale = math.sqrt(self.mup_base_d_model / d_in)

        self.base_shapes = {
            "w_query.weight": self.w_query.weight.shape,
            "w_key.weight": self.w_key.weight.shape, 
            "w_value.weight": self.w_value.weight.shape,
            "w_out.weight": self.out_proj.weight.shape,
        }
        
        self._init_weights()
    
    def _init_weights(self):
        std_qk = 1.0 / math.sqrt(self.d_out)
        nn.init.normal_(self.w_query.weight, mean=0.0, std=std_qk)
        nn.init.normal_(self.w_key.weight, mean=0.0, std=std_qk)
        std_v = 1.0 / math.sqrt(self.d_out)
        nn.init.normal_(self.w_value.weight, mean=0.0, std=std_v)
        std_out = 1.0 / math.sqrt(self.d_out)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=std_out)
        if self.w_query.bias is not None:
            nn.init.zeros_(self.w_query.bias)
        if self.w_key.bias is not None:
            nn.init.zeros_(self.w_key.bias)
        if self.w_value.bias is not None:
            nn.init.zeros_(self.w_value.bias)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        keys = self.w_key(x)  # shape (2, 6, 4)
        queries = self.w_query(x)  # shape (2, 6, 4)
        values = self.w_value(x)  # shape (2, 6, 4)

        # shape: (batch_size, seq_len, n_heads, head_dim)
        queries = queries.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        keys = keys.view(
            batch_size, num_tokens, self.num_heads, self.head_dim
        )  # New shape: (2, 6, 2, 2)
        values = values.view(
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
        
        attn_scores = queries @ keys.transpose(2, 3) / (math.sqrt(self.head_dim) * self.mup_scale)
        
        mask_bool = self.mask.to(torch.bool)[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2).contiguous()
        context_vec = context_vec.view(batch_size, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec

    def get_mup_lr_scale(self, param_name: str, base_lr: float) -> float:
        if 'w_query.weight' in param_name or 'w_key.weight' in param_name:
            return base_lr / self.d_out
        elif 'w_value.weight' in param_name or 'out_proj.weight' in param_name:
            return base_lr
        else:
            return base_lr
