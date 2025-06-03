# from: https://github.com/fxmeng/TransMLA/blob/main/transmla/transformers/mla.py

from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
import logging
logger = logging.getLogger(__name__)

from nn.rope import RotaryPositionalEmbeddings

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
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
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling is None:
        scaling = module.head_dim**-0.5

    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling

    if softcap is not None:
        attn_weights = attn_weights / softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * softcap
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class MLAAttention(nn.Module):
    """
    Modified from `transformers.models.llama.modeling_deepseek_v3.DeepseekV3Attention`
    add support for attention bias and softcapping
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_heads = config['n_heads']
        self.num_key_value_heads = config.get('n_kv_heads', config['n_heads'])
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.get('drop_rate', 0.1)
        self.rope_theta = config.get('rope_theta', 10000.0)
        self.q_lora_rank = config.get('q_lora_rank', None)
        self.kv_lora_rank = config.get('kv_lora_rank', config['emb_dim'] // 2)
        self.qk_rope_head_dim = config.get('qk_rope_head_dim', 64)
        
        if config.get('qk_nope_head_dim', 0) == 0:
            self.qk_nope_head_dim = (config['emb_dim'] // self.num_heads) - self.qk_rope_head_dim
        else:
            self.qk_nope_head_dim = config['qk_nope_head_dim']
            
        self.v_head_dim = config.get('v_head_dim', config['emb_dim'] // self.num_heads)
        self.qk_head_dim = self.qk_rope_head_dim + self.qk_nope_head_dim
        self.softcap = config.get('softcap', None)
        
        attention_bias = config.get('qkv_bias', False)
        query_pre_attn_scalar = config.get('query_pre_attn_scalar', 1.0)
        
        self.is_causal = True
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(config['emb_dim'], self.num_heads * self.qk_head_dim, bias=attention_bias)
        else:
            self.q_a_proj = nn.Linear(config['emb_dim'], self.q_lora_rank, bias=False)
            self.q_b_proj = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=attention_bias)

        self.kv_a_proj_with_mqa = nn.Linear(
            config['emb_dim'],
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=attention_bias,
        )
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=attention_bias,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config['emb_dim'],
            bias=False,
        )
        
        
        self.rope = RotaryPositionalEmbeddings(
            dim=self.qk_rope_head_dim, 
            max_seq_len=config['max_seq_length']
        )
        
        # Register causal mask
        self.register_buffer(
            "mask", torch.triu(torch.ones(config['max_seq_length'], config['max_seq_length']), diagonal=1)
        )

        self.scaling = query_pre_attn_scalar ** (-0.5)

    def forward(self, x):
        batch_size, seq_length = x.shape[:-1]
        hidden_states = x
        
        if self.q_lora_rank is None:
            q_states = self.q_proj(hidden_states)
        else:
            q_states = self.q_b_proj(self.q_a_proj(hidden_states))
        
        # [batch, seq, num_heads, head_dim]
        q_states = q_states.view(batch_size, seq_length, self.num_heads, self.qk_head_dim)
        q_states = q_states.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        k_pass_compressed, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_pass_expanded = self.kv_b_proj(k_pass_compressed)
        k_pass_expanded = k_pass_expanded.view(batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_pass_expanded = k_pass_expanded.transpose(1, 2)  # [batch, num_heads, seq, head_dim]
        k_pass, value_states = torch.split(k_pass_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_rot = k_rot.unsqueeze(1).expand(batch_size, self.num_heads, seq_length, self.qk_rope_head_dim)
        q_rot_reshaped = q_rot.transpose(1, 2)  # [batch, seq, num_heads, rope_dim]
        k_rot_reshaped = k_rot.transpose(1, 2)  # [batch, seq, num_heads, rope_dim]
        q_rot_rope = self.rope(q_rot_reshaped).transpose(1, 2)  # [batch, num_heads, seq, rope_dim]
        k_rot_rope = self.rope(k_rot_reshaped).transpose(1, 2)  # [batch, num_heads, seq, rope_dim
        query_states = torch.cat((q_pass, q_rot_rope), dim=-1)
        key_states = torch.cat((k_pass, k_rot_rope), dim=-1)
        mask_bool = self.mask.to(torch.bool)[:seq_length, :seq_length]
        attn_output, attn_weights = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            softcap=self.softcap,
        )
        
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output
    
    @torch.no_grad()
    def normalize_matrices(self):
        """Normalize weight matrices for compatibility with training loop"""
        self._normalize_matrix(self.o_proj.weight, dim=0)
        if self.q_lora_rank is None:
            self._normalize_matrix(self.q_proj.weight)
        else:
            self._normalize_matrix(self.q_a_proj.weight)
            self._normalize_matrix(self.q_b_proj.weight)
        self._normalize_matrix(self.kv_a_proj_with_mqa.weight)
        self._normalize_matrix(self.kv_b_proj.weight)

    @staticmethod
    def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).type_as(x)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(self.l2_normalize(w, dim=dim))