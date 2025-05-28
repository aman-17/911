import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from nn.ffn import FeedForward, muPFeedForward, muPNormalizedFeedForward
from nn.norms import LayerNorm
from nn.multihead_attention import MultiHeadAttention, muPMultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            max_seq_len=cfg["max_seq_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            use_rope=cfg["rope"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x



class muPTransformerBlock(nn.Module):
    """
    1. Attention and FFN follow muP scaling rules
    2. Residual connections have no additional scaling
    3. Layer norms are applied as standard
    4. Learning rates are scaled per parameter type
    """
    
    def __init__(
        self, 
        cfg: Dict[str, Any], 
        mup_base_d_model: Optional[int] = None,
        use_normalized_ffn: bool = False
    ):
        super().__init__()
        self.use_normalized_ffn = use_normalized_ffn
        
        self.att = muPMultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            max_seq_len=cfg["max_seq_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"],
            use_rope=cfg["rope"],
            mup_base_d_model=mup_base_d_model,
        )
        if use_normalized_ffn:
            self.ff = muPNormalizedFeedForward(cfg, mup_base_d_model)
        else:
            self.ff = muPFeedForward(cfg, mup_base_d_model)
        
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_resid = nn.Dropout(cfg["drop_rate"])
    
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut  # Add the original input back
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x
    
    def get_mup_lr_scales(self, base_lr: float) -> Dict[str, float]:
        lr_scales = {}
        for name, _ in self.att.named_parameters():
            param_name = f"att.{name}"
            lr_scales[param_name] = self.att.get_mup_lr_scale(name, base_lr)
        for name, _ in self.ff.named_parameters():
            param_name = f"ff.{name}"
            lr_scales[param_name] = self.ff.get_mup_lr_scale(name, base_lr)
        for name, _ in self.norm1.named_parameters():
            lr_scales[f"norm1.{name}"] = base_lr
        for name, _ in self.norm2.named_parameters():
            lr_scales[f"norm2.{name}"] = base_lr
            
        return lr_scales
    
    @torch.no_grad()
    def normalize_matrices(self):
        if hasattr(self.att, 'normalize_matrices'):
            self.att.normalize_matrices()
        if hasattr(self.ff, 'normalize_matrices'):
            self.ff.normalize_matrices()
