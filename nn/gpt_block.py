import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional
from nn.norms import LayerNorm
from nn.transformer_block import TransformerBlock, muPTransformerBlock


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["max_seq_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class nGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["max_seq_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.sz_init_value = 1.0
        self.sz_init_scaling = 1.0 / math.sqrt(cfg["emb_dim"])
        self.sz = nn.Parameter(torch.empty(cfg["vocab_size"]))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reset the scaling parameter.
        """
        nn.init.ones_(self.sz)
        with torch.no_grad():
            self.sz.mul_(self.sz_init_scaling)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        tok_embeds = self.l2_normalize(tok_embeds, dim=-1)
        pos_embeds = self.l2_normalize(pos_embeds, dim=-1)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        sz = self.sz * (self.sz_init_value / self.sz_init_scaling)
        logits = sz * self.out_head(x)
        return logits
    
    @torch.no_grad()
    def normalize_matrices(self):
        self._normalize_matrix(self.out_head.weight)
        self._normalize_matrix(self.tok_emb.weight, dim=-1)
        self._normalize_matrix(self.pos_emb.weight, dim=-1)

    @staticmethod
    def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).type_as(x)
    
    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(self.l2_normalize(w, dim=dim))


class muPGPTModel(nn.Module):
    """
    1. Embeddings: O(1) initialization, O(1) learning rate
    2. Output head: O(1/√d_model) initialization, O(1) learning rate  
    3. Attention/FFN: follow muP scaling within blocks
    4. No additional scaling for residual connections
    """
    
    def __init__(
        self, 
        cfg: Dict[str, Any], 
        mup_base_d_model: Optional[int] = None,
        use_normalized_blocks: bool = False
    ):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.vocab_size = cfg["vocab_size"]
        self.n_layers = cfg["n_layers"]
        self.use_normalized_blocks = use_normalized_blocks
    
        self.mup_base_d_model = mup_base_d_model or cfg["emb_dim"]
        self.mup_scale = math.sqrt(self.mup_base_d_model / cfg["emb_dim"])
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["max_seq_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[muPTransformerBlock(
                cfg, 
                mup_base_d_model=mup_base_d_model,
                use_normalized_ffn=use_normalized_blocks
            ) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
        self.base_shapes = {
            "tok_emb.weight": self.tok_emb.weight.shape,
            "pos_emb.weight": self.pos_emb.weight.shape,
            "out_head.weight": self.out_head.weight.shape,
        }
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=1.0)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=1.0)
        std_out = 1.0 / math.sqrt(self.emb_dim)
        nn.init.normal_(self.out_head.weight, mean=0.0, std=std_out)
    
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
    def get_mup_lr_scales(self, base_lr: float) -> Dict[str, float]:
        lr_scales = {}
        lr_scales["tok_emb.weight"] = base_lr
        lr_scales["pos_emb.weight"] = base_lr
        lr_scales["out_head.weight"] = base_lr
        for name, _ in self.final_norm.named_parameters():
            lr_scales[f"final_norm.{name}"] = base_lr
        for block_idx, block in enumerate(self.trf_blocks):
            block_lr_scales = block.get_mup_lr_scales(base_lr)
            for param_name, lr_scale in block_lr_scales.items():
                lr_scales[f"trf_blocks.{block_idx}.{param_name}"] = lr_scale
        return lr_scales
    
    def create_mup_optimizer(self, base_lr: float, weight_decay: float = 0.0, **kwargs):
        lr_scales = self.get_mup_lr_scales(base_lr)
        lr_groups = {}
        for name, param in self.named_parameters():
            if name in lr_scales:
                lr = lr_scales[name]
                if lr not in lr_groups:
                    lr_groups[lr] = []
                lr_groups[lr].append(param)
            else:
                if base_lr not in lr_groups:
                    lr_groups[base_lr] = []
                lr_groups[base_lr].append(param)

        param_groups = []
        for lr, params in lr_groups.items():
            param_groups.append({
                'params': params,
                'lr': lr,
                'weight_decay': weight_decay
            })
        from torch.optim import AdamW
        optimizer = AdamW(param_groups, **kwargs)
        return optimizer
    
    @torch.no_grad()
    def normalize_matrices(self):
        if self.use_normalized_blocks:
            for block in self.trf_blocks:
                block.normalize_matrices()


class muPGPTConfig:    
    @staticmethod
    def create_base_config(
        vocab_size: int = 50257,
        max_seq_length: int = 1024,
        emb_dim: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        drop_rate: float = 0.1,
        qkv_bias: bool = False,
        rope: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        return {
            "vocab_size": vocab_size,
            "max_seq_length": max_seq_length,
            "emb_dim": emb_dim,
            "n_heads": n_heads,
            "n_layers": n_layers,
            "drop_rate": drop_rate,
            "qkv_bias": qkv_bias,
            "rope": rope,
            **kwargs
        }
    
    @staticmethod
    def scale_config(
        base_config: Dict[str, Any], 
        width_multiplier: float
    ) -> Dict[str, Any]:
        scaled_config = base_config.copy()
        base_emb_dim = base_config["emb_dim"]
        scaled_config["emb_dim"] = int(base_emb_dim * width_multiplier)
        if scaled_config["emb_dim"] % scaled_config["n_heads"] != 0:
            scaled_config["n_heads"] = max(1, scaled_config["emb_dim"] // 64)
            if scaled_config["emb_dim"] % scaled_config["n_heads"] != 0:
                scaled_config["emb_dim"] = (scaled_config["emb_dim"] // scaled_config["n_heads"]) * scaled_config["n_heads"]
        
        return scaled_config