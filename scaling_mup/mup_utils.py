from mup import get_shapes, make_base_shapes
import torch
from nn.gpt_block import muPGPTModel, muPGPTConfig


def save_base_shapes(output_path: str, d_model: int = 768):
    base_config = muPGPTConfig.create_base_config(
        vocab_size=50257,
        max_seq_length=1024,
        emb_dim=d_model,
        n_heads=max(1, d_model // 64),
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
        rope=True,
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = muPGPTModel(
        base_config, 
        mup_base_d_model=d_model,
        use_normalized_blocks=False
    )
    base_model = base_model.to(device)
    base_shapes = get_shapes(base_model)
    scaled_config = muPGPTConfig.create_base_config(
        vocab_size=50257,
        max_seq_length=1024,
        emb_dim=d_model * 2,
        n_heads=max(1, (d_model * 2) // 64),
        n_layers=12,
        drop_rate=0.1,
        qkv_bias=False,
        rope=True,
    )
    
    scaled_model = muPGPTModel(
        scaled_config,
        mup_base_d_model=d_model,
        use_normalized_blocks=False
    )
    scaled_model = scaled_model.to(device)
    delta_shapes = get_shapes(scaled_model)
    
    make_base_shapes(base_shapes, delta_shapes, savefile=output_path)
