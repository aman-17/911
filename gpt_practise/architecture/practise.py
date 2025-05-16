import torch

GPT_CONFIG_124M = {
"vocab_size": 50257, # Vocabulary size
"context_length": 1024, # Context length
"emb_dim": 768, # Embedding dimension
"n_heads": 12, # Number of attention heads
"n_layers": 12, # Number of layers
"drop_rate": 0.1, # Dropout rate
"qkv_bias": False # Query-Key-Value bias
}


class MHAttention(torch.nn.Module):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.query = torch.nn.Linear(cfg["context_length"], cfg["emb_dim"])
        self.key = torch.nn.Linear(cfg["context_length"], cfg["emb_dim"])
        self.value = torch.nn.Linear(cfg["context_length"], cfg["emb_dim"])
    
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        q_update = torch.view(q,(batch_size, seq_len, embed_dim))
        k_update = torch.view(k,(batch_size, seq_len, embed_dim))
        v_update = torch.view(v,(batch_size, seq_len, embed_dim))
        attn_scores = q_update @ k_update.transpose(1, 2)
        attn_weight = attn_scores / (embed_dim ** 0.5)
        attn_weight = attn_weight @ v_update(1,2)
        
        return x