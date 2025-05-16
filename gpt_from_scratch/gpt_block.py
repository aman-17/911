import torch
import torch.nn as nn
from ffn import LayerNorm
from transformer_block import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], 
            cfg["vocab_size"], 
            bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

def generate_text_simple(model: nn.Module,
                        idx: torch.Tensor,
                        max_new_tokens: int,
                        context_size: int
                    ) -> torch.Tensor:
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
    idx = torch.tensor(idx, dtype=torch.long).unsqueeze(0)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        
        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]  # Focus on last token
            probas = torch.softmax(logits, dim=-1)
            idx_next = torch.argmax(probas, dim=-1, keepdim=True)
            
        # Append new token
        idx = torch.cat((idx, idx_next), dim=1)
    
    return idx[0].tolist()


# GPT_CONFIG_124M = {
# "vocab_size": 50257, # Vocabulary size
# "context_length": 1024, # Context length
# "emb_dim": 768, # Embedding dimension
# "n_heads": 12, # Number of attention heads
# "n_layers": 12, # Number of layers
# "drop_rate": 0.1, # Dropout rate
# "qkv_bias": False # Query-Key-Value bias
# }

# import tiktoken

# tokenizer = tiktoken.get_encoding("gpt2")
# start_context = "Hello, I am"
# encoded = tokenizer.encode(start_context)
# print("encoded:", encoded)
# encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# print("encoded_tensor.shape:", encoded_tensor.shape)
# model = GPTModel(GPT_CONFIG_124M)
# model.eval()
# with torch.no_grad():
#     out = generate_text_simple(
#         model=model,
#         idx=encoded_tensor,
#         max_new_tokens=6,
#         context_size=GPT_CONFIG_124M["context_length"]
#     )   
# print(f"Output: {out}")
# print(f"Output length: {len(out[0])}")

# decoded_text = tokenizer.decode(out.squeeze(0).tolist())
# print(decoded_text)