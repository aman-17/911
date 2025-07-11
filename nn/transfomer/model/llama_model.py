import torch
import torch.nn as nn

from nn.attention.utils import precompute_freqs_cis
from nn.norms import RMSNorm
from nn.transfomer.block.llama_transformer import LlamaTransformerBlock
from nn.utils import autocast_precision, ensure_multiple_of


class LlamaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["max_seq_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        hidden_size = int(8 * cfg["emb_dim"] / 3)
        if cfg.get("hidden_size_multiplier") is not None:
            hidden_size = int(cfg["hidden_size_multiplier"] * hidden_size)
        hidden_size = ensure_multiple_of(hidden_size, cfg.get("hidden_size_multiple_of", 256))
        self.trf_blocks = nn.Sequential(*[LlamaTransformerBlock({**cfg, "block_idx": i, "hidden_size": hidden_size}) for i in range(cfg["n_layers"])])
        self.final_norm = RMSNorm(cfg["emb_dim"], dtype=autocast_precision(cfg["dtype"]))
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        if cfg.get("attention", "mha") == "mla":
            self.register_buffer(
                "freqs_cis",
                precompute_freqs_cis(
                    cfg["emb_dim"],
                    cfg["max_seq_length"],
                    cfg["seq_length"],
                    cfg["beta_fast"],
                    cfg["beta_slow"],
                    cfg["rope_factor"],
                    cfg["rope_theta"],
                ),
                persistent=False,
            )

        self.apply(self._init_weights)
        if cfg.get("tie_embeddings", True):
            self.out_head.weight = self.tok_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, in_idx, start_pos: int = 0):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        if self.cfg.get("attention", "mha") == "mla":
            freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
            mask = None
            if seq_len > 1:
                mask = torch.full((seq_len, seq_len), float("-inf"), device=in_idx.device).triu_(1)
            for block in self.trf_blocks:
                x = block(x, start_pos, freqs_cis, mask)
        else:
            for block in self.trf_blocks:
                x = block(x)

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
