import glob
from dataclasses import dataclass, field

import torch
from tqdm import tqdm

from interpretability.nn.sae import SAEConfig, SparseAutoencoder


@dataclass
class TrainConfig:
    d_model: int = 2048
    dict_size: int = 32768
    k: int = 32
    batch_size: int = 4096
    lr: float = 3e-4
    target_tokens: int = 50_000_000
    checkpoint_path: str = "olmo2_1b_sae_layer8.pt"
    activation_glob: str = "activations_chunk_*.pt"
    device: str = "cuda"
    log_every: int = 100


def train(cfg: TrainConfig = field(default_factory=TrainConfig)) -> None:
    sae = SparseAutoencoder(SAEConfig(d_model=cfg.d_model, dict_size=cfg.dict_size, k=cfg.k)).to(cfg.device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg.lr)

    activation_files = sorted(glob.glob(cfg.activation_glob))
    if not activation_files:
        raise FileNotFoundError(f"No activation files matched: {cfg.activation_glob!r}")

    tokens_seen = 0
    step = 0
    pbar = tqdm(total=cfg.target_tokens, desc="SAE training", unit="tok", unit_scale=True)

    while tokens_seen < cfg.target_tokens:
        chunk_order = torch.randperm(len(activation_files)).tolist()
        for ci in chunk_order:
            chunk = torch.load(activation_files[ci], weights_only=True).to(cfg.device)
            perm = torch.randperm(len(chunk), device=cfg.device)

            for i in range(0, len(chunk), cfg.batch_size):
                batch = chunk[perm[i : i + cfg.batch_size]]

                out = sae(batch)
                loss = sae.loss(batch, out.recon)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                sae.normalize_decoder()

                n = len(batch)
                tokens_seen += n
                step += 1
                pbar.update(n)

                if step % cfg.log_every == 0:
                    pbar.set_postfix(mse=f"{loss.item():.4f}")

                if tokens_seen >= cfg.target_tokens:
                    break

            del chunk
            if tokens_seen >= cfg.target_tokens:
                break

    pbar.close()
    torch.save(sae.state_dict(), cfg.checkpoint_path)
    print(f"Saved checkpoint → {cfg.checkpoint_path}")


if __name__ == "__main__":
    train(TrainConfig())
