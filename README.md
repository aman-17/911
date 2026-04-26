# 911

A research framework for the full LLM lifecycle - pre-training, post-training, and mechanistic interpretability - built from scratch in PyTorch.

---

## What's inside

| Component | What it does |
|---|---|
| `pre_training/` | Train language models from scratch with FSDP, tensor parallelism, and multiple architecture variants |
| `post_training/` | Inference utilities, KV-cache generation, and nucleus sampling rollouts |
| `interpretability/` | Train Sparse Autoencoders on model activations and steer features at generation time |

---

## Installation

```bash
git clone https://github.com/aman-17/911
cd 911
pip install -e .
```

Requires Python 3.11+ and PyTorch 2.7+ with CUDA 12.8:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
```

---

## Usage

### Pre-training

```bash
python pre_training/train.py
```

Supports FSDP sharding strategies (`FULL_SHARD`, `SHARD_GRAD_OP`, `HYBRID_SHARD`), activation checkpointing, mixed precision, and W&B logging. Configure via YAML:

```yaml
model:
  emb_dim: 2048
  n_heads: 16
  n_layers: 24
  attention: grouped_query

training:
  batch_size: 512
  lr: 3e-4
  fsdp_strategy: FULL_SHARD
```

### Post-training inference

```python
from post_training.inference.inference_utils import generate_top_p

output = generate_top_p(model, tokenizer, prompt="Hello!", max_new_tokens=200, top_p=0.9, temperature=0.8)
```

KV-cache rollouts for RLHF-style training:

```python
from post_training.inference.rollout import sample_response

tokens, text, log_probs = sample_response(model, tokenizer, prompt_ids, max_new_tokens=512)
```

### Interpretability — SAE training

**Step 1: Collect residual stream activations**
```bash
python -m interpretability.data.lymsys_chat1b
```
Runs OLMo-2 1B inference and saves activations from layer 8 to disk in 200K-token chunks.

**Step 2: Train the SAE**
```bash
python -m interpretability.train
```
Trains a TopK Sparse Autoencoder (k=32, 32K dictionary) on the collected activations for 50M tokens.

### Interpretability — Feature steering

```python
from interpretability.inference import run_steered_generation

output = run_steered_generation(feature_idx=4821, scale=3.0, prompt="Tell me about your day")
print(output)
```

Or use `FeatureSteerer` directly for full control:

```python
from interpretability.inference import FeatureSteerer

with FeatureSteerer(model, sae, layer_idx=8).set_feature(4821, scale=3.0):
    output_ids = model.generate(**inputs, max_new_tokens=200)
```