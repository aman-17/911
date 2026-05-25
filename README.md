# 911

A mini training infrastructure for LLMs from scratch: pre-training, post-training inference, and mechanistic interpretability with Sparse Autoencoders, built in pure PyTorch.

```
pip install 911
```

---

## Components

| Package | Description |
|---|---|
| `pre_training` | Train GPT, LLaMA, Qwen3, nGPT from scratch with FSDP multi-GPU |
| `post_training` | KV-cache generation, nucleus sampling, rollout for RLHF pipelines |
| `vlms` | SFT pipeline for Qwen3-VL / Qwen3-VL-Moe (image + video + text) with LoRA, packing, and flash-attention |
| `interpretability` | Collect activations, train TopK SAEs, steer features at inference |

---

## Installation

Requires Python ≥ 3.11 and PyTorch ≥ 2.6.

```bash
pip install 911
```

For GPU training with CUDA 12.8:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install 911
```

For the feature-steering web app:

```bash
pip install "911[serve]"
```

For VLM fine-tuning (image / video):

```bash
pip install "911[vlms]"
```

---

## Pre-training

### Quickstart

```bash
# single GPU / CPU
911-train

# multi-GPU with torchrun
torchrun --nprocs-per-node 8 -m pre_training.train
```

Configuration lives in `config.yaml`. Set the active variant and point `train_data` at a directory of `.npy` shards or a `.txt` file:

```yaml
model:
  active: qwen3_0_6B   # see variants below

train_data: /data/fineweb
batch_size: 8
num_epochs: 2
```

### Model variants

| Variant | Arch | Params (approx) |
|---|---|---|
| `gpt2_small` / `medium` / `large` / `xl` | GPT-2 | 117M – 1.5B |
| `nanogpt_small` / `medium` | nanoGPT | 117M – 350M |
| `ngpt_small` / `medium` | nGPT | 117M – 350M |
| `llamalike1B` | LLaMA-3 | 1B |
| `llama8B` / `70B` / `405B` | LLaMA-3 | 8B – 405B |
| `qwen3_0_6B` | Qwen3 | 0.6B |

### Attention mechanisms

Set `attention` in `config.yaml`:

| Value | Module |
|---|---|
| `mha` | Multi-Head Attention (default) |
| `gqa` | Grouped Query Attention |
| `mla` | Multi-Head Latent Attention (DeepSeek-style) |
| `nsa` | Native Sparse Attention |
| `minmax` | MinMax Attention |

### Distributed training (FSDP)

```yaml
distributed:
  fsdp:
    sharding_strategy: FULL_SHARD   # FULL_SHARD | SHARD_GRAD_OP | HYBRID_SHARD | NO_SHARD
    mixed_precision: true
    activation_checkpointing: true
    cpu_offload: false
    backward_prefetch: BACKWARD_PRE
```

### Data preparation

Download and tokenize a HuggingFace dataset into `.npy` shards:

```bash
python -m pre_training.data.web_crawling.datasets_from_hf \
  --dataset HuggingFaceFW/fineweb-edu \
  --dataset_config sample-10BT \
  --tokenizer gpt2 \
  --output_dir /data/fineweb \
  --shard_size 100000000
```

---

## Post-training

### Top-p generation

```python
from post_training.inference.inference_utils import generate_top_p
from post_training.data.data_tokenizer import load_model_and_tokenizer

model, tokenizer = load_model_and_tokenizer(device="cuda")
response = generate_top_p(model, tokenizer, prompt, device="cuda", max_new_tokens=512)
```

### KV-cache rollout (for RLHF)

Returns token ids, per-token log-probs, and the full sequence — everything a reward model or PPO trainer needs:

```python
from post_training.inference.rollout import sample_response

result = sample_response(
    model, tokenizer, prompt,
    device="cuda",
    max_new_tokens=512,
    temperature=0.9,
    top_p=0.9,
)
# result["text"], result["log_probs"], result["full_token_ids"]
```

---

## VLM fine-tuning (Qwen3-VL)

Supervised fine-tuning for **Qwen3-VL** and **Qwen3-VL-Moe** on multi-modal conversations (image + video + text). Adapted from the upstream Qwen-VL trainer and rewired against a local copy of the modeling code, so the model definition lives in the repo rather than `transformers.models.qwen3_vl`.

### Quickstart

```bash
torchrun --nproc-per-node 8 -m vlms.train.train_qwen \
  --model_name_or_path Qwen/Qwen3-VL-4B-Instruct \
  --dataset_use cambrian_737k \
  --output_dir ./out/qwen3vl-sft \
  --bf16 True \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 1 \
  --model_max_length 4096 \
  --tune_mm_llm True \
  --tune_mm_mlp True \
  --tune_mm_vision False
```

Or via the installed script:

```bash
911-train-vlm --model_name_or_path Qwen/Qwen3-VL-4B-Instruct ...
```

### What you can control

| Flag | Effect |
|---|---|
| `--tune_mm_llm` | unfreeze the language backbone |
| `--tune_mm_mlp` | unfreeze the vision-to-LLM merger MLP |
| `--tune_mm_vision` | unfreeze the vision encoder |
| `--lora_enable` | LoRA on attention proj layers (r/alpha/dropout configurable) |
| `--data_packing` | pack multiple samples into one sequence (uses flash-attn varlen) |
| `--data_flatten` | flatten samples without padding |
| `--max_pixels` / `--min_pixels` | image token budget (defaults: 16 → 576 vision tokens) |
| `--video_max_frames` / `--video_fps` | video sampling controls |

### Datasets

Datasets are registered in [`vlms/data/__init__.py`](src/vlms/data/__init__.py). Each entry points at a `.json` / `.jsonl` annotation file in ShareGPT format and a data root for resolving relative `image:` / `video:` paths. Add your dataset by appending to `data_dict` and pass its key via `--dataset_use my_dataset` (comma-separated for multiple, `name%50` for 50% sampling).

### Notes

- Requires `transformers` from git/main — pinned automatically by `pip install 911[vlms]`. Qwen3-VL modeling uses utilities (`vision_utils`, `output_capturing`, kernel-hub integrations) that aren't in any released `transformers` yet.
- `flash-attn` is a hard runtime dependency for the packed/flattened code path; install it separately on the GPU box.
- The Moe variant (`Qwen3-VL-Moe`) still imports from `transformers.models.qwen3_vl_moe` — only the dense variant has been pulled local.

---

## Interpretability

### Step 1 — Collect activations

Runs OLMo-2 1B over lmsys-chat-1M, capturing residual stream activations at layer 8. Saves 200K-token chunks to disk.

```bash
python -m interpretability.data.lymsys_chat1b
```

### Step 2 — Train a Sparse Autoencoder

TopK SAE (k=32, 32K-feature dictionary) trained over 50M tokens:

```bash
python -m interpretability.train
```

Or from Python:

```python
from interpretability.train import train, TrainConfig

train(TrainConfig(
    d_model=2048,
    dict_size=32768,
    k=32,
    target_tokens=50_000_000,
    checkpoint_path="sae_layer8.pt",
))
```

### Step 3 — Analyze features

Pre-computes top activating examples per feature. Produces `feature_analysis.json` consumed by the web app:

```bash
python -m interpretability.analyze
```

### Step 4 — Steer features at generation

```python
from interpretability.inference import run_steered_generation

output = run_steered_generation(
    feature_idx=4821,
    scale=3.0,
    prompt="Tell me about your day",
)
```

For fine-grained control, use `FeatureSteerer` as a context manager:

```python
from interpretability.inference import FeatureSteerer

with FeatureSteerer(model, sae, layer_idx=8).set_feature(4821, scale=3.0):
    output_ids = model.generate(**inputs, max_new_tokens=200)
```

### Web app

```bash
uvicorn interpretability.app.main:app --reload
```

Opens a UI at `http://localhost:8000` for browsing SAE features and interactive steering.