# Changelog

All notable changes to this project are documented here.

---

## [Unreleased] — 2026-05-07

### Added
- `pip install 911` support via `src/` layout — moved all three packages under `src/`
- `911-train` CLI entry point registered via `[project.scripts]`
- `[serve]` optional dependency group for FastAPI feature explorer

### Changed
- All `print()` calls replaced with `logging.getLogger(__name__)` across every module
- All bare internal imports in `pre_training/` prefixed to fully-qualified `pre_training.xxx` paths
- `pyproject.toml` rewritten: `[tool.setuptools.packages.find] where = ["src"]`, proper `package-data`, `pyyaml` added to core dependencies

### Fixed
- `pre_training/data/web_crawling/` and `interpretability/app/` were missing `__init__.py`
- Stale commented-out code block in `dataset_utils.py` removed

---

## [0.5.0] — 2026-04-30

### Added
- **Qwen3-4B model loader** (`interpretability/models/qwen3_4b.py`) for running interpretability on Qwen3
- **Feature steering web app** (PR #21) — FastAPI server with streaming SSE generation, `FeatureSteerer` context manager, per-feature activation analysis endpoint, and static UI at `interpretability/app/`
- **SAE training pipeline** (PR #19) — `interpretability/train.py` with TopK Sparse Autoencoder (k=32, 32K dictionary), activation collection from lmsys-chat-1M via OLMo-2 1B, and feature analysis pre-computation (`interpretability/analyze.py`)

---

## [0.4.0] — 2026-02-02

### Added
- **GatedMultiHeadAttention** (`nn/attention/gated_attention.py`)
- **MinMax Attention** (`nn/attention/minmax_attention.py`) — standalone attention variant using T5-style layer norm
- **Tensor Parallelism** (`nn/distributed/parallel/tensor_parallel.py`) — `ColwiseParallel` / `RowwiseParallel` via `parallelize_module`, wired into MHA and GQA via `apply_tp()`
- TP integration into transformer blocks
- **FSDP** with configurable sharding strategies (`FULL_SHARD`, `SHARD_GRAD_OP`, `HYBRID_SHARD`, `NO_SHARD`), mixed precision, activation checkpointing, CPU offload, and backward prefetch
- **KV-cache** for `MultiHeadAttention` — `use_cache` config flag
- `use_cache` support in GPT transformer block
- Throughput tracking (tokens/sec) logged to W&B
- `python-publish.yml` GitHub Actions workflow for PyPI publishing
- `pylint.yml` CI workflow for style and lint checks
- `get_rank()` utility fix in `nn/distributed/utils.py`

### Changed
- Training loop updated for FSDP: proper `dist.barrier()`, all-gather for loss averaging across ranks

---

## [0.3.0] — 2025-06-20

### Added
- **Qwen3 architecture** (`nn/transfomer/model/qwen_model.py`, `block/qwen3_transformer.py`) with GQA, `qk_norm`, and Qwen3-specific RMSNorm
- **nanoGPT architecture** (`nn/transfomer/model/gpt_model.py` `nanoGPTModel`, `block/nanoGPT_transformer.py`)
- **z-loss** (softmax auxiliary loss) — `CrossEntropyLoss` with configurable `z_loss_multiplier`, tracked separately in training loop and W&B
- **MLA + NSA merged** (PR #14, #15) — Multi-Head Latent Attention and Native Sparse Attention landed in main
- Repo cleanup pass — removed stale files, normalised import structure

### Fixed
- GQA key/value head group logic
- Multiple Qwen3 forward pass issues

---

## [0.2.0] — 2025-06-05

### Added
- **LLaMA architecture** (`nn/transfomer/model/llama_model.py`, `block/llama_transformer.py`) — RMSNorm, SwiGLU FFN, RoPE, GQA
- **DDP (DistributedDataParallel)** support alongside FSDP
- **Multi-Head Latent Attention (MLA)** (`nn/attention/multihead_latent_attention.py`) — DeepSeek-style low-rank KV compression with `naive` and `absorb` implementations, dynamic KV cache
- **Native Sparse Attention (NSA)** (PR #12, `nn/attention/native_sparse_attention.py`) — compressed attention, selected-block attention, and sliding window attention combined
- `dtype` and `use_flash_attn` config flags
- `.npy` shard support in data loader
- Beaker cluster training scripts
- `freqs_cis` precomputation for LLaMA RoPE

### Fixed
- Loss calculation bug
- `IterableDataset` shuffle buffer — was yielding only half the buffer; fixed to yield all then clear

---

## [0.1.0] — 2025-05-23

### Added
- **Initial GPT-2 implementation** — `GPTModel`, `GPTTransformerBlock`, multi-head self-attention with causal mask
- **nGPT architecture** (PR #10) — normalised GPT with hyperspherical representations
- **Grouped Query Attention (GQA)** (`nn/attention/groupquery_attention.py`) with Flash Attention support
- **Rotary Positional Embeddings (RoPE)** — standard, complex, and YaRN variants
- **Iterable dataset** (`data/data_loader.py`) with shuffle buffer and distributed sharding (PR #3, #5, #8)
  - Support for `.npy` tokenized shards and raw `.txt` files
  - Multiple `.txt` file support
  - Gradient clipping
- **Cosine LR scheduler** via `torch.optim.lr_scheduler.CosineAnnealingLR`
- **W&B integration** — loss, learning rate, tokens seen, and throughput logging
- `config.yaml` — YAML-driven configuration with named model variants
- HuggingFace dataset tokenization and `.npy` shard export (`data/web_crawling/datasets_from_hf.py`)
- Wikipedia crawler (`data/web_crawling/wiki_crawler.py`)
- `.gitignore`, project structure, initial `pyproject.toml`
