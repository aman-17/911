import functools
import os
import time
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import BackwardPrefetch, CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

import wandb
from config_utils import load_config
from data.dataset_utils import create_train_loader
from nn.loss_function import calc_loss_batch, calc_total_loss
from nn.transfomer.model.gpt_model import GPTModel, nanoGPTModel, nGPTModel
from nn.transfomer.model.llama_model import LlamaModel
from nn.transfomer.model.qwen_model import Qwen3Model
from nn.utils import generate_text_simple


def setup(rank: Optional[int] = None, world_size: Optional[int] = None) -> Tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        print(f"Running with torchrun: rank={rank}, world_size={world_size}, local_rank={local_rank}")

    else:
        if rank is None or world_size is None:
            raise ValueError("rank and world_size must be provided when not using torchrun")
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29501"
        local_rank = rank
        print(f"Running with mp.spawn: rank={rank}, world_size={world_size}")

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    elif torch.backends.mps.is_available():
        dist.init_process_group("gloo", rank=rank, world_size=world_size)

    return rank, world_size, local_rank


def cleanup():
    dist.destroy_process_group()


def get_fsdp_config(train_config: Dict) -> Dict:
    fsdp_config = train_config.get("distributed", {}).get("fsdp", {})
    sharding_strategy_map = {
        "FULL_SHARD": ShardingStrategy.FULL_SHARD,
        "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
        "NO_SHARD": ShardingStrategy.NO_SHARD,
        "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
    }

    backward_prefetch_map = {
        "BACKWARD_PRE": BackwardPrefetch.BACKWARD_PRE,
        "BACKWARD_POST": BackwardPrefetch.BACKWARD_POST,
    }

    return {
        "sharding_strategy": sharding_strategy_map.get(fsdp_config.get("sharding_strategy", "FULL_SHARD"), ShardingStrategy.FULL_SHARD),
        "mixed_precision": fsdp_config.get("mixed_precision", True),
        "activation_checkpointing": fsdp_config.get("activation_checkpointing", True),
        "cpu_offload": fsdp_config.get("cpu_offload", False),
        "backward_prefetch": backward_prefetch_map.get(fsdp_config.get("backward_prefetch", "BACKWARD_PRE"), BackwardPrefetch.BACKWARD_PRE),
        "forward_prefetch": fsdp_config.get("forward_prefetch", False),
        "use_orig_params": fsdp_config.get("use_orig_params", True),
    }


def setup_fsdp_model(model: torch.nn.Module, train_config: Dict) -> torch.nn.Module:
    fsdp_config = get_fsdp_config(train_config)
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=100_000_000)
    mixed_precision_policy = None
    if fsdp_config["mixed_precision"]:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float32,
        )
    cpu_offload_policy = None
    if fsdp_config["cpu_offload"]:
        cpu_offload_policy = CPUOffload(offload_params=True)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision_policy,
        cpu_offload=cpu_offload_policy,
        backward_prefetch=fsdp_config["backward_prefetch"],
        forward_prefetch=fsdp_config["forward_prefetch"],
        sharding_strategy=fsdp_config["sharding_strategy"],
        use_orig_params=fsdp_config["use_orig_params"],
    )

    if fsdp_config["activation_checkpointing"]:

        def check_fn(submodule):
            return isinstance(submodule, (torch.nn.TransformerEncoderLayer, torch.nn.TransformerDecoderLayer))

        apply_activation_checkpointing(model, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)

    return model


def setup_distributed_model(model: torch.nn.Module, world_size: int, train_config: Dict) -> torch.nn.Module:
    if world_size == 1:
        return model
    return setup_fsdp_model(model, train_config)


def generate_and_print_sample(model, tokenizer, start_context, device, rank):
    if rank == 0:
        model.eval()
        base_model = model.module if hasattr(model, "module") else model
        context_size = base_model.max_seq_len if hasattr(base_model, "max_seq_len") else base_model.cfg.get("max_seq_length", 4096)
        encoded = tokenizer.encode(start_context)
        encoded = torch.tensor(encoded, dtype=torch.long, device=device)  # .unsqueeze(0)

        with torch.no_grad():
            token_ids = generate_text_simple(model=model, idx=encoded, max_new_tokens=50, context_size=context_size)
            decoded_text = tokenizer.decode(token_ids)  # .squeeze(0).tolist())
            cleaned_text = decoded_text.replace("\n", " ")
            print(f"[Rank {rank}] Generated: {cleaned_text}")
        model.train()


def create_model(train_config: Dict, device: torch.device) -> torch.nn.Module:
    if train_config["model_arch"] == "gpt":
        model = GPTModel(train_config)
    elif train_config["model_arch"] == "nanogpt":
        model = nanoGPTModel(train_config)
    elif train_config["model_arch"] == "ngpt":
        model = nGPTModel(train_config)
    elif train_config["model_arch"] == "qwen3":
        model = Qwen3Model(train_config)
    else:
        model = LlamaModel(train_config)
    return model.to(device)


def train_911(
    rank,
    world_size,
    train_config,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
):
    rank, world_size, local_rank = setup(rank, world_size)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = create_model(train_config, device)
    model = setup_distributed_model(model, world_size, train_config)
    if rank == 0:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="911-training", config=train_config)

    train_loader, tokenizer = create_train_loader(train_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["initial_lr"],
        weight_decay=train_config["weight_decay"],
    )

    learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs * len(train_loader), eta_min=0.0, last_epoch=-1)

    train_ce_losses, train_z_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = 0
    start_time = time.time()
    last_eval_time = start_time
    last_eval_tokens = 0

    for epoch in range(num_epochs):
        model.train()
        ce_epoch_loss = 0.0
        z_epoch_loss = 0.0

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            ce_loss, z_loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss = (ce_loss + z_loss) / world_size
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            learning_rate_scheduler.step()

            tokens_seen += input_batch.numel()
            ce_epoch_loss += ce_loss.item()
            z_epoch_loss += z_loss.item()
            if rank == 0 and global_step % 10 == 0:
                print(f"Step {global_step}: CE Loss = {ce_loss.item():.4f}, Z Loss = {z_loss.item():.4f}, LR = {learning_rate_scheduler.get_last_lr()[0]:.6f}")

            if global_step % eval_freq == 0:
                if torch.cuda.is_available():
                    dist.barrier(device_ids=[local_rank])
                else:
                    dist.barrier()
                total_ce_loss, total_z_loss = calc_total_loss(train_loader, model, device, eval_iter)

                gathered_ce_losses = [torch.zeros(1).to(device) for _ in range(world_size)]
                dist.all_gather(gathered_ce_losses, torch.tensor([total_ce_loss]).to(device))
                avg_train_ce_loss = sum(loss.item() for loss in gathered_ce_losses) / world_size

                gathered_z_losses = [torch.zeros(1).to(device) for _ in range(world_size)]
                dist.all_gather(gathered_z_losses, torch.tensor([total_z_loss]).to(device))
                avg_train_z_loss = sum(loss.item() for loss in gathered_z_losses) / world_size

                if rank == 0:
                    current_time = time.time()
                    time_elapsed = current_time - last_eval_time
                    tokens_processed = tokens_seen - last_eval_tokens

                    if time_elapsed > 0:
                        tps = tokens_processed / time_elapsed
                    else:
                        tps = 0.0
                    last_eval_time = current_time
                    last_eval_tokens = tokens_seen

                    train_ce_losses.append(avg_train_ce_loss)
                    train_z_losses.append(avg_train_z_loss)
                    track_tokens_seen.append(tokens_seen)
                    wandb.log(
                        {
                            "global train CE loss": avg_train_ce_loss,
                            "global train Z loss": avg_train_z_loss,
                            "lr": learning_rate_scheduler.get_last_lr()[0],
                            "tokens_seen": tokens_seen,
                            "epoch": epoch,
                            "global_step": global_step,
                            "throughput_tps": tps,
                        }
                    )
                generate_and_print_sample(model, tokenizer, start_context, device, rank)

            global_step += 1

    if rank == 0:
        # torch.save(model.module.state_dict(), "model_checkpoint.pt") \
        wandb.finish()

    cleanup()

    return train_ce_losses, train_z_losses, track_tokens_seen


def run_training(rank, world_size, train_config):
    train_911(
        rank=rank,
        world_size=world_size,
        train_config=train_config,
        num_epochs=train_config["num_epochs"],
        eval_freq=train_config["DEFAULT_EVAL_FREQ"],
        eval_iter=train_config["DEFAULT_EVAL_ITER"],
        start_context=train_config["DEFAULT_START_CONTEXT"],
    )


def main():
    train_config = load_config()
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        run_training(rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]), train_config=train_config)
    else:
        if torch.cuda.is_available():
            world_size = torch.cuda.device_count()
            mp.spawn(run_training, args=(world_size, train_config), nprocs=world_size, join=True)
        else:
            run_training(0, 1, train_config)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
