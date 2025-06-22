import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    CPUOffload,
    BackwardPrefetch,
    ShardingStrategy,
    MixedPrecision,
    FullStateDictConfig,
    StateDictType,
    FullOptimStateDictConfig,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
)
import functools

import wandb

from config_utils import load_config
from data.dataset_utils import create_train_loader
from nn.loss_function import calc_loss_batch, calc_total_loss
from nn.transfomer.model.gpt_model import GPTModel, nanoGPTModel, nGPTModel
from nn.transfomer.model.llama_model import LlamaModel
from nn.utils import generate_text_simple
from optim.scheduler import CosWithWarmup


DEFAULT_EVAL_FREQ = 10
DEFAULT_EVAL_ITER = 50
GRADIENT_CLIP_VALUE = 1.0
DEFAULT_START_CONTEXT = "The Bowmanstown Borough Authority was incorporated August 24, 1997 and"


class TrainerState:
    def __init__(self, max_steps: int, max_tokens: Optional[int] = None):
        self.global_step = 0
        self.max_steps = max_steps
        self.global_train_tokens_seen = 0
        self.max_tokens = max_tokens or max_steps * 1000


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    elif torch.backends.mps.is_available():
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def create_model(train_config: Dict, device: torch.device) -> torch.nn.Module:
    if train_config["model_arch"] == "gpt":
        model = GPTModel(train_config)
    elif train_config["model_arch"] == "nanogpt":
        model = nanoGPTModel(train_config)
    elif train_config["model_arch"] == "ngpt":
        model = nGPTModel(train_config)
    else:
        model = LlamaModel(train_config)
    
    return model.to(device)


def get_fsdp_config(train_config: Dict) -> Dict:
    """Get FSDP configuration from train config."""
    fsdp_config = train_config.get("distributed", {}).get("fsdp", {})
    
    # Map string to enum values
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
        "sharding_strategy": sharding_strategy_map.get(
            fsdp_config.get("sharding_strategy", "FULL_SHARD"), 
            ShardingStrategy.FULL_SHARD
        ),
        "mixed_precision": fsdp_config.get("mixed_precision", True),
        "activation_checkpointing": fsdp_config.get("activation_checkpointing", True),
        "cpu_offload": fsdp_config.get("cpu_offload", False),
        "backward_prefetch": backward_prefetch_map.get(
            fsdp_config.get("backward_prefetch", "BACKWARD_PRE"),
            BackwardPrefetch.BACKWARD_PRE
        ),
        "forward_prefetch": fsdp_config.get("forward_prefetch", False),
        "use_orig_params": fsdp_config.get("use_orig_params", True),
    }


def setup_fsdp_model(model: torch.nn.Module, train_config: Dict) -> torch.nn.Module:
    """Setup model with FSDP."""
    fsdp_config = get_fsdp_config(train_config)
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100_000_000
    )
    mixed_precision_policy = None
    if fsdp_config["mixed_precision"]:
        mixed_precision_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float32,  # Keep buffers in fp32 for stability
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
    
    return model


def setup_distributed_model(model: torch.nn.Module, rank: int, world_size: int, train_config: Dict) -> torch.nn.Module:
    if world_size == 1:
        return model
    
    strategy = train_config.get("distributed", {}).get("strategy", "ddp").lower()
    
    if strategy == "fsdp":
        return setup_fsdp_model(model, train_config)
    else:
        if torch.cuda.is_available():
            return DDP(model, device_ids=[rank], output_device=rank)
        else:
            return DDP(model)


def create_optimizer_and_scheduler(model: torch.nn.Module, train_config: Dict, num_epochs: int, train_loader_len: int):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["initial_lr"],
        weight_decay=train_config["weight_decay"],
    )
    
    scheduler = CosWithWarmup(
        warmup_fraction=train_config.get("warmup_fraction", 0.1),
        alpha_f=train_config.get("alpha_f", 0.1)
    )
    
    return optimizer, scheduler


def update_learning_rate(optimizer: torch.optim.Optimizer, scheduler, trainer_state: TrainerState):
    for param_group in optimizer.param_groups:
        scheduler.set_lr(param_group, trainer=trainer_state)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    tokens_seen: int,
    train_ce_losses: List[float],
    train_z_losses: List[float],
    track_tokens_seen: List[int],
    train_config: Dict,
    rank: int
) -> None:
    """Save checkpoint with separate files for model, optimizer, and training state."""
    checkpoint_config = train_config.get("checkpoint", {})
    save_dir = checkpoint_config.get("save_dir", "checkpoints")
    checkpoint_dir = Path(save_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    step_dir = checkpoint_dir / f"step_{global_step}"
    step_dir.mkdir(exist_ok=True)
    
    strategy = train_config.get("distributed", {}).get("strategy", "ddp").lower()
    
    if strategy == "fsdp" and isinstance(model, FSDP):
        save_fsdp_checkpoint(model, optimizer, step_dir, checkpoint_config, rank)
    else:
        save_ddp_checkpoint(model, optimizer, step_dir, checkpoint_config, rank)
    
    if rank == 0 and checkpoint_config.get("save_training_state", True):
        training_state = {
            "epoch": epoch,
            "global_step": global_step,
            "tokens_seen": tokens_seen,
            "train_ce_losses": train_ce_losses,
            "train_z_losses": train_z_losses,
            "track_tokens_seen": track_tokens_seen,
            "config": train_config
        }
        torch.save(training_state, step_dir / "train.pt")
    
    if rank == 0:
        print(f"Checkpoint saved at step {global_step} in {step_dir}")
        keep_last_n = checkpoint_config.get("keep_last_n")
        if keep_last_n and keep_last_n > 0:
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n)


def save_fsdp_checkpoint(
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step_dir: Path,
    checkpoint_config: Dict,
    rank: int
) -> None:
    if checkpoint_config.get("save_model", True):
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        ):
            model_state_dict = model.state_dict()
            torch.save(model_state_dict, step_dir / f"model_rank_{rank}.pt")
    
    if checkpoint_config.get("save_optimizer", True):
        with FSDP.state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
        ):
            optim_state_dict = FSDP.optim_state_dict(model, optimizer)
            torch.save(optim_state_dict, step_dir / f"optim_rank_{rank}.pt")


def save_ddp_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step_dir: Path,
    checkpoint_config: Dict,
    rank: int
) -> None:
    if rank != 0:
        return

    model_to_save = model.module if hasattr(model, 'module') else model
    
    if checkpoint_config.get("save_model", True):
        torch.save(model_to_save.state_dict(), step_dir / "model.pt")
    
    if checkpoint_config.get("save_optimizer", True):
        torch.save(optimizer.state_dict(), step_dir / "optim.pt")


def cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    step_dirs = [d for d in checkpoint_dir.iterdir() if d.is_dir() and d.name.startswith("step_")]
    if len(step_dirs) <= keep_last_n:
        return
    
    step_dirs.sort(key=lambda x: int(x.name.split("_")[1]))
    for old_dir in step_dirs[:-keep_last_n]:
        import shutil
        shutil.rmtree(old_dir)
        print(f"Removed old checkpoint: {old_dir}")


def should_save_checkpoint(global_step: int, train_config: Dict) -> bool:
    """Check if we should save a checkpoint at this step."""
    checkpoint_config = train_config.get("checkpoint", {})
    save_frequency = checkpoint_config.get("save_frequency", 1000)
    return global_step > 0 and global_step % save_frequency == 0


def get_model_dtype(model: torch.nn.Module, train_config: Dict) -> torch.dtype:
    """Get the appropriate dtype for model inputs based on FSDP mixed precision."""
    strategy = train_config.get("distributed", {}).get("strategy", "ddp").lower()
    
    if strategy == "fsdp" and isinstance(model, FSDP):
        fsdp_config = train_config.get("distributed", {}).get("fsdp", {})
        if fsdp_config.get("mixed_precision", False):
            return torch.float16
    
    return torch.float32


def convert_inputs_to_model_dtype(input_batch: torch.Tensor, target_batch: torch.Tensor, model: torch.nn.Module, train_config: Dict):
    """Convert input tensors to the appropriate dtype for the model."""
    model_dtype = get_model_dtype(model, train_config)
    
    if model_dtype != input_batch.dtype:
        input_batch = input_batch.to(dtype=model_dtype)
    
    # Keep target as int64 for loss computation
    return input_batch, target_batch


def generate_and_print_sample(model, tokenizer, start_context, device, rank):
    if rank == 0:
        model.eval()
        base_model = model.module if hasattr(model, "module") else model
        context_size = base_model.pos_emb.weight.shape[0]
        encoded = tokenizer.encode(start_context)
        encoded = torch.tensor(
            encoded, dtype=torch.long, device=device
        )  # .unsqueeze(0)

        with torch.no_grad():
            token_ids = generate_text_simple(
                model=model, idx=encoded, max_new_tokens=50, context_size=context_size
            )
            decoded_text = tokenizer.decode(token_ids)  # .squeeze(0).tolist())
            cleaned_text = decoded_text.replace("\n", " ")
            print(f"[Rank {rank}] Generated: {cleaned_text}")
        model.train()


def train_911(
    rank,
    world_size,
    train_config,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
):
    setup(rank, world_size)
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model = create_model(train_config, device)
    model = setup_distributed_model(model, rank, world_size, train_config)
    if rank == 0:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="911-training", config=train_config)

    train_loader, tokenizer = create_train_loader(train_config)
    optimizer, learning_rate_scheduler = create_optimizer_and_scheduler(
        model, train_config, num_epochs, len(train_loader)
    )

    train_ce_losses, train_z_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = 0
    trainer = TrainerState(
        max_steps=num_epochs * len(train_loader),
        max_tokens=num_epochs * len(train_loader) * train_config.get("sequence_length", 1024)
    )

    for epoch in range(num_epochs):
        model.train()
        ce_epoch_loss = 0.0
        z_epoch_loss = 0.0

        for input_batch, target_batch in train_loader:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)
            
            # Convert inputs to appropriate dtype for FSDP mixed precision
            input_batch, target_batch = convert_inputs_to_model_dtype(
                input_batch, target_batch, model, train_config
            )

            optimizer.zero_grad()
            ce_loss, z_loss = calc_loss_batch(input_batch, target_batch, model, device)
            
            # Combine losses before backward to avoid double backward pass
            total_loss = (ce_loss + z_loss) / world_size
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_VALUE)

            optimizer.step()
            trainer.global_step = global_step
            trainer.global_train_tokens_seen = tokens_seen
            update_learning_rate(optimizer, learning_rate_scheduler, trainer)

            tokens_seen += input_batch.numel()
            ce_epoch_loss += ce_loss.item()
            z_epoch_loss += z_loss.item()

            if global_step % eval_freq == 0:
                dist.barrier()
                total_ce_loss, total_z_loss = calc_total_loss(
                    train_loader, model, device, eval_iter
                )

                gathered_ce_losses = [
                    torch.zeros(1).to(device) for _ in range(world_size)
                ]
                dist.all_gather(
                    gathered_ce_losses, torch.tensor([total_ce_loss]).to(device)
                )
                avg_train_ce_loss = (
                    sum(loss.item() for loss in gathered_ce_losses) / world_size
                )

                gathered_z_losses = [
                    torch.zeros(1).to(device) for _ in range(world_size)
                ]
                dist.all_gather(
                    gathered_z_losses, torch.tensor([total_z_loss]).to(device)
                )
                avg_train_z_loss = (
                    sum(loss.item() for loss in gathered_z_losses) / world_size
                )

                if rank == 0:
                    train_ce_losses.append(avg_train_ce_loss)
                    train_z_losses.append(avg_train_z_loss)
                    track_tokens_seen.append(tokens_seen)
                    wandb.log(
                        {
                            "global train CE loss": avg_train_ce_loss,
                            "global train Z loss": avg_train_z_loss,
                            "lr": optimizer.param_groups[0]['lr'],
                            "tokens_seen": tokens_seen,
                            "epoch": epoch,
                            "global_step": global_step,
                        }
                    )
                generate_and_print_sample(model, tokenizer, start_context, device, rank)

            global_step += 1
            
            if should_save_checkpoint(global_step, train_config):
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    tokens_seen=tokens_seen,
                    train_ce_losses=train_ce_losses,
                    train_z_losses=train_z_losses,
                    track_tokens_seen=track_tokens_seen,
                    train_config=train_config,
                    rank=rank
                )

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=num_epochs,
        global_step=global_step,
        tokens_seen=tokens_seen,
        train_ce_losses=train_ce_losses,
        train_z_losses=train_z_losses,
        track_tokens_seen=track_tokens_seen,
        train_config=train_config,
        rank=rank
    )
    
    if rank == 0:
        wandb.finish()

    cleanup()

    return train_ce_losses, train_z_losses, track_tokens_seen


def run_training(rank, world_size, train_config):
    train_ce_losses, train_z_losses, tokens_seen = train_911(
        rank=rank,
        world_size=world_size,
        train_config=train_config,
        num_epochs=train_config["num_epochs"],
        eval_freq=train_config.get("eval_freq", DEFAULT_EVAL_FREQ),
        eval_iter=train_config.get("eval_iter", DEFAULT_EVAL_ITER),
        start_context=train_config.get("start_context", DEFAULT_START_CONTEXT),
    )


def main():
    train_config = load_config()

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        mp.spawn(
            run_training, args=(world_size, train_config), nprocs=world_size, join=True
        )
    else:
        world_size = 1
        run_training(0, world_size, train_config)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
