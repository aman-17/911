import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import (
    FullStateDictConfig,
    StateDictType,
    FullOptimStateDictConfig,
)

import sys
sys.path.append('.')
from config_utils import load_config
from nn.transfomer.model.gpt_model import GPTModel, nanoGPTModel, nGPTModel
from nn.transfomer.model.llama_model import LlamaModel


def create_model_from_config(config: Dict) -> torch.nn.Module:
    if config["model_arch"] == "gpt":
        return GPTModel(config)
    elif config["model_arch"] == "nanogpt":
        return nanoGPTModel(config)
    elif config["model_arch"] == "ngpt":
        return nGPTModel(config)
    else:
        return LlamaModel(config)


def setup_fsdp_for_unsharding(model: torch.nn.Module) -> FSDP:
    from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
    import functools
    
    auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100_000_000
    )
    
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        use_orig_params=True,
    )
    
    return model


def setup_distributed():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29502"
    dist.init_process_group("nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)


def unshard_model_checkpoint(
    sharded_checkpoint_dir: Path,
    model: FSDP,
    output_path: Path
) -> None:
    model_files = list(sharded_checkpoint_dir.glob("model_rank_*.pt"))
    if not model_files:
        print("No sharded model files found!")
        return

    sharded_state_dicts = []
    for model_file in sorted(model_files):
        print(f"Loading {model_file}")
        sharded_state_dict = torch.load(model_file, map_location="cpu")
        sharded_state_dicts.append(sharded_state_dict)
    
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
    ):
        full_state_dict = model.state_dict()
    
    torch.save(full_state_dict, output_path / "model.pt")
    print(f"Unsharded model saved to {output_path / 'model.pt'}")


def unshard_optimizer_checkpoint(
    sharded_checkpoint_dir: Path,
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    output_path: Path
) -> None:
    optim_files = list(sharded_checkpoint_dir.glob("optim_rank_*.pt"))
    if not optim_files:
        print("No sharded optimizer files found!")
        return
    
    sharded_optim_state_dicts = []
    for optim_file in sorted(optim_files):
        print(f"Loading {optim_file}")
        sharded_optim_state_dict = torch.load(optim_file, map_location="cpu")
        sharded_optim_state_dicts.append(sharded_optim_state_dict)

    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True)
    ):
        full_optim_state_dict = FSDP.optim_state_dict(model, optimizer)
    
    torch.save(full_optim_state_dict, output_path / "optim.pt")
    print(f"Unsharded optimizer saved to {output_path / 'optim.pt'}")


def copy_training_state(
    sharded_checkpoint_dir: Path,
    output_path: Path
) -> None:
    train_file = sharded_checkpoint_dir / "train.pt"
    if train_file.exists():
        shutil.copy2(train_file, output_path / "train.pt")
        print(f"Training state copied to {output_path / 'train.pt'}")
    else:
        print("No training state file found!")


def unshard_checkpoint(
    checkpoint_dir: str,
    output_dir: str,
    model_config_path: Optional[str] = None
) -> None:
    """Main function to unshard FSDP checkpoint."""
    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory {checkpoint_path} does not exist!")
    
    output_path.mkdir(parents=True, exist_ok=True)
    if model_config_path:
        config = load_config(model_config_path)
    else:
        train_file = checkpoint_path / "train.pt"
        if train_file.exists():
            training_state = torch.load(train_file, map_location="cpu")
            config = training_state.get("config", {})
        else:
            raise ValueError("No model configuration provided and no train.pt found!")
    
    print(f"Using model architecture: {config.get('model_arch', 'unknown')}")
    if torch.cuda.is_available():
        setup_distributed()
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    model = create_model_from_config(config)
    model = model.to(device)
    model = setup_fsdp_for_unsharding(model)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.get("initial_lr", 1e-4),
        weight_decay=config.get("weight_decay", 0.1),
    )
    if any(checkpoint_path.glob("model_rank_*.pt")):
        unshard_model_checkpoint(checkpoint_path, model, output_path)
    elif (checkpoint_path / "model.pt").exists():
        shutil.copy2(checkpoint_path / "model.pt", output_path / "model.pt")
        print("Model checkpoint copied (already unsharded)")
    
    if any(checkpoint_path.glob("optim_rank_*.pt")):
        unshard_optimizer_checkpoint(checkpoint_path, model, optimizer, output_path)
    elif (checkpoint_path / "optim.pt").exists():
        shutil.copy2(checkpoint_path / "optim.pt", output_path / "optim.pt")
        print("Optimizer checkpoint copied (already unsharded)")
    
    copy_training_state(checkpoint_path, output_path)
    print(f"Unsharding complete! Unsharded checkpoint saved to {output_path}")
    
    if torch.cuda.is_available():
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Unshard FSDP checkpoints")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to sharded checkpoint directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output unsharded checkpoint directory"
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=None,
        help="Path to model configuration file (optional, will try to load from train.pt)"
    )
    
    args = parser.parse_args()
    
    try:
        unshard_checkpoint(
            args.checkpoint_dir,
            args.output_dir,
            args.model_config_path
        )
    except Exception as e:
        print(f"Error unsharding checkpoint: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
    # python unshard_checkpoint.py --checkpoint_dir checkpoints/step_1000 --output_dir unsharded_checkpoints/step_1000 --model_config_path config.yaml
