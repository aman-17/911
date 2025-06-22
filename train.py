import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from torch.nn.parallel import DistributedDataParallel as DDP

from config_utils import load_config
from data.dataset_utils import create_train_loader
from nn.transfomer.model.gpt_model import GPTModel, nanoGPTModel, nGPTModel
from nn.transfomer.model.llama_model import LlamaModel
from nn.loss_function import calc_loss_batch, calc_total_loss
from nn.utils import generate_text_simple


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
    if train_config["model_arch"] == "gpt":
        model = GPTModel(train_config)
    elif train_config["model_arch"] == "nanogpt":
        model = nanoGPTModel(train_config)
    elif train_config["model_arch"] == "ngpt":
        model = nGPTModel(train_config)
    else:
        model = LlamaModel(train_config)

    model = model.to(device)
    if torch.cuda.is_available():
        model = DDP(model, device_ids=[rank], output_device=rank)
    elif world_size > 1:
        model = DDP(model)
    if rank == 0:
        wandb.login(key=os.getenv("WANDB_API_KEY"))
        wandb.init(project="911-training", config=train_config)

    train_loader, tokenizer = create_train_loader(train_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config["initial_lr"],
        weight_decay=train_config["weight_decay"],
    )

    learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader), eta_min=0.0, last_epoch=-1
    )

    train_ce_losses, train_z_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        ce_epoch_loss = 0.0
        z_epoch_loss = 0.0

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            ce_loss, z_loss = calc_loss_batch(input_batch, target_batch, model, device)
            scaled_ce_loss = ce_loss / world_size
            scaled_ce_loss.backward()
            scaled_z_loss = z_loss / world_size
            scaled_z_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            learning_rate_scheduler.step()

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
                            "lr": learning_rate_scheduler.get_last_lr()[0],
                            "tokens_seen": tokens_seen,
                            "epoch": epoch,
                            "global_step": global_step,
                        }
                    )
                generate_and_print_sample(model, tokenizer, start_context, device, rank)

            global_step += 1

    if rank == 0:
        torch.save(model.module.state_dict(), "model_checkpoint.pt")
        wandb.finish()

    cleanup()

    return train_ce_losses, train_z_losses, track_tokens_seen


def run_training(rank, world_size, train_config):
    train_ce_losses, train_z_losses, tokens_seen = train_911(
        rank=rank,
        world_size=world_size,
        train_config=train_config,
        num_epochs=train_config["num_epochs"],
        eval_freq=10,
        eval_iter=50,
        start_context="The Bowmanstown Borough Authority was incorporated August 24, 1997 and",
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
