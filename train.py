import torch
import yaml

import wandb
from data.dataset_utils import create_train_loader
from nn.gpt_block import GPTModel, nGPTModel
from nn.utils import generate_text_simple
from nn.loss_function import calc_loss_batch, calc_total_loss

with open("config.yaml") as f:
    train_config = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nGPTModel(train_config)
model.to(device)

wandb.login()
run = wandb.init(project="911-training", config=train_config)

train_loader, tokenizer = create_train_loader(train_config)


def generate_and_print_sample(model, tokenizer, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = tokenizer.encode(start_context)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded, max_new_tokens=50, context_size=context_size
        )
        decoded_text = tokenizer.decode(token_ids)
        print(decoded_text.replace("\n", " "))
    model.train()


def train_911(
    model,
    train_loader,
    optimizer,
    lr_scheduler,
    device,
    num_epochs,
    eval_freq,
    eval_iter,
    start_context,
):
    train_losses, track_tokens_seen = [], []
    step_loss = []
    tokens_seen = 0
    global_step = -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            wandb.log({"Batch loss": loss.item()})
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            model.normalize_matrices()
            # lr_scheduler.step()
            step_loss.append(loss.item())
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss = calc_total_loss(train_loader, model, device, eval_iter)
                train_losses.append(train_loss)
                track_tokens_seen.append(tokens_seen)
                wandb.log(
                    {
                        "global train loss": train_loss,
                        "lr": lr_scheduler.get_last_lr()[0],
                        "grad norm": torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0),
                        "tokens_seen": tokens_seen,
                    }
                )
                generate_and_print_sample(
                    model, train_loader.dataset.tokenizer, start_context
                )
    # if hasattr(model, 'normalize_matrices'):
    #     model.normalize_matrices()

    if not train_losses:
        train_losses = step_loss
    return train_losses, track_tokens_seen


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_config["initial_lr"],
    weight_decay=train_config["weight_decay"],
)
learning_rate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, train_config["num_epochs"], eta_min=0.0, last_epoch=-1
)

train_losses, tokens_seen = train_911(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    lr_scheduler=learning_rate_scheduler,
    device=device,
    num_epochs=train_config["num_epochs"],
    eval_freq=10,
    eval_iter=50,
    start_context="Tell me about Porsche Speedster",
)
wandb.finish()
