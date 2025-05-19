import torch
import torch.nn as nn
import yaml
from data.data_tokenizer import TokenizerV0, load_txt_file
from data.dataset_utils import create_dataloader_v1
from nn.gpt_block import GPTModel, generate_text_simple
from nn.loss_function import calc_loss_batch, calc_total_loss
import wandb

with open("initial_data.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

vocab = load_txt_file(raw_text)
tokenizer = TokenizerV0(vocab)
total_characters = len(raw_text)
total_tokens = len(tokenizer.encode(raw_text))

train_ratio = 0.90
split_idx = int(train_ratio * len(raw_text))
train_data = raw_text[:split_idx]

torch.manual_seed(123)

with open('config.yaml') as f:
    train_config = yaml.safe_load(f)

train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=train_config["max_seq_length"],
    stride=train_config["max_seq_length"],
    drop_last=True,
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(train_config)
model.to(device)

wandb.login()
run = wandb.init(project="911-training", config=train_config)

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = tokenizer.encode(start_context)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model,
            idx=encoded,
            max_new_tokens=50,
            context_size=context_size
        )
        decoded_text = tokenizer.decode(token_ids)
        print(decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(model, train_loader, optimizer, device,
                      num_epochs, eval_freq, eval_iter, start_context):
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
            optimizer.step()
            step_loss.append(loss.item())
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss = calc_total_loss(train_loader, model, device, eval_iter)
                train_losses.append(train_loss)
                track_tokens_seen.append(tokens_seen)
                wandb.log({"global train loss": train_loss})
                generate_and_print_sample(
                    model, 
                    train_loader.dataset.tokenizer,
                    device,
                    start_context
                )
    if not train_losses:
        train_losses = step_loss
    return train_losses, track_tokens_seen

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.0004, 
    weight_decay=0.1
)

train_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=train_config["num_epochs"],
    eval_freq=5,
    eval_iter=1,
    start_context="Estimates of the age of the Moon range from"
)

print(train_losses)