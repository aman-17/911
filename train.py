import torch
import torch.nn as nn
import yaml
from data.data_tokenizer import TokenizerV0, load_txt_file
from data.dataset_utils import create_dataloader_v1
from nn.gpt_block import GPTModel, generate_text_simple
from nn.loss_function import calc_loss_batch, calc_total_loss
# import matplotlib.pyplot as plt

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
    max_length=train_config["context_length"],
    stride=train_config["context_length"],
    drop_last=True,
    shuffle=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GPTModel(train_config)
model.to(device)

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
            print(f"Step {global_step}, Batch Loss: {loss.item()}")
            optimizer.step()
            step_loss.append(loss.item())
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss = calc_total_loss(train_loader, model, device, eval_iter)
                train_losses.append(train_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}")
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
num_epochs = 3

train_losses, tokens_seen = train_model_simple(
    model=model,
    train_loader=train_loader,
    optimizer=optimizer,
    device=device,
    num_epochs=num_epochs,
    eval_freq=5,
    eval_iter=1,
    start_context="Estimates of the age of the Moon range from"
)

print(train_losses)

# def plot_losses(epochs_seen, tokens_seen, train_losses):
#     fig, ax1 = plt.subplots(figsize=(5, 3))
    
#     # Plot losses against epochs
#     ax1.plot(epochs_seen, train_losses, label="Training loss")
#     ax1.plot(epochs_seen, linestyle="--", label="Validation loss")
#     ax1.set_xlabel("Epochs")
#     ax1.set_ylabel("Loss")
#     ax1.legend(loc="upper right")
    
#     # Add second x-axis for tokens
#     ax2 = ax1.twiny()
#     ax2.plot(tokens_seen, train_losses, alpha=0)
#     ax2.set_xlabel("Tokens seen")
    
#     fig.tight_layout()
#     plt.show()

# Call plotting function
# epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
# plot_losses(epochs_tensor, tokens_seen, train_losses)