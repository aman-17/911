import torch

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), target_batch.view(-1)
    )
    return loss

def calc_total_loss(data_loader, model, device, num_batches=None):
    total_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)
        print(f"Calculating loss for {num_batches} batches")
    else:
        num_batches = min(num_batches, len(data_loader))
        print(f"Calculating loss for {num_batches} batches")
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break      
    return total_loss / (num_batches + 0.000001)