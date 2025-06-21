import torch
import torch.distributed as dist


def calc_loss_batch(input_batch, target_batch, model, device) -> torch.Tensor:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), target_batch.view(-1)
    )
    return loss


def calc_total_loss_for_dataset_v1(
    data_loader, model, device, num_batches=None
) -> float:
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


def calc_total_loss(data_loader, model, device, num_batches=None) -> float:
    model.eval()
    total_loss = 0.0
    batch_count = 0

    try:
        if num_batches is None:
            num_batches = len(data_loader)
        else:
            num_batches = min(num_batches, len(data_loader))
    except (TypeError, AttributeError):
        if num_batches is None:
            num_batches = 5

    with torch.no_grad():
        for i, (input_batch, target_batch) in enumerate(data_loader):
            if i < num_batches:
                loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_loss += loss.item()
                batch_count += 1
            else:
                break

    model.train()
    return total_loss / max(1, batch_count)
