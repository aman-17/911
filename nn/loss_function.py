from typing import Literal, Tuple

import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Cross entropy loss that optionally computes the softmax auxiliary loss (z-loss) as well.

    :param logits: Predicted unnormalized logits with shape ``(N, vocab_size)``.
    :param labels: Ground truth class indices with shape ``(N,)``.
    :param ignore_index: Specifies a target value that is ignored and does not contribute to
        the input gradient.
    :param reduction: Specifies the reduction to apply to the output.
        Can be "none", "mean", or "sum".
    :param compute_z_loss: Compute the softmax auxiliary loss as well.
    :param z_loss_multiplier: The multiplier to apply to the z-loss.

    :returns: The cross entropy loss and optionally the z-loss.
    """
    logits = logits.float()
    ce_loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    z_squared = logits.logsumexp(-1).pow(2)
    mask = labels != ignore_index
    if reduction == "mean":
        z_squared = (z_squared * mask).sum() / mask.sum()
    elif reduction == "sum":
        z_squared = (z_squared * mask).sum()

    z_loss = z_loss_multiplier * z_squared

    return ce_loss, z_loss


def calc_loss_batch(input_batch, target_batch, model, device) -> tuple[torch.Tensor, torch.Tensor]:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    ce_loss, z_loss = cross_entropy_loss(logits=logits.view(-1, logits.size(-1)), labels=target_batch.view(-1))
    return ce_loss, z_loss


def calc_total_loss_for_dataset_v1(data_loader, model, device, num_batches=None) -> tuple[float, float]:
    total_ce_loss = 0.0
    total_ze_loss = 0.0
    if num_batches is None:
        num_batches = len(data_loader)
        print(f"Calculating loss for {num_batches} batches")
    else:
        num_batches = min(num_batches, len(data_loader))
        print(f"Calculating loss for {num_batches} batches")

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            ce_loss, z_loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_ce_loss += ce_loss.item()
            total_ze_loss += z_loss.item()
        else:
            break
    return total_ce_loss / (num_batches + 0.000001), total_ze_loss / (num_batches + 0.000001)


def calc_total_loss(data_loader, model, device, num_batches=None) -> tuple[float, float]:
    model.eval()
    total_ce_loss = 0.0
    total_ze_loss = 0.0
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
                ce_loss, z_loss = calc_loss_batch(input_batch, target_batch, model, device)
                total_ce_loss += ce_loss.item()
                total_ze_loss += z_loss.item()
                batch_count += 1
            else:
                break

    model.train()
    return total_ce_loss / max(1, batch_count), total_ze_loss / max(1, batch_count)
