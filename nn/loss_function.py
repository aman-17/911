import logging
from typing import Callable, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import (
    PrepareModuleInput,
    PrepareModuleOutput,
    parallelize_module,
)

from nn.distributed.utils import get_local_tensor

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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
    loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2)
    mask = labels != ignore_index
    if reduction == "mean":
        z_squared = (z_squared * mask).sum() / mask.sum()
    elif reduction == "sum":
        z_squared = (z_squared * mask).sum()

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss


_fused_linear_cross_entropy_loss: Optional[Callable] = None

try:
    from liger_kernel.ops.fused_linear_cross_entropy import (  # type: ignore
        LigerFusedLinearCrossEntropyFunction,
    )

    _fused_linear_cross_entropy_loss = LigerFusedLinearCrossEntropyFunction.apply
except ImportError:
    pass
except Exception:
    log.exception("Error importing liger-kernel")


@torch._dynamo.disable()
def fused_linear_cross_entropy_loss(
    _input: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    bias: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    reduction: Literal["mean", "sum", "none"] = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
    ce_weight: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
    softcap: Optional[float] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Cross entropy loss fused with the linear layer that computes the logits.

    :param _input: The inputs to pass through the linear layer to produce the logits ``(N, D)``.
    :param weight: The weight of the linear layer.
    :param labels: Ground truth class indices with shape ``(N,)``.
    :param bias: Optional bias for the linear layer.
    :param ignore_index: Specifies a target value that is ignored and does not contribute to
        the input gradient.
    :param reduction: Specifies the reduction to apply to the output.
        Can be "none", "mean", or "sum".
    :param compute_z_loss: Compute the softmax auxiliary loss as well.
    :param z_loss_multiplier: The multiplier to apply to the z-loss.

    :returns: The cross entropy loss and optionally the z-loss.
    """
    if _fused_linear_cross_entropy_loss is None:
        raise RuntimeError("'fused_linear_cross_entropy_loss' requires liger-kernel")
    ce_loss, z_loss = _fused_linear_cross_entropy_loss(
        _input,
        weight,
        labels,
        bias,
        ce_weight,
        ignore_index,
        z_loss_multiplier,
        label_smoothing,
        reduction,
        softcap,
        compute_z_loss,
    )
    if compute_z_loss:
        return ce_loss, z_loss
    else:
        return ce_loss, None


class _CELossFnWrapper(nn.Module):
    def __init__(
        self,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction: Literal["mean", "sum", "none"] = reduction
        self.z_loss_multiplier = z_loss_multiplier
        self.base_loss_fn = cross_entropy_loss
        self.tp_enabled = False

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if logits.shape[:-1] != labels.shape:
            raise RuntimeError(f"expected labels to have shape {logits.shape[:-1]}, but found {tuple(labels.shape)} instead")

        # shape: (B * S, V)
        logits_for_loss = logits.view(-1, logits.shape[-1])

        # shape: (B, S) -> (B * S,)
        labels_for_loss = labels.view(-1)

        ce_loss, z_loss = self.base_loss_fn(
            logits_for_loss,
            labels_for_loss,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            z_loss_multiplier=self.z_loss_multiplier or 1e-4,
        )

        if self.reduction == "none":
            ce_loss = ce_loss.view(labels.shape)
            if z_loss is not None:
                z_loss = z_loss.view(labels.shape)
        elif self.tp_enabled:
            ce_loss = ce_loss.unsqueeze(0)
            if z_loss is not None:
                z_loss = z_loss.unsqueeze(0)

        return ce_loss, z_loss


class CrossEntropyLoss(nn.Module):
    def __init__(
        self,
        *,
        ignore_index: int = -100,
        reduction: Literal["mean", "sum", "none"] = "mean",
        z_loss_multiplier: Optional[float] = None,
        compile: bool = False,
    ):
        super().__init__()

        self.loss_fn = _CELossFnWrapper(
            ignore_index=ignore_index,
            reduction=reduction,
            z_loss_multiplier=z_loss_multiplier,
        )
        self._tp_enabled = False

        self._compile_enabled = False
        if compile:
            if torch.cuda.is_available():
                self.compile()
                self._compile_enabled = True
            else:
                log.warning("Skipping loss compilation since CUDA is not available")

    @property
    def compile_enabled(self) -> bool:
        return self._compile_enabled

    @property
    def tp_enabled(self) -> bool:
        return self._tp_enabled

    @property
    def z_loss_enabled(self) -> bool:
        return self.loss_fn.z_loss_multiplier is not None

    @property
    def reduction(self) -> Literal["sum", "mean", "none"]:
        return self.loss_fn.reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        div_factor: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute the CE loss and optionally Z-loss.

        :param logits: The logits of shape ``(B, S, V)``.
        :param labels: The target labels of shape ``(B, S)``.
        """
        ce_loss, z_loss = self.loss_fn(get_local_tensor(logits), get_local_tensor(labels))

        if self.reduction != "none" and ce_loss.numel() > 1:
            # This will be the same case with tensor/sequence parallel loss where we have a DTensor.
            assert self.tp_enabled
            if self.reduction == "sum":
                ce_loss = ce_loss.sum()
                if z_loss is not None:
                    z_loss = z_loss.sum()
            elif self.reduction == "mean":
                ce_loss = ce_loss.mean()
                if z_loss is not None:
                    z_loss = z_loss.mean()
            else:
                raise NotImplementedError(self.reduction)

        if div_factor is not None:
            ce_loss = ce_loss / div_factor
            if z_loss is not None:
                z_loss = z_loss / div_factor

        return ce_loss, z_loss

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layouts: Optional[Tuple[Placement, ...]] = None,
        shard_dimension: int = 1,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = False,
    ):
        desired_input_layouts: Tuple[Placement, ...]
        if input_layouts is None or len(input_layouts) == 3:
            desired_input_layouts = (Shard(shard_dimension), Shard(shard_dimension), Replicate())
        elif len(input_layouts) == 2:
            desired_input_layouts = (Shard(shard_dimension), Shard(shard_dimension))
        else:
            raise ValueError(f"expected 2 or 3 input layouts, found {len(input_layouts)}")
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleInput(
                input_layouts=input_layouts,  # type: ignore
                desired_input_layouts=desired_input_layouts,  # type: ignore
                use_local_output=False,
            ),
        )

        inner_output_layout = Shard(shard_dimension) if self.reduction == "none" else Shard(0)
        parallelize_module(
            self.loss_fn,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                output_layouts=(  # type: ignore
                    inner_output_layout,
                    None if not self.z_loss_enabled else inner_output_layout,
                ),
                desired_output_layouts=(  # type: ignore
                    inner_output_layout,
                    None if not self.z_loss_enabled else inner_output_layout,
                ),
                use_local_output=False,
            ),
        )

        expected_output_layout = Shard(shard_dimension) if self.reduction == "none" else Replicate()
        desired_output_layout = output_layout or Replicate()
        parallelize_module(
            self,
            device_mesh=tp_mesh,
            parallelize_plan=PrepareModuleOutput(
                output_layouts=(  # type: ignore
                    expected_output_layout,
                    None if not self.z_loss_enabled else expected_output_layout,
                ),
                desired_output_layouts=(  # type: ignore
                    desired_output_layout,
                    None if not self.z_loss_enabled else desired_output_layout,
                ),
                use_local_output=use_local_output,
            ),
        )

        self._tp_enabled = True
        self.loss_fn.tp_enabled = True


def calc_loss_batch(input_batch, target_batch, model, device) -> tuple[torch.Tensor, torch.Tensor]:
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss_fn = CrossEntropyLoss(z_loss_multiplier=1e-4)
    ce_loss, z_loss = loss_fn(logits, target_batch)

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
