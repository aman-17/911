import torch
import torch.nn as nn
from hyperparams.optimizer import Optimizer
from typing import Any, Optional, cast
from torch import Tensor
from functools import wraps
from weakref import ref
import warnings


class LRScheduler:
    r"""Adjusts the learning rate during optimization."""

    _get_lr_called_within_step: bool = False

    def __init__(
        self,
        optimizer: Optimizer,
        last_epoch: int = -1,
    ):  # noqa: D107
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                initial_lr = group["lr"]
                if isinstance(initial_lr, Tensor):
                    initial_lr = initial_lr.clone()
                group.setdefault("initial_lr", initial_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        f"in param_groups[{i}] when resuming an optimizer"
                    )
        self.base_lrs: list[float] = [
            group["initial_lr"] for group in optimizer.param_groups
        ]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def patch_track_step_called(opt: Optimizer):
            if hasattr(opt.step, "_wrapped_by_lr_sched"):
                # we've already patched
                return opt.step

            def wrap_step(step_fn):
                opt_ref = ref(self.optimizer)
                func = step_fn.__func__

                @wraps(func)
                def wrapper(*args, **kwargs):
                    opt = opt_ref()
                    opt._opt_called = True  # type: ignore[union-attr]
                    return func.__get__(opt, opt.__class__)(*args, **kwargs)

                wrapper._wrapped_by_lr_sched = True  # type: ignore[attr-defined]
                return wrapper

            opt.step = wrap_step(opt.step)  # type: ignore[method-assign]

        patch_track_step_called(self.optimizer)
        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and perform a step."""
        self._step_count = 0
        self.step()

    def state_dict(self):
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> list[float]:
        """Return last computed learning rate by current scheduler."""
        return self._last_lr

    def get_lr(self) -> list[float]:
        """Compute learning rate using chainable form of the scheduler."""
        raise NotImplementedError

    def step(self, epoch: Optional[int] = None):
        """Perform a step."""
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_wrapped_by_lr_sched"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif not getattr(self.optimizer, "_opt_called", False):
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        self._step_count += 1

        with _enable_get_lr_call(self):
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                if hasattr(self, "_get_closed_form_lr"):
                    values = cast(list[float], self._get_closed_form_lr())
                else:
                    values = self.get_lr()

        for param_group, lr in zip(self.optimizer.param_groups, values):
            if isinstance(param_group["lr"], Tensor):
                param_group["lr"].fill_(lr)
            else:
                param_group["lr"] = lr

        self._last_lr: list[float] = [
            group["lr"] for group in self.optimizer.param_groups
        ]


class LinearLR(LRScheduler):
        ]

class CosineAnnealingLR(LRScheduler):
        ]
