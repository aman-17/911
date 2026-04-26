import torch
import torch.nn as nn


class ActivationCollector:
    """Collects residual stream activations from a single layer via a forward hook.

    Usage:
        with ActivationCollector(model.model.layers[8]) as collector:
            model(**inputs)
            hidden_states = collector.pop()  # list of [batch, seq_len, d_model]
    """

    def __init__(self, layer: nn.Module) -> None:
        self._buffer: list[torch.Tensor] = []
        self._handle = layer.register_forward_hook(self._hook)

    def _hook(self, _module: nn.Module, _input: tuple, output: torch.Tensor | tuple) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._buffer.append(hidden.detach().cpu())

    def pop(self) -> list[torch.Tensor]:
        """Return and clear the buffer."""
        out, self._buffer = self._buffer, []
        return out

    @property
    def buffer_size(self) -> int:
        return len(self._buffer)

    def remove(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "ActivationCollector":
        return self

    def __exit__(self, *_: object) -> None:
        self.remove()
