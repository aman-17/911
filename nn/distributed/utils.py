import torch

from torch.distributed.tensor import DTensor



def get_local_tensor(x: torch.Tensor) -> torch.Tensor:
    if isinstance(x, DTensor):
        x = x.to_local()
        # An `AsyncCollectiveTensor` might be returned, which means the local tensor is not ready
        # yet (i.e. communication is not finished). In this case we need to call `.wait()`
        # to wait the local tensor to be ready.
        if hasattr(x, "wait"):
            return x.wait()  # type: ignore
        else:
            return x
    else:
        return x

