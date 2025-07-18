import math

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from nn.activations import GELU
from nn.utils import autocast_precision

from torch.distributed import DeviceMesh
from torch.distributed.tensor import Placement, Replicate, Shard
from torch.distributed.tensor.parallel import parallelize_module

from nn.distributed.utils import get_tp_wrappers
from nn.distributed.parallel.tensor_parallel import SequenceParallel

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"] if "hidden_dim" in cfg else 4 * self.emb_dim
        self.dtype = autocast_precision(cfg["dtype"])
        self.w1 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)
        self.w2 = nn.Linear(self.hidden_dim, self.emb_dim, dtype=self.dtype)
        self.w3 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)
        # self.layers = nn.Sequential(
        #     nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        #     GELU(),
        #     nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        # )

    def forward(self, x):
        x = x.to(self.dtype)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan={
                "w1": colwise_parallel(),
                "w2": rowwise_parallel(
                    output_layouts=output_layout, use_local_output=use_local_output
                ),
                "w3": colwise_parallel(),
            },
        )


class Qwen3FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"] if "hidden_dim" in cfg else 3 * self.emb_dim
        self.dtype = autocast_precision(cfg["dtype"])
        self.w1 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)
        self.w2 = nn.Linear(self.hidden_dim, self.emb_dim, dtype=self.dtype)
        self.w3 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)

    def forward(self, x):
        x = x.to(self.dtype)
        x_fc1 = self.w1(x)
        x_fc3 = self.w3(x)
        x = F.silu(x_fc1) * x_fc3
        return self.w2(x)
    
    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        rowwise_parallel, colwise_parallel, prepare_module_input = get_tp_wrappers(
            float8_enabled=float8_enabled
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan=prepare_module_input(
                input_layouts=None if input_layout is None else (input_layout,),
                desired_input_layouts=(Replicate(),),
            ),
        )

        parallelize_module(
            module=self,
            device_mesh=tp_mesh,
            parallelize_plan={
                "w1": colwise_parallel(),
                "w2": rowwise_parallel(
                    output_layouts=output_layout, use_local_output=use_local_output
                ),
                "w3": colwise_parallel(),
            },
        )


class NormalizedFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"] if "hidden_dim" in cfg else 4 * self.emb_dim
        self.dtype = autocast_precision(cfg["dtype"])
        self.w1 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)
        self.w2 = nn.Linear(self.hidden_dim, self.emb_dim, dtype=self.dtype)
        self.w3 = nn.Linear(self.emb_dim, self.hidden_dim, dtype=self.dtype)
        # self.layers = nn.Sequential(
        #     nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
        #     GELU(),
        #     nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        # )
        self.sw_init_value = 1.0
        self.sw_init_scaling = 1.0
        self.sw1 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        self.sw3 = torch.nn.Parameter(torch.empty(self.hidden_dim, dtype=self.dtype))
        self.sqrt_d_model = math.sqrt(self.emb_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.ones_(self.sw1)
        nn.init.ones_(self.sw3)
        with torch.no_grad():
            self.sw1.mul_(self.sw_init_scaling)
            self.sw3.mul_(self.sw_init_scaling)

    def forward(self, x):
        x = x.to(self.dtype)
        sw1 = self.sw1 * ((self.sw_init_value / self.sw_init_scaling) * self.sqrt_d_model)
        sw3 = self.sw3 * (self.sw_init_value / self.sw_init_scaling)
        return self.w2(F.silu(sw1 * self.w1(x)) * (sw3 * self.w3(x)))

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the normalized FFN"
        )

    @torch.no_grad()
    def normalize_matrices(self):
        self._normalize_matrix(self.w1.weight)
        self._normalize_matrix(self.w2.weight, dim=0)
        self._normalize_matrix(self.w3.weight)

    @staticmethod
    def l2_normalize(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        return x / torch.linalg.vector_norm(x, dim=dim, keepdim=True).type_as(x)

    def _normalize_matrix(self, w: torch.Tensor, dim: int = -1):
        w.copy_(self.l2_normalize(w, dim=dim))


class nanoGPTFeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.emb_dim = cfg["emb_dim"]
        self.hidden_dim = cfg["hidden_dim"] if "hidden_dim" in cfg else 4 * self.emb_dim
        self.dtype = autocast_precision(cfg["dtype"])
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], self.hidden_dim),
            GELU(),
            nn.Linear(self.hidden_dim, cfg["emb_dim"]),
        )

    def forward(self, x):
        x = x.to(self.dtype)
        return self.layers(x)

    def apply_tp(
        self,
        tp_mesh: DeviceMesh,
        input_layout: Optional[Placement] = None,
        output_layout: Optional[Placement] = None,
        use_local_output: bool = True,
        float8_enabled: bool = False,
    ):
        del tp_mesh, input_layout, output_layout, use_local_output, float8_enabled

        raise NotImplementedError(
            "TP is not implemented yet for the nanoGPT FFN"
        )
