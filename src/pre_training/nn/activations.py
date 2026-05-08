import torch.nn as nn
import torch.nn.functional as F


class SiLU(nn.Module):
    def forward(self, x):
        return F.silu(x)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)
