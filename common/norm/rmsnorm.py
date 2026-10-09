import math
import torch
import torch.nn as nn
from typing import Optional, Tuple

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    论文: Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)
    被广泛用于 LLaMA, Mistral, Gemma, DeepSeek, Qwen 等现代大模型。
    省去了传统 LayerNorm 的均值计算与偏移偏置 (bias)，减少计算并保持稳定性。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # RMS = sqrt(mean(x^2) + eps)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SwiGLU(nn.Module):
    """
    SwiGLU: Gated Linear Unit with Swish/SiLU activation
    论文: GLU Variants Improve Transformer (Noam Shazeer, 2020)
    公式: SwiGLU(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down
    """
    def __init__(self, hidden_dim: int, intermediate_dim: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=bias)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch, seq_len, intermediate_dim)
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
