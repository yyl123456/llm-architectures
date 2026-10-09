import torch
import torch.nn as nn
from typing import Tuple

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """
    预计算旋转位置编码 (RoPE) 的频率复数项或 cos/sin。
    论文: RoFormer: Enhanced Transformer with Rotary Position Embedding (Su et al.)
    dim: 必须是偶数，代表 head_dim
    end: 最大预计算序列长度
    theta: 基频底数，如 10000.0 (LLaMA-1/2), 500000.0 (LLaMA-3), 1000000.0 (Qwen-2)
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), f"freqs_cis shape {freqs_cis.shape} vs x {(x.shape[1], x.shape[-1])}"
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将 RoPE 应用于查询 (Query) 和键 (Key) 张量。
    xq: [batch_size, seq_len, n_heads, head_dim]
    xk: [batch_size, seq_len, n_kv_heads, head_dim]
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
