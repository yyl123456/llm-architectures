import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from common.rope.rotary_embedding import apply_rotary_emb

class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA)
    论文: GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints (Ainslie et al., 2023)
    适用: LLaMA-2 (34B/70B), LLaMA-3 (全部尺寸), Mistral, Qwen2 等。
    特点: n_query_heads 与 n_kv_heads 成倍数关系 (n_heads // n_kv_heads)。
         当 n_kv_heads == 1 为 MQA，当 n_kv_heads == n_heads 为标准 MHA。
    """
    def __init__(
        self,
        dim: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: Optional[int] = None,
        bias: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim if head_dim is not None else dim // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=bias)
        self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=bias)
        self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=bias)
        self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=bias)

    @staticmethod
    def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """扩展 Key / Value 头数匹配 Query 头数 (batch, seqlen, n_kv_heads, head_dim) -> (batch, seqlen, n_heads, head_dim)"""
        if n_rep == 1:
            return x
        bs, slen, n_kv_heads, head_dim = x.shape
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # 注入旋转位置编码
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        # GQA 广播 KV 头
        keys = self.repeat_kv(xk, self.n_rep)
        values = self.repeat_kv(xv, self.n_rep)

        # 转置为 [bsz, n_heads, seqlen, head_dim]
        xq = xq.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Scaled Dot-Product Attention
        scores = torch.matmul(xq, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # [bsz, n_heads, seqlen, head_dim]

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
