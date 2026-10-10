import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.norm.rmsnorm import RMSNorm, SwiGLU
from common.rope.rotary_embedding import precompute_freqs_cis, apply_rotary_emb

@dataclass
class Qwen2_5Config:
    vocab_size: int = 152064
    hidden_size: int = 3584
    num_hidden_layers: int = 28
    num_attention_heads: int = 28
    num_key_value_heads: int = 4
    intermediate_size: int = 18944
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_seq_len: int = 32768

class Qwen2_5Attention(nn.Module):
    """
    Qwen-2.5 核心注意力机制：带 QK-Norm 的 Grouped-Query Attention
    """
    def __init__(self, config: Qwen2_5Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # QK-Norm: 对每个 Head 的 Query 和 Key 向量施加 RMSNorm 归一化
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(bsz, seqlen, self.num_kv_heads, self.head_dim)

        # 核心：QK-Norm 处理
        q = self.q_norm(q)
        k = self.k_norm(k)

        # 注入 RoPE
        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        # GQA 广播 KV
        if self.num_kv_groups > 1:
            k = (
                k[:, :, :, None, :]
                .expand(bsz, seqlen, self.num_kv_heads, self.num_kv_groups, self.head_dim)
                .reshape(bsz, seqlen, self.num_heads, self.head_dim)
            )
            v = (
                v[:, :, :, None, :]
                .expand(bsz, seqlen, self.num_kv_heads, self.num_kv_groups, self.head_dim)
                .reshape(bsz, seqlen, self.num_heads, self.head_dim)
            )

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, v)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(output)


class Qwen2_5DecoderLayer(nn.Module):
    def __init__(self, config: Qwen2_5Config):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen2_5Attention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(
            hidden_dim=config.hidden_size,
            intermediate_dim=config.intermediate_size,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        h = x + self.self_attn(self.input_layernorm(x), freqs_cis=freqs_cis, mask=mask)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out


class Qwen2_5Model(nn.Module):
    def __init__(self, config: Qwen2_5Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            Qwen2_5DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        head_dim = config.hidden_size // config.num_attention_heads
        self.freqs_cis = precompute_freqs_cis(head_dim, config.max_seq_len, theta=config.rope_theta)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        x = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:seqlen].to(x.device)

        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, freqs_cis=freqs_cis, mask=mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


if __name__ == "__main__":
    print("Testing Qwen-2.5 Reference Implementation...")
    cfg = Qwen2_5Config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=512,
        max_seq_len=512
    )
    model = Qwen2_5Model(cfg)
    dummy_input = torch.randint(0, 1000, (2, 8))
    out = model(dummy_input)
    print("Qwen-2.5 Forward Output shape:", out.shape)
    assert out.shape == (2, 8, 1000)
