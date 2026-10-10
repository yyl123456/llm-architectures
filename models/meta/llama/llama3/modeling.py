import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.norm.rmsnorm import RMSNorm, SwiGLU
from common.attention.gqa import GroupedQueryAttention
from common.rope.rotary_embedding import precompute_freqs_cis

@dataclass
class LLaMA3Config:
    vocab_size: int = 128256
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 14336
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    max_seq_len: int = 8192

class LLaMA3DecoderLayer(nn.Module):
    def __init__(self, config: LLaMA3Config):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            n_kv_heads=config.num_key_value_heads,
            head_dim=config.hidden_size // config.num_attention_heads,
            bias=False
        )
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.feed_forward = SwiGLU(
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
        # Pre-LN 残差连接结构
        h = x + self.attention(self.attention_norm(x), freqs_cis=freqs_cis, mask=mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class LLaMA3Model(nn.Module):
    """
    LLaMA-3 独立最小化模型参考实现
    """
    def __init__(self, config: LLaMA3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            LLaMA3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        head_dim = config.hidden_size // config.num_attention_heads
        self.freqs_cis = precompute_freqs_cis(head_dim, config.max_seq_len, theta=config.rope_theta)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        x = self.embed_tokens(input_ids)
        freqs_cis = self.freqs_cis[:seqlen].to(x.device)

        # Causal Attention Mask
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, freqs_cis=freqs_cis, mask=mask)

        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

if __name__ == "__main__":
    print("Testing LLaMA-3 Reference Implementation...")
    cfg = LLaMA3Config(
        vocab_size=1000,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        intermediate_size=512,
        max_seq_len=512
    )
    model = LLaMA3Model(cfg)
    dummy_input = torch.randint(0, 1000, (2, 8))
    out = model(dummy_input)
    print("LLaMA-3 Forward Output shape:", out.shape)
    assert out.shape == (2, 8, 1000)
