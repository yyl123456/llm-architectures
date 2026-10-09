import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.norm.rmsnorm import RMSNorm
from common.rope.rotary_embedding import precompute_freqs_cis, apply_rotary_emb

@dataclass
class DeepSeekV3Config:
    vocab_size: int = 129280
    hidden_size: int = 2048         # 演示/小规模配置，全尺寸为 7168
    num_hidden_layers: int = 4      # 演示层数，全尺寸为 61
    num_attention_heads: int = 16   # 全尺寸为 128
    
    # MLA 超参数
    kv_lora_rank: int = 512         # KV 潜在压缩向量维度
    q_lora_rank: int = 768          # Q 压缩维度
    qk_nope_head_dim: int = 64      # 不带 RoPE 的 QK 头维度
    qk_rope_head_dim: int = 32      # 解耦 RoPE 维度
    v_head_dim: int = 64            # Value 头维度
    
    # MoE 超参数
    moe_intermediate_size: int = 512
    n_routed_experts: int = 16      # 演示用 16 专家，全尺寸 256
    num_experts_per_tok: int = 4    # 演示用 Top-4，全尺寸 Top-8
    n_shared_experts: int = 1
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_seq_len: int = 2048


class MultiHeadLatentAttention(nn.Module):
    """
    MLA (Multi-Head Latent Attention) 最小化清晰实现
    核心特性：
    1. KV 联合压缩入低秩潜在空间 c_kv (极大幅度降低推理 KV Cache)
    2. 解耦位置编码 (Decoupled RoPE)：一部分做矩阵折叠吸收，一部分独立携带位置信息
    """
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.kv_lora_rank = config.kv_lora_rank

        # Q 压缩投影与升维
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(
            config.q_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim),
            bias=False
        )

        # KV 压缩投影与升维 (核心：KV Cache 仅需保存 c_kv 与 k_pe)
        self.wkv_a = nn.Linear(config.hidden_size, config.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_b = nn.Linear(
            config.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False
        )

        # 输出投影
        self.wo = nn.Linear(self.num_heads * self.v_head_dim, config.hidden_size, bias=False)
        self.softmax_scale = 1.0 / math.sqrt(self.qk_nope_head_dim + self.qk_rope_head_dim)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # 1. 计算 Query
        q_compressed = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_compressed).view(bsz, seqlen, self.num_heads, self.qk_nope_head_dim + self.qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        # 2. 计算 Compressed KV
        kv_compressed = self.wkv_a(x)
        c_kv, k_pe = torch.split(kv_compressed, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        # 解耦位置项的单头广播至所有头 (或者每个头共享)
        k_pe = k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        # 3. 对解耦项注入 RoPE
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, freqs_cis=freqs_cis)

        # 4. 从压缩向量恢复 Key 和 Value
        kv = self.wkv_b(c_kv).view(bsz, seqlen, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        # 5. 拼接 Q 与 K (内容部分 + 解耦位置部分)
        q_all = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)  # [bsz, heads, seqlen, total_dim]
        k_all = torch.cat([k_nope, k_pe], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        # 6. 计算 Attention Scores
        scores = torch.matmul(q_all, k_all.transpose(-2, -1)) * self.softmax_scale
        if mask is not None:
            scores = scores + mask
        attn_weights = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(attn_weights, v)  # [bsz, heads, seqlen, v_head_dim]

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class DeepSeekMoE(nn.Module):
    """
    DeepSeekMoE: 细粒度专家 + 独立共享专家
    """
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok

        # 路由门控权重
        self.router = nn.Linear(config.hidden_size, self.n_routed_experts, bias=False)

        # 细粒度路由专家群
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
            ) for _ in range(self.n_routed_experts)
        ])

        # 共享专家 (始终参与计算)
        self.shared_expert = nn.Sequential(
            nn.Linear(config.hidden_size, config.moe_intermediate_size * config.n_shared_experts, bias=False),
            nn.SiLU(),
            nn.Linear(config.moe_intermediate_size * config.n_shared_experts, config.hidden_size, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # 共享专家输出
        shared_out = self.shared_expert(x_flat)

        # 路由专家计算 (Top-K Gating)
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=-1)
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        routed_out = torch.zeros_like(x_flat)
        # 为演示清晰度采用直观 dispatch，在工程上通常使用 batched/scatter-gather 算子
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                sub_x = x_flat[mask]
                sub_weights = (topk_weights * (topk_indices == i).float()).sum(dim=-1, keepdim=True)[mask]
                routed_out[mask] += expert(sub_x) * sub_weights

        total_out = shared_out + routed_out
        return total_out.view(bsz, seqlen, hidden_dim)


class DeepSeekV3DecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MultiHeadLatentAttention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = DeepSeekMoE(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), freqs_cis=freqs_cis, mask=mask)
        out = h + self.moe(self.ffn_norm(h))
        return out


class DeepSeekV3Model(nn.Module):
    def __init__(self, config: DeepSeekV3Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepSeekV3DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(config.qk_rope_head_dim, config.max_seq_len, theta=config.rope_theta)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[:seqlen].to(x.device)

        # 因果掩码 (Causal Mask)
        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, freqs_cis=freqs_cis, mask=mask)

        x = self.norm(x)
        logits = self.head(x)
        return logits


if __name__ == "__main__":
    print("Testing DeepSeek-V3 Minimal Reference Model...")
    cfg = DeepSeekV3Config(vocab_size=1000, hidden_size=256, num_hidden_layers=2)
    model = DeepSeekV3Model(cfg)
    dummy_input = torch.randint(0, 1000, (2, 16))
    out = model(dummy_input)
    print("DeepSeek-V3 forward succeeded! Output shape:", out.shape)
    assert out.shape == (2, 16, 1000)
