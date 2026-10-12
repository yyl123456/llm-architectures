import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.norm.rmsnorm import RMSNorm
from common.rope.rotary_embedding import precompute_freqs_cis, apply_rotary_emb

@dataclass
class DeepSeekV41Config:
    vocab_size: int = 129280
    hidden_size: int = 2048              # 教学演示维度，全尺寸为 5120
    num_hidden_layers: int = 4           # 演示层数，全尺寸为 40
    num_attention_heads: int = 16        # 全尺寸为 64
    head_dim: int = 128                  # 全尺寸为 512
    qk_rope_head_dim: int = 64
    
    # MLA 动态与低秩参数
    q_lora_rank: int = 512               # 全尺寸 1280
    kv_lora_rank: int = 512
    o_lora_rank: int = 256               # 输出低秩投影 (全尺寸 1024)
    o_groups: int = 4                    # 输出分组 (全尺寸 8)
    
    # 384 细粒度专家
    n_routed_experts: int = 16           # 演示用 16 专家，全尺寸 384
    num_experts_per_tok: int = 4         # 全尺寸 6
    n_shared_experts: int = 1
    moe_intermediate_size: int = 512     # 全尺寸 2304
    
    # 稳定性与控制
    rms_norm_eps: float = 1e-6
    swiglu_limit: float = 10.0           # 激活截断
    num_nextn_predict_layers: int = 2    # MTP 预测层 (全尺寸 3)
    max_position_embeddings: int = 4096  # 全尺寸 1048576 (1M)


class OutputLoRAProjection(nn.Module):
    """
    V4.1 特有的低秩分组输出投影层 (Output-LoRA):
    针对大模型输出维度过大造成的显存带宽瓶颈，将多头注意力输出先经由低秩投影再升维。
    """
    def __init__(self, in_features: int, out_features: int, rank: int, groups: int):
        super().__init__()
        self.groups = groups
        self.down_proj = nn.Linear(in_features, rank, bias=False)
        self.up_proj = nn.Linear(rank, out_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up_proj(self.down_proj(x))


class SqrtSoftplusRouter(nn.Module):
    """
    DeepSeek-V4.1 特有的 SqrtSoftplus 门控路由：
    s_i = sqrt(softplus(x * W_r))
    相比标准 softmax 避免了极端权重垄断，保持 384 个专家的激活多样性。
    """
    def __init__(self, hidden_size: int, n_experts: int, top_k: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, n_experts, bias=False)
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        # SqrtSoftplus 激活
        scores = torch.sqrt(F.softplus(logits))
        topk_weights, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        # 归一化选中的 top-k 概率
        topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        return topk_weights, topk_indices


class DeepSeekV41MoE(nn.Module):
    """
    DeepSeek-V4.1 细粒度专家系统：384 选 6 路由专家 + 1 共享专家
    """
    def __init__(self, config: DeepSeekV41Config):
        super().__init__()
        self.config = config
        self.router = SqrtSoftplusRouter(config.hidden_size, config.n_routed_experts, config.num_experts_per_tok)

        # 路由专家群
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size, config.moe_intermediate_size, bias=False),
                nn.SiLU(),
                nn.Linear(config.moe_intermediate_size, config.hidden_size, bias=False)
            ) for _ in range(config.n_routed_experts)
        ])

        # 常驻共享专家
        self.shared_expert = nn.Sequential(
            nn.Linear(config.hidden_size, config.moe_intermediate_size * config.n_shared_experts, bias=False),
            nn.SiLU(),
            nn.Linear(config.moe_intermediate_size * config.n_shared_experts, config.hidden_size, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, hidden_dim = x.shape
        x_flat = x.view(-1, hidden_dim)

        # 1. 共享专家常驻前向
        shared_out = self.shared_expert(x_flat)

        # 2. 路由专家 Top-K 前向
        topk_weights, topk_indices = self.router(x_flat)
        routed_out = torch.zeros_like(x_flat)

        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                sub_x = x_flat[mask]
                sub_w = (topk_weights * (topk_indices == i).float()).sum(dim=-1, keepdim=True)[mask]
                routed_out[mask] += expert(sub_x) * sub_w

        out = shared_out + routed_out
        return out.view(bsz, seqlen, hidden_dim)


class DeepSeekV41Attention(nn.Module):
    """
    DeepSeek-V4.1 潜在多头注意力 (带 Output-LoRA 与 Q-LoRA)
    """
    def __init__(self, config: DeepSeekV41Config):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim

        # Q 压缩与解绑
        self.wq_a = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, self.num_heads * (self.head_dim + self.qk_rope_head_dim), bias=False)

        # KV 压缩
        self.wkv_a = nn.Linear(config.hidden_size, config.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_b = nn.Linear(config.kv_lora_rank, self.num_heads * (self.head_dim * 2), bias=False)

        # 输出分组低秩投影
        self.o_proj = OutputLoRAProjection(
            in_features=self.num_heads * self.head_dim,
            out_features=config.hidden_size,
            rank=config.o_lora_rank,
            groups=config.o_groups
        )
        self.scale = 1.0 / math.sqrt(self.head_dim + self.qk_rope_head_dim)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # 计算 Q
        q_comp = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_comp).view(bsz, seqlen, self.num_heads, self.head_dim + self.qk_rope_head_dim)
        q_nope, q_pe = torch.split(q, [self.head_dim, self.qk_rope_head_dim], dim=-1)

        # 计算 KV
        kv_comp = self.wkv_a(x)
        c_kv, k_pe = torch.split(kv_comp, [self.config.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_pe = k_pe.unsqueeze(2).expand(-1, -1, self.num_heads, -1)

        # 注入位置编码
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, freqs_cis=freqs_cis)

        # 从潜在特征升维
        kv = self.wkv_b(c_kv).view(bsz, seqlen, self.num_heads, self.head_dim * 2)
        k_nope, v = torch.split(kv, [self.head_dim, self.head_dim], dim=-1)

        q_all = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)
        k_all = torch.cat([k_nope, k_pe], dim=-1).transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q_all, k_all.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn_probs = F.softmax(scores.float(), dim=-1).type_as(q)
        context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.o_proj(context)


class DeepSeekV41DecoderLayer(nn.Module):
    def __init__(self, config: DeepSeekV41Config):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = DeepSeekV41Attention(config)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.moe = DeepSeekV41MoE(config)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), freqs_cis=freqs_cis, mask=mask)
        out = h + self.moe(self.ffn_norm(h))
        return out


class DeepSeekV41Model(nn.Module):
    """
    DeepSeek-V4.1-Flash 最小化独立参考模型实现
    包含 Next-Token-Prediction 以及原生 MTP (Next-N-Predict) 机制
    """
    def __init__(self, config: DeepSeekV41Config):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            DeepSeekV41DecoderLayer(config) for _ in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # MTP 推测预测头
        self.mtp_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_nextn_predict_layers)
        ])
        self.freqs_cis = precompute_freqs_cis(config.qk_rope_head_dim, config.max_position_embeddings)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        bsz, seqlen = input_ids.shape
        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[:seqlen].to(x.device)

        mask = torch.full((seqlen, seqlen), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            x = layer(x, freqs_cis=freqs_cis, mask=mask)

        hidden_states = self.norm(x)
        logits = self.lm_head(hidden_states)

        # 联合输出未来连续 Token 的推测 logits
        mtp_logits = [head(hidden_states) for head in self.mtp_heads]
        return logits, mtp_logits


if __name__ == "__main__":
    print("Testing DeepSeek-V4.1-Flash Standalone Implementation...")
    cfg = DeepSeekV41Config(vocab_size=1000, hidden_size=256, num_hidden_layers=2)
    model = DeepSeekV41Model(cfg)
    dummy_input = torch.randint(0, 1000, (2, 16))
    main_logits, mtp_logits = model(dummy_input)
    print("Main Logits Shape:", main_logits.shape)
    print("MTP Heads Count:", len(mtp_logits), "Shape:", mtp_logits[0].shape)
    assert main_logits.shape == (2, 16, 1000)
    assert len(mtp_logits) == cfg.num_nextn_predict_layers
    print("DeepSeek-V4.1-Flash Forward Test Passed Successfully!")
