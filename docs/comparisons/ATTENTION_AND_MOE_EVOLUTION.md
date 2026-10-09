# 大模型核心架构与算法演进对比 (Architecture Comparisons)

本文档对主流开源大模型家族（LLaMA, DeepSeek, Qwen, Mistral, Gemma, Phi）进行横向对比，揭示现代 LLM 在计算效率、显存开销、长文本与可扩展性方面的架构设计权衡。

---

## 1. 核心算子演进全景

### 1.1 注意力机制演进 (Attention Evolution)

```
Multi-Head Attention (MHA)
       │ [LLaMA-1 / Baichuan-1]
       ▼
Grouped-Query Attention (GQA)  ───► 成为工业界 Dense 事实标准 (LLaMA-3, Qwen-2.5, Mistral)
       │ (KV Heads 显著减少，例如 8:1 或 7:1)
       │
       ├─────────────────────────────────────────┐
       ▼                                         ▼
Sliding Window Attention (SWA)           Multi-Head Latent Attention (MLA)
[Mistral-7B / Gemma-2 交替机制]           [DeepSeek-V2 / DeepSeek-V3]
- 感受野局部化，降低显存                   - 低秩潜在空间投影 (c_kv)
                                         - KV Cache 压缩为原来 1/5 甚至更低
                                         - Decoupled RoPE 解耦位置编码
```

### 1.2 稀疏专家 (MoE) 演进

1. **经典稀疏 MoE (如 Mixtral 8x7B)**：
   - 8 个大专家，每个 Token 路由至 Top-2 专家。
   - 路由粗粒度，缺乏通用知识沉淀。
2. **细粒度与共享专家 (如 DeepSeekMoE / DeepSeek-V3)**：
   - 256 个细粒度路由专家，每个 Token 动态激活 8 个；
   - 1 个固定激活的共享专家 (Shared Expert)，固定沉淀常识知识；
   - 路由偏置校准 (Dynamic Bias Correction) 替代辅助损失，消除对模型主目标的负面干扰。

### 1.3 归一化与训练稳定性 (Norm Stability)

| 方案 | 机制 | 解决痛点 | 典型模型 |
|---|---|---|---|
| **Pre-RMSNorm** | 仅保留方差均方根缩放，去除均值与 Bias | 计算更轻，梯度反传更顺畅 | LLaMA, DeepSeek, Qwen |
| **QK-Norm** | 分别对 Query 与 Key 施加 RMSNorm | 防止 Attention Logits 随深度和长度爆炸 | Qwen-2.5 |
| **Dual RMSNorm** | Pre-Norm + 残差后 Post-Norm | 超深层网络（如 46 层以上）数值稳定性 | Gemma-2 |
| **Logit Soft-capping** | 经过 tanh 软约束（cap=50/30） | 防止极值截断与概率分布退化 | Gemma-2 |

---

## 2. 经典模型超参数对比矩阵

| 特性 | LLaMA-3.1 (8B) | DeepSeek-V3 (671B) | Qwen-2.5 (7B) | Mixtral (8x7B) | Gemma-2 (9B) |
|---|---|---|---|---|---|
| **总参数量** | 8.03B | 671B | 7.61B | 46.7B | 9.24B |
| **激活参数量** | 8.03B | ~37B | 7.61B | ~12.9B | 9.24B |
| **隐藏层维度** | 4096 | 7168 | 3584 | 4096 | 3584 |
| **网络层数** | 32 | 61 | 28 | 32 | 42 |
| **Query 头数** | 32 | 128 | 28 | 32 | 16 |
| **KV 头数** | 8 (GQA) | MLA (低秩) | 4 (GQA) | 8 (GQA) | 8 (GQA) |
| **词表大小** | 128,256 | 129,280 | 152,064 | 32,000 | 256,000 |
| **激活函数** | SwiGLU | SwiGLU | SwiGLU | SwiGLU | GeGLU |
| **最大上下文** | 128k | 128k | 128k | 32k | 8k |
| **RoPE $\theta$** | 500,000 | 10,000 | 1,000,000 | 1,000,000 | 10,000 |
