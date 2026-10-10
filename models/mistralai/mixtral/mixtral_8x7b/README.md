# Mixtral 8x7B / 8x22B 架构卡片 (Model Architecture Card)

---

## 1. 模型概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | Mixtral 8x7B / Mixtral 8x22B |
| **开源机构** | Mistral AI |
| **发布时间** | 2023-12 (8x7B) / 2024-04 (8x22B) |
| **模型类型** | Sparse Mixture-of-Experts (稀疏混合专家) |
| **总参数量** | 46.7B (8x7B) / 141B (8x22B) |
| **单 Token 激活参数**| ~12.9B (8x7B, 8 选 2 激活) / ~39B (8x22B) |
| **上下文窗口** | 32k (8x7B) / 64k (8x22B) |
| **开源协议** | Apache 2.0 |
| **官方技术报告** | [Mixtral of Experts (arXiv:2401.04088)](https://arxiv.org/abs/2401.04088) |

---

## 2. 核心架构亮点与核心算法

### 2.1 稀疏门控专家系统 (Sparse MoE with Top-2 Router)
- **结构设计**：每个 Transformer 层的标准 FFN 被替换为 8 个独立的专家 FFN。
- **Top-2 路由**：门控网络对 8 个专家计算 Softmax 概率分布，为每个 Token 动态挑选权重最高的 2 个专家进行前向传播，并将结果加权求和：
  $$y = \sum_{i \in \text{Top2}} P_i(x) \cdot \text{Expert}_i(x)$$
- 从而在仅消耗 12.9B 计算量的开销下，获得了逼近 45B+ 稠密模型的知识容量与推理质量。

### 2.2 滑动窗口注意力 (Sliding Window Attention - SWA)
- 在 Mistral-7B 中提出并在 Mixtral 中部分继承，每个 Token 仅直接注意自身前 $W=4096$ 个 Token，跨越 $L$ 层后获得 $L \times W$ 的有效理论感受野，降低 Attention 显存复杂度。

### 2.3 分组查询注意力 (GQA)
- 32 个 Query Heads，8 个 KV Heads（4:1 比例），兼顾生成吞吐与表示能力。
