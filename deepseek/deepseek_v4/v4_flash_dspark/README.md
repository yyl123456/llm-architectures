# DeepSeek-V4-Flash-DSpark 架构预研卡片

---

## 1. 基本概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | DeepSeek-V4-Flash-DSpark |
| **开源机构** | 深度求索 (DeepSeek-AI) |
| **发布时间** | **2026-07-04 (开源适配 2026-06/07)** |
| **核心机制** | 动态稀疏注意力与马尔可夫激活 (DSpark / Markov Rank 256) |
| **代码/权重库** | [deepseek-ai/DeepSeek-V4-Flash-DSpark](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark) |

---

## 2. 核心架构亮点与算法突破

1. **DSpark 动态稀疏推理引擎**：
   - 在高层（Layer 37~39）采用块大小为 5 的 DSpark 稀疏机制，激活专用的 128 个二级路由专家（激活 3 专家）；
   - 大幅减少自回归生成阶段对全量 KV 块的访存带宽压力。
2. **沉淀研究方向**：
   - DSpark Sinkhorn 稀疏最优传输迭代算法；
   - 噪声 Token 与推测解码的高速验证流。
