# DiffusionGemma-26B 架构预研卡片

---

## 1. 基本概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | DiffusionGemma-26B-A4B-it |
| **开源机构** | Google DeepMind |
| **发布时间** | **2026-06-09** |
| **架构形态** | 稀疏混合专家扩散变换器 (Sparse MoE Diffusion Transformer) |
| **总参数量** | 26B (激活参数仅 4B) |
| **官方代码库** | [google/diffusiongemma-26B-A4B-it](https://huggingface.co/google) |

---

## 2. 核心架构亮点与算法突破

1. **首个大参数量 MoE 扩散生成架构**：
   - 传统扩散模型多为纯 Dense DiT，Google 首次将 Gemma 的 MoE 稀疏路由引入去噪 Transformer；
   - 激活仅 4B 参数，即可展现 26B 级别的深厚概念理解与艺术渲染。
