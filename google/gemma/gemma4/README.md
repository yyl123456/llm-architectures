# Gemma 4 架构预研卡片

---

## 1. 基本概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | Gemma 4 12B / 31B |
| **开源机构** | Google DeepMind |
| **发布时间** | **2026-06-03** |
| **任务形态** | 原生音视频文字一体多模态 (Omnimodal) |
| **代码/权重库** | [google/gemma-4-12B-it](https://huggingface.co/google) |

---

## 2. 核心架构亮点与算法突破

1. **统一端到端多模态感知**：
   - 原生集成音频、视觉与文本 Token，统一解码。
2. **极深 Transformer 稳定性设计**：
   - 继承 Gemma 家族 Dual RMSNorm 与 Logit Soft-capping 机制，结合交替滑动窗口注意力，兼顾长程与局部感受野。
