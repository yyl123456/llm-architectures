# Nemotron 3.5 Lightning 架构预研卡片

---

## 1. 基本概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | Nemotron 3.5 Lightning / Nemotron 3 Ultra |
| **开源机构** | NVIDIA |
| **发布时间** | **2026-08-11 (Lightning) / 2026-06-04 (Ultra)** |
| **开源协议** | NVIDIA Open Model License |
| **评测表现** | 专为 NVIDIA 硬件全栈与 TensorRT-LLM 优化，代码/推理高吞吐基准卓越 |

---

## 2. 核心架构亮点与算法突破

1. **硬件感知网络架构设计 (Hardware-Aware Transformer)**：
   - 针对 GPU Tensor Core 矩阵乘法单元对齐头维度与 MLP 隐藏层宽度；
   - 原生支持 FP4 / FP8 高吞吐量化推理。
