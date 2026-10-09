# Qwen3.8-Flash-Next 架构预研卡片

---

## 1. 基本概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | Qwen3.8-Flash-Next (FP8 / BF16) |
| **开源机构** | 阿里巴巴 (QwenLM) |
| **发布时间** | **2026-08-24** |
| **网络层数** | 48 层 |
| **隐藏层维度** | 2560 (24 Q-heads, 2 KV-heads) |
| **MoE 专家架构** | **512 个路由专家，每个 Token 激活 10 个专家** (Top-10 Router) |
| **专家切分大小** | `moe_intermediate_size`: 640 |
| **上下文窗口** | **262k tokens (支持 1M 扩展)** |
| **开源协议** | Apache 2.0 |
| **代码/权重库** | [Qwen/Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) |

---

## 2. 核心架构亮点与算法突破

1. **512 超细粒度混合专家系统 (512 Ultra Fine-grained MoE)**：
   - 相比上一代的粗粒度 MoE，阿里通义在 Flash-Next 中将专家数量直接推高至 **512 个**，单 Token 激活 10 个专家；
   - 保证了单步生成浮点运算量（FLOPs）极其轻量，而知识容量大幅超越同级模型。
2. **极速首字响应与超高吞吐 (Instant First-Token Latency)**：
   - 专为复杂 Agentic 反思循环与即时代码生成设计，吞吐性能在消费级单卡上保持领先。
3. **沉淀研究方向**：
   - 512 路由专家的 Top-10 Gating 分发与负载均衡机制；
   - Qwen4 实验性架构（`qwen4_exp_text`）的前向流转机制。
