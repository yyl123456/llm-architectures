# Gemma 1 / 2 架构卡片 (Model Architecture Card)

---

## 1. 模型概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | Gemma-2 (2B, 9B, 27B) |
| **开源机构** | Google DeepMind |
| **发布时间** | 2024-06 |
| **参数规格** | 2.6B, 9.2B, 27.2B (Dense 架构) |
| **上下文窗口** | 8192 tokens |
| **开源协议** | Gemma Terms of Use |
| **技术报告** | [Gemma 2: Improving Open Language Models at a Practical Size (arXiv:2408.00118)](https://arxiv.org/abs/2408.00118) |

---

## 2. 核心架构亮点与核心算法

### 2.1 交替注意力 (Alternating Local and Global Attention)
- Gemma-2 将层交替组织：一层为局部的滑动窗口注意力（Sliding Window Attention，窗口大小为 4096），紧随一层为全上下文的全局注意力（Global Attention）。在大幅降低注意力计算开销的同时保留全局长程捕获能力。

### 2.2 Logit Soft-Capping (Logit 软截断)
- 为了防止极深或极宽模型中注意力和最终 Logit 数值发散溢出，Gemma-2 引入基于 `tanh` 的 Soft-capping 约束：
  $$\text{scores} = \text{cap} \cdot \tanh\left(\frac{Q K^T}{\sqrt{d_k} \cdot \text{cap}}\right)$$
  - Attention 层的 cap 设为 50.0；
  - 最终 LM Head Logits 的 cap 设为 30.0。

### 2.3 Pre-Norm 与 Post-Norm 双重归一化 (Dual RMSNorm)
- 每一层除了输入端的 Pre-Norm 外，在 Attention/FFN 残差输出送入下一层前额外增加 Post-Norm，大幅增强极深架构（如 27B 的 46 层）在训练过程中的数值稳定性。

### 2.4 GeGLU 激活函数
- 采用近似高斯误差线性门控单元（GELU-gated Linear Unit），不同于主流 LLaMA/Mistral 采用的 SwiGLU。
