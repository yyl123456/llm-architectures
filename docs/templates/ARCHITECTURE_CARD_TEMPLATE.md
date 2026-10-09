# 架构卡片标准模板 (Model Architecture Card Template)

> 目录路径示范: `meta/llama/llama3/README.md` 或 `deepseek/deepseek_v3/v3/README.md`

---

## 1. 模型概览

| 属性 | 参数 / 说明 |
|---|---|
| **模型名称** | `<Vendor>-<Series>-<Version>` (如 DeepSeek-V3) |
| **开源机构** | `<Company / Org>` (如 DeepSeek-AI) |
| **发布时间** | `YYYY-MM` (如 2024-12) |
| **核心规模** | 总参数: `XXB` / 激活参数: `XXB` |
| **最大上下文** | `XXk tokens` (如 128k) |
| **开源协议** | `MIT / Apache-2.0 / Llama Community License` |
| **官方论文 / 报告** | [Link to Paper](https://arxiv.org/...) |
| **官方开源仓库** | [GitHub Link](https://github.com/...) |

---

## 2. 核心架构亮点与数学原理

### 2.1 注意力机制 (Attention)
- **类型**：[MHA / GQA / MQA / MLA / SWA]
- **原理详解**：
  - 阐述其 KV 压缩方式、头数配比、投影方式。
  - 公式简述。

### 2.2 位置编码 (Positional Encoding)
- **类型**：[RoPE / YaRN / ALiBi / SuScaledRoPE]
- **基频与缩放参数**：`rope_theta = ...`
- **长文本扩展机制**：解释如何支撑 32k/128k/1M 上下文。

### 2.3 归一化与激活函数 (Norm & Activation)
- **归一化方案**：[Pre-RMSNorm / Post-RMSNorm / QK-Norm / Dual-Norm]
- **激活函数**：[SwiGLU / GeGLU / SiLU]
- **数值稳定性技巧**：如 QK-Norm、Epsilon 选取、Logit Capping。

### 2.4 FFN / 专家系统 (MoE)
- **拓扑结构**：[Dense / Fine-grained MoE / Top-K Router / Shared Expert]
- **路由策略**：路由算法、无辅助 Loss 平衡机制等。

---

## 3. 架构拓扑与张量流转图

```text
[Input Token IDs: (batch, seq_len)]
                │
         [Embedding Layer]
                │
       ┌────────▼────────┐
       │ Transformer     │ <─── 循环 N 层
       │ Decoder Layer   │
       │                 │
       │  Input Norm     │ (RMSNorm)
       │       │         │
       │  Self-Attention │ (GQA / MLA + RoPE)
       │       │ + 残差  │
       │  Post-Attn Norm │ (RMSNorm)
       │       │         │
       │  FFN / MoE      │ (SwiGLU / MoE Router)
       │       │ + 残差  │
       └────────┬────────┘
                ▼
        [Final RMSNorm]
                │
        [LM Head (Linear)]
                │
  [Logits: (batch, seq_len, vocab_size)]
```

---

## 4. 核心超参数配置表

```json
{
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 128256,
  "rms_norm_eps": 1e-05,
  "rope_theta": 500000.0
}
```

---

## 5. 相比前序版本的演进差异 (Delta Analysis)

1. **改进点 1**：...
2. **改进点 2**：...
3. **动机与收益**：...
