# LLaMA-3 / LLaMA-3.1 架构卡片 (Model Architecture Card)

---

## 1. 模型概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | LLaMA-3 / LLaMA-3.1 (8B, 70B, 405B) |
| **开源机构** | Meta AI |
| **发布时间** | 2024-04 (LLaMA-3) / 2024-07 (LLaMA-3.1) |
| **参数规格** | 8B, 70B, 405B (Dense 架构) |
| **最大上下文** | 8k (LLaMA-3) -> 128k tokens (LLaMA-3.1) |
| **开源协议** | Llama 3 Community License |
| **技术报告** | [The Llama 3 Herd of Models (arXiv:2407.21783)](https://arxiv.org/abs/2407.21783) |
| **官方代码库** | [meta-llama/llama3](https://github.com/meta-llama/llama3) |

---

## 2. 核心架构亮点与核心算法

### 2.1 全尺寸分组查询注意力 (Grouped-Query Attention - GQA)
- **从 LLaMA-2 到 LLaMA-3 的演进**：
  - LLaMA-2 仅在 34B 和 70B 采用了 GQA，而在 7B/13B 上依旧使用标准多头注意力（MHA）。
  - LLaMA-3/3.1 在所有尺寸（包括 8B）上全面标配 GQA（8 个 KV Heads），显著减少推理时的 KV Cache 显存消耗，提高并发批处理吞吐。

### 2.2 扩展词表与 BPE 标量
- 采用 Tiktoken BPE 分词器，词表大小由 LLaMA-2 的 32,000 激增至 **128,256**。
- 大词表大幅提升了文本与代码压缩率，每个 Token 代表更多信息量，但增加了 Embedding 与 LM Head 的显存与计算开销。

### 2.3 RoPE Base 缩放与长文本支持 (128k)
- **RoPE Theta 调整**：
  - LLaMA-2: $\theta = 10,000$
  - LLaMA-3: $\theta = 500,000$ (提升 50 倍以支持 8k 长度)
  - LLaMA-3.1: 进一步结合 LLaMA-3.1 RoPE Scaling 方案，将上下文扩展至 **128k**。

### 2.4 Pre-RMSNorm + SwiGLU FFN
- 采用 Pre-RMSNorm 保障深层网络前向与梯度流动的数值稳定性。
- FFN 采用带门控的 SwiGLU 激活，其中隐藏层扩展比一般设为 $\frac{8}{3} \times d$，并取 256 的倍数对齐。

---

## 3. 架构拓扑与张量流转图

```text
               Token IDs [batch, seq_len]
                           │
                 Embedding (128256, 4096)
                           │
             ┌─────────────▼─────────────┐
             │    LLaMA-3 Decoder Layer  │  (x 32 Layers for 8B)
             │                           │
             │   RMSNorm (Pre-Norm)      │
             │         │                 │
             │   GQA (分组查询注意力)    │  (32 Q-heads, 8 KV-heads)
             │   - 注入 RoPE 旋转位置编码│
             │         │ + 残差          │
             │   RMSNorm                 │
             │         │                 │
             │   SwiGLU FFN              │
             │   (gate_proj * up_proj)   │
             │         │ + 残差          │
             └─────────────┬─────────────┘
                           │
                     RMSNorm (Final)
                           │
                 LM Head (Linear -> 128256)
                           │
                 Logits [batch, seq_len, 128256]
```

---

## 4. 关键超参数对照表 (8B vs 70B vs 405B)

| 参数名 | 8B | 70B | 405B |
|---|---|---|---|
| `dim` (hidden_size) | 4096 | 8192 | 16384 |
| `n_layers` | 32 | 80 | 126 |
| `n_heads` | 32 | 64 | 128 |
| `n_kv_heads` | 8 | 8 | 8 |
| `multiple_of` | 1024 | 4096 | 4096 |
| `ffn_dim_multiplier` | 1.3 | 1.3 | 1.2 |
| `norm_eps` | 1e-5 | 1e-5 | 1e-5 |
| `rope_theta` | 500000.0 | 500000.0 | 500000.0 |
| `vocab_size` | 128256 | 128256 | 128256 |
