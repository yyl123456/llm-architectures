# Qwen-2.5 架构卡片 (Model Architecture Card)

---

## 1. 模型概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | Qwen-2.5 (通义千问 2.5) |
| **开源机构** | 阿里巴巴 (Alibaba Cloud) |
| **发布时间** | 2024-09 |
| **参数规格** | Dense (0.5B, 1.5B, 3B, 7B, 14B, 32B, 72B) / MoE (A14B 等) |
| **最大上下文** | 最高 128k tokens (生成最长 8k) |
| **开源协议** | Apache 2.0 (部分尺寸为 Qwen 协议) |
| **技术报告** | [Qwen2.5 Technical Report (arXiv:2412.15115)](https://arxiv.org/abs/2412.15115) |
| **官方代码库** | [QwenLM/Qwen2.5](https://github.com/QwenLM/Qwen2.5) |

---

## 2. 核心架构亮点与核心算法

### 2.1 QK-Norm 训练稳定化技术
- **背景**：随着模型规模扩大和上下文增长到 128k，Attention 内部的 Query 和 Key 内积容易在 FP16/BF16 下发生极值放大或下溢，导致 Attention 权重趋近于 One-hot 或 NaN。
- **机制**：在计算 Attention 分数之前，分别对 Query 头和 Key 头施加独立的 RMSNorm 归一化：
  $$\tilde{Q} = \text{RMSNorm}(Q), \quad \tilde{K} = \text{RMSNorm}(K)$$
  再注入 RoPE 并计算点积。该技术极大地增强了极端长文本与大规模预训练的数值稳定性。

### 2.2 全尺寸 GQA (Grouped Query Attention)
- 在所有主流规格（从 0.5B 到 72B）上均采用 GQA。例如 7B 模型采用 28 个 Query 头，4 个 KV 头（7:1 比例），将 KV Cache 压缩为原先的 1/7。

### 2.3 双重 Embedding 绑定与大词表
- 词表大小扩展至 152,064，对中文、英文、多语言代码和数学符号有极佳的 Tokenization 压缩率。
- 0.5B/1.5B/3B 等小尺寸绑定输入输出权重（Tie Word Embeddings）以节约显存参数；7B/14B/72B 则解耦以释放大容量表示能力。

### 2.4 RoPE 超参数设置
- 预训练阶段采用 $\theta = 1,000,000$ (100万) 的超大基频，结合 Dual Chunk Attention 机制无损外推 128k 上下文。

---

## 3. 架构拓扑与张量流转图

```text
               Token IDs [batch, seq_len]
                           │
                 Embedding (152064, dim)
                           │
             ┌─────────────▼─────────────┐
             │    Qwen-2.5 Decoder Layer │  (x N Layers)
             │                           │
             │   RMSNorm (Pre-Norm)      │
             │         │                 │
             │   Linear Q, K, V          │
             │         │                 │
             │   QK-Norm                 │  <── 核心创新点 (Q-RMSNorm & K-RMSNorm)
             │   - q_norm(Q), k_norm(K)  │
             │   - 注入 RoPE 旋转位置编码│
             │         │                 │
             │   GQA Scaled Dot-Product  │
             │         │ + 残差          │
             │   RMSNorm                 │
             │         │                 │
             │   SwiGLU FFN              │
             │         │ + 残差          │
             └─────────────┬─────────────┘
                           │
                     RMSNorm (Final)
                           │
                        LM Head
                           │
                 Logits [batch, seq_len, 152064]
```

---

## 4. 关键超参数对照表 (7B vs 14B vs 72B)

| 参数名 | 7B | 14B | 72B |
|---|---|---|---|
| `hidden_size` | 3584 | 5120 | 8192 |
| `num_hidden_layers` | 28 | 48 | 80 |
| `num_attention_heads` | 28 | 40 | 64 |
| `num_key_value_heads` | 4 | 8 | 8 |
| `intermediate_size` | 18944 | 13824 | 29568 |
| `rope_theta` | 1000000.0 | 1000000.0 | 1000000.0 |
| `vocab_size` | 152064 | 152064 | 152064 |
