# DeepSeek-V3 架构卡片 (Model Architecture Card)

---

## 1. 模型概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | DeepSeek-V3 |
| **开源机构** | 深度求索 (DeepSeek-AI) |
| **发布时间** | 2024-12 |
| **总参数量** | 671B |
| **单 Token 激活参数** | 37B (含 1 个共享专家 + 8 个路由专家) |
| **上下文窗口** | 128k tokens |
| **开源协议** | DeepSeek License (宽松开源可商用) |
| **技术报告** | [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) |
| **官方代码库** | [deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3) |

---

## 2. 核心架构亮点与核心算法

### 2.1 Multi-Head Latent Attention (MLA)
- **核心动机**：大模型生成阶段的核心性能瓶颈是 **KV Cache 的显存占用与访存带宽**。传统的 MHA/GQA 依然需要缓存完整未压缩的 Key 和 Value 向量。
- **低秩下投影 (Low-rank Compression)**：
  - Key 与 Value 被压缩进一个共享的隐空间潜在向量 $c_t^{KV} \in \mathbb{R}^{d_c}$，压缩维度 $d_c \ll n_{heads} \times d_k$。
  - 推理时仅需缓存标量极小的潜在向量 $c_t^{KV}$，显著降低显存开销（不到标准 GQA 的 1/5）。
- **Decoupled RoPE (解耦位置编码)**：
  - 传统的 RoPE 直接乘入含位置信息的向量，无法将矩阵相乘进行结合律吸收折叠。
  - MLA 将 Key/Query 拆分为：**不含位置的吸收项**（吸收进 Projection 权重） + **含位置的解耦项 (Decoupled RoPE Key/Query)**。

### 2.2 DeepSeekMoE: 极致细粒度专家与共享专家
- **细粒度切分 (Fine-grained Segmentation)**：传统 MoE（如 Mixtral 8 选 2）专家数量少且单个专家过大。DeepSeekMoE 将每个专家切细（总共 256 个路由专家），每个 Token 动态激活 8 个路由专家。
- **共享专家隔离 (Shared Expert)**：设置 1 个固定被所有 Token 激活的共享专家，专门捕捉通用常识知识，路由专家专注于垂直细分领域知识。
- **无辅助 Loss 负载均衡 (Auxiliary-loss-free Load Balancing)**：
  - 传统的 MoE 依赖 Auxiliary Loss 强制专家均衡，但这往往会损伤模型预训练性能。
  - DeepSeek-V3 引入**动态偏置更新机制 (Dynamic Bias Correction)**：在每个 Step 监控各专家 Token 负载，根据偏差动态调整专家的路由偏置 $b_i$，无需梯度反传即实现完美负载均衡。

### 2.3 多 Token 预测目标 (Multi-Token Prediction - MTP)
- 训练时除了预测当前第 $t+1$ 个 Token 外，并行多头预测未来的 $t+2$ 个 Token。加速推理推测解码（Speculative Decoding）。

---

## 3. 架构拓扑与张量流转图

```text
               Token IDs [batch, seq_len]
                           │
                    Embedding Layer
                           │
             ┌─────────────▼─────────────┐
             │ DeepSeek-V3 Decoder Layer │  (x 61 Layers)
             │                           │
             │   RMSNorm (Pre-Norm)      │
             │         │                 │
             │   MLA (潜在多头注意力)    │
             │   - KV Down-proj -> c_kv  │  (KV Cache 只存 c_kv & k_pe)
             │   - Decoupled RoPE        │
             │         │ + 残差          │
             │   RMSNorm                 │
             │         │                 │
             │   MoE 模块                │
             │   - 1 个 Shared Expert    │
             │   - 256 选 8 路由专家     │
             │   - 动态偏置路由评分      │
             │         │ + 残差          │
             └─────────────┬─────────────┘
                           │
                     RMSNorm (Final)
                           │
                      LM Head
                           │
                 Logits [batch, seq_len, vocab]
```

---

## 4. 关键超参数表

| 参数名 | 取值 | 说明 |
|---|---|---|
| `vocab_size` | 129280 | 扩充多语言词表 |
| `hidden_size` | 7168 | 隐藏层维度 |
| `num_hidden_layers` | 61 | 模型层数 (第 1-3 层为 Dense, 后续为 MoE) |
| `num_attention_heads` | 128 | MLA 注意力头数 |
| `kv_lora_rank` ($d_c$) | 512 | KV 潜在压缩维度 (MLA 核心) |
| `q_lora_rank` | 1536 | Query 潜在压缩维度 |
| `qk_rope_head_dim` | 64 | 解耦 RoPE 维度 |
| `v_head_dim` | 128 | Value 投影单头维度 |
| `n_routed_experts` | 256 | 细粒度路由专家总数 |
| `num_experts_per_tok` | 8 | 单 Token 激活专家数 |
| `n_shared_experts` | 1 | 共享专家数 |
