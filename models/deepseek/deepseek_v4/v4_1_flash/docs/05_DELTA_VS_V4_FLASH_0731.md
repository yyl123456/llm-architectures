# V4.1-Flash vs V4-Flash-0731 小版本精准演进差分

> 官方配置对齐源：`deepseek-ai/DeepSeek-V4-Flash-0731` vs `deepseek-ai/DeepSeek-V4.1-Flash`

---

## 1. 紧邻前一小版本详细参数对齐矩阵

| 架构参数字段 | V4-Flash-0731 (2026-07) | V4.1-Flash (2026-09) | 演进成因与实际收益 |
|---|---|---|---|
| `model_type` | `deepseek_v4` | **`deepseek_v41_text`** | 核心建模类正式重构升级 |
| `hidden_size` | 4096 | **5120 (+25%)** | 拓宽通道宽度，强化代码数理表征 |
| `num_hidden_layers` | 43 | **40 (-3层)** | 深度收敛以换取更宽特征，控制 Kernel 启动时延 |
| `n_routed_experts` | 256 | **384 (+50%)** | 专家切分更细，路由空间大幅扩大 |
| `moe_intermediate_size` | 2048 | **2304** | 单专家容量适度提升，弥补层数减少 |
| `q_lora_rank` | 1024 | **1280** | 随 5120 维度等比例放大 Query 潜在表达能力 |
| `num_nextn_predict_layers` | 1 | **3 (Next-3-Predict)** | 投机解码层翻倍，单卡吐字吞吐提升 2.5x~3x |
| `dspark_n_routed_experts` | 未特化 | **128 (Top-3)** | 在高层（Layer 37~39）增设专用稀疏路由专家群 |
| `engram_layer_ids` | 未启用 | **`[1, 14]`** | 外挂 16M 词表 N-gram 嵌入表，强化固定短语与代码记忆 |
| `compress_ratios` | `[0, 0, 4, 128, ...]` | **`[0, 0, 2, ..., 1, ..., 0]`** | 消除极端 128 压缩带来的局部信息断层 |
