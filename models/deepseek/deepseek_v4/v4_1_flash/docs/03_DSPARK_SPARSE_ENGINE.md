# DSpark 动态稀疏引擎与高层特化 (DSpark Sparse Engine)

> 源码锚点：`inference/model.py:DSparkAttention`, `inference/model.py:DSparkBlock`

---

## 1. 核心痛点与设计背景

当网络堆叠至深层（Layer 37~39）时，全局自回归计算对高维特征图的访存带宽消耗达到峰值，成为 1M 长文本生成吞吐的主要瓶颈。

---

## 2. 核心架构参数

| 参数字段 | 官方配置值 | 架构工程含义 |
|---|---|---|
| `dspark_target_layer_ids` | `[37, 38, 39]` | DSpark 引擎汇聚特征的顶层目标层 |
| `dspark_block_size` | 5 | 稀疏块大小 |
| `dspark_markov_rank` | 256 | 马尔可夫隐空间低秩维度 |
| `dspark_n_routed_experts` | **128** | 高层专属的 128 个轻量稀疏路由专家 |
| `dspark_num_experts_per_tok`| **3** | 单 Token 仅激活 3 个 DSpark 专属专家 |
| `hc_sinkhorn_iters` | 20 | 20 轮 Sinkhorn 最优传输迭代过滤 |

---

## 3. 工作流程与收益

1. 汇聚 Layer 37, 38, 39 的深层特征（总维度 $5120 \times 3$）；
2. 经过 `main_proj` 投影降维后，交由 128 个专属的轻量稀疏专家（3 激活）前向计算；
3. 将深层长文本自回归推理对全量 KV 块的访存带宽消耗**压降 70% 以上**。
