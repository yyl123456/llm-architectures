# 分层 KV 压缩与长程跳跃索引机制 (Hierarchical Compression & Indexing)

> 源码锚点：`inference/model.py:Compressor`, `inference/model.py:Indexer`, `inference/model.py:Attention`

---

## 1. 为什么 40 层不是层层压缩？

在 `inference/model.py` 中有严格的硬件工程限制：
```python
is_backbone = layer_id < args.n_layers
self.is_kv_source = is_backbone and layer_id in args.kv_source_layers
if self.is_kv_source:
    self.compressor = Compressor(args, layer_id)
```
- **核心逻辑**：全网 40 层中，**仅在 Layer 2, 8, 14, 20 这 4 个动力源层才真正运行 Compressor 计算压缩！**
- 其余 36 层自身根本不压缩，而是直接复用这 4 个源层缓存好的潜在特征。

---

## 2. 核心参数表

| 参数字段 | 官方配置值 | 架构工程含义 |
|---|---|---|
| `kv_source_layer_ids` | `[2, 8, 14, 20]` | 4 个等距分布的 KV 压缩源动力层 |
| `index_source_layer_ids` | `[2, 8, 14, 20, 24, 28, 32, 36]` | 8 个长程跳跃索引源节点 |
| `index_n_heads` | 32 | 索引专用注意力头数 |
| `index_head_dim` | 128 | 索引头向量维度 |
| `index_topk` | **512** | 动态挑选的历史最优关注块数 |
| `sliding_window` | **128** | 局部纯原始未压缩滑动窗口大小 |
| `candidate_source_layer_id` | **20** | 全局候选分水岭层 (抽取 2048 候选块) |
| `compress_ratios` | `[0, 0, 2, ..., 1, ..., 0]` | 梯级平滑非对称压缩比 |

---

## 3. 双流拼接注意力机制 (Dual-Stream Concatenation)

每一层的最终 Attention 计算由两部分无缝拼接：
1. **局部高精流**：自身当前位置前 128 个 Token 的纯原始 KV（保真近期上下文细节）；
2. **长程稀疏流**：由 Indexer 选出的 512 个深层压缩潜在块（支撑百万 Token 跨度秒级跳跃命中）。
