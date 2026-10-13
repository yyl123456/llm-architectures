# Engram 外部超大词法记忆机制 (Layer 1 & 14)

> 源码锚点：`inference/engram.py`、`inference/model.py:ParallelEngramEmbedding`

---

## 1. 设计动机与背景痛点

在传统的自回归 Transformer 中，模型必须耗费宝贵的隐层浮点权重去“死记硬背”常见的固定短语、代码模板、函数签名以及领域专有名词。这不仅挤占了模型深层网络的逻辑推理容量，而且随着词表扩展，静态 Embedding 参数开销巨大。

---

## 2. 核心架构与参数规范

| 参数字段 | 官方配置值 | 架构工程含义 |
|---|---|---|
| `engram_layer_ids` | `[1, 14]` | 仅在 **Layer 1**（浅层语法）和 **Layer 14**（中层语义模式）挂载 |
| `engram_vocab_size` | **16,000,000 (1600万)** | 专属性超大外部 N-gram 词表桶大小 |
| `engram_max_ngram_size` | **4** | 最大支持 4-gram 连续 Token 短语匹配 |
| `engram_n_heads` | 8 | 8 个独立的词法投影头 |
| `engram_head_dim` | 256 | 每个头的特征维度（总投影维度 2048） |
| `engram_compressed_vocab_size` | 99,092 | 压缩后的哈希查找词表 |

---

## 3. 计算流程与源码实现

```text
[输入 Token 序列: input_ids]
             │
             ├──► NgramHashState(start_pos) ──► 快速滑动窗口计算 2-gram/3-gram/4-gram Hash IDs
             │
             ├──► 查表: ParallelEngramEmbedding[hash_ids]
             │
             ├──► 门控缩放与 LayerNorm
             │
             └──► 混入残差流: h = h + engram_features (仅在 Layer 1 和 Layer 14 执行)
```

### 多模态隔离特性
- 当处理多模态图像 Token（`image_mask`）时，视觉 Token 不参与 N-gram 计算，自动跳过 Engram 嵌入，防止图像切片破坏语言短语哈希。
