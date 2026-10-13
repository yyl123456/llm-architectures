# DeepSeek-V4.1-Flash 架构主卡片 (Architecture Hub)

> 官方代码库：[deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)  
> 规范参考：Sebastian Raschka Architecture Gallery & Frontier ML Open Model Index

---

## 1. 基础概览 (Profile)

| 核心属性 | 官方参数 / 指标 |
|---|---|
| **模型名称** | DeepSeek-V4.1-Flash (Base & Instruct) |
| **开源机构** | 深度求索 (DeepSeek-AI) |
| **发布时间** | **2026-09-10** (Apache 2.0 / MIT 开源) |
| **主干参数规模** | **552B 总参数** / **8B ~ 16B 单 Token 动态激活** |
| **上下文窗口** | **1,048,576 tokens (1M 极限长文本)** |
| **核心评测登顶** | Codeforces **3471** / Terminal-Bench **90.6%** / DeepSWE **74.2%** |

---

## 2. 全景架构流转图 (Raschka Gallery 视觉呈现)

![DeepSeek-V4.1-Flash Architecture](<../../../../assets/architectures/deepseek-v4-1-flash.webp>)

---

## 3. 40 层深度异构功能分工全景 (Layer Specialization)

DeepSeek-V4.1 并非均一结构，40 层 Decoder（Layer 0 ~ 39）加 3 层 MTP（Layer 40 ~ 42）按功能严格划分为 7 大类别：

```text
Layer Index:
[0] ──────────────────── 基础纯原始窗口层 (Raw Window KV, 压缩率 0)
[1] ──────────────────── 🌟 Engram 词法记忆注入层 1 (16M 词表 N-gram Hash)
[2] ──────────────────── 🚀 KV 压缩动力源 1 + 🔍 稀疏检索索引源 1 (Compressor + Indexer)
[3 ~ 7] ──────────────── 梯级高速压缩层 (compress_ratio = 2, 复用 Layer 2 压缩特征)
[8] ──────────────────── 🚀 KV 压缩动力源 2 + 🔍 稀疏检索索引源 2
[9 ~ 13] ─────────────── 梯级高速压缩层 (compress_ratio = 2)
[14] ─────────────────── 🌟 Engram 词法记忆注入层 2 + 🚀 KV 压缩动力源 3 + 🔍 索引源 3
[15 ~ 19] ────────────── 梯级高速压缩层 (compress_ratio = 2)
[20] ─────────────────── 🎯 全局分水岭候选源 (Candidate Source Layer) + 🚀 KV 压缩源 4 + 🔍 索引源 4
[21 ~ 23] ────────────── 中位平滑层 (compress_ratio = 1)
[24, 28, 32, 36] ─────── 🔍 高层稀疏检索索引源 (维护 Indexer 检索 512 候选块)
[25~27, 29~31, 33~35] ── 中位平滑推理层 (compress_ratio = 1)
[37, 38, 39] ─────────── ⚡ DSpark 动态稀疏目标层 (为 DSpark 投机引擎汇聚高层语义特征)
[40, 41, 42] ─────────── 🔮 Next-3-Predict MTP 投机解码层 (DSparkBlock 独立专家群)
```

---

## 4. 专项核心技术深度专题导航 (Modular Deep-Dives)

为了保证主文档清晰精炼，具体技术深度推导与源码拆解已分拆为独立专题文档，点击直达：

1. 📖 **[01. Engram 外部 1600 万超大词法记忆机制](docs/01_ENGRAM_LEXICAL_MEMORY.md)**：解析 Layer 1 与 Layer 14 外挂 N-gram 嵌入表、卸载代码短语记忆的机制。
2. 📖 **[02. 分层 KV 压缩与长程跳跃索引 (Compressor & Indexer)](docs/02_HIERARCHICAL_KV_INDEXING.md)**：推导 4 个源动力层与 8 个索引节点如何将 1M 上下文 KV Cache 压缩至 1/8。
3. 📖 **[03. DSpark 动态稀疏引擎与高层特化](docs/03_DSPARK_SPARSE_ENGINE.md)**：剖析 Layer 37~39 专属 128 稀疏路由专家与 Sinkhorn 传输过滤。
4. 📖 **[04. Hyper-Connections (HC) 4 路残差超连接机制](docs/04_HYPER_CONNECTIONS_RESIDUAL.md)**：解析 4 路并行流转拓扑与解决梯度饱和之数学证明。
5. 📖 **[05. 与上一小版本 (V4-Flash-0731) 精准演进差分](docs/05_DELTA_VS_V4_FLASH_0731.md)**：逐项对齐 2026 年 7 月版本到 9 月版本的参数改动全貌与动机。

---

## 5. 独立建模与调试验证代码

- 纯 PyTorch 自包含独立参考实现：[`modeling.py`](modeling.py)
- 本地单步验证指令：
  ```bash
  python models/deepseek/deepseek_v4/v4_1_flash/modeling.py
  ```
