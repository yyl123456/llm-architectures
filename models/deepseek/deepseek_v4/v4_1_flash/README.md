# DeepSeek-V4.1-Flash 架构预研卡片

---

## 1. 基本概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | DeepSeek-V4.1-Flash |
| **开源机构** | 深度求索 (DeepSeek-AI) |
| **发布时间** | **2026-09-10** |
| **总参数量** | 552B (Backbone Params) |
| **激活参数量** | 8B ~ 16B (动态可伸缩) |
| **上下文窗口** | **1,048,576 tokens (1M)** |
| **开源协议** | MIT License |
| **代码/权重库** | [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) |

---

## 2. 核心架构亮点与算法突破

1. **384 细粒度路由专家 (Fine-Grained 384 Experts)**：
   - 专家总数由 V3 的 256 扩展到 384，每个 Token 动态激活 6 个专家；
   - 保持 1 个全局共享专家，极致降低单 Token 激活参数至 8B~16B。
2. **分层 KV 压缩与索引机制 (Layer-wise KV Compression & Indexing)**：
   - 为支撑 1M 极长上下文，引入 `compress_ratios`（分层压缩率）与 `index_source_layer_ids`；
   - 配合 DSpark 动态稀疏机制，使得 1M 上下文在长程推理中保持超低内存占用与高吞吐。
3. **1~100 连续可控思考深度 (Controllable Reasoning Effort 1-100)**：
   - 支持动态设定 `reasoning_effort`，在快速问答与复杂逻辑/竞赛代码之间自由权衡。
4. **评测基准登顶表现**：
   - Codeforces Rating: **3471**；
   - Terminal-Bench 2.1: **90.6%**（开源第一）；
   - DeepSWE v1.1: 真实 GitHub 缺陷修复率 **74.2%**。
