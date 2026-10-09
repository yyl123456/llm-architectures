# AGENTS.md - LLM Architectures 协作与架构规范

本文档是 **LLM Architectures（开源大模型结构与算法全景学习库）** 的唯一全局治理规范与工作指南。面向人类开发者以及所有在此代码库工作的 AI Agent。

---

## 1. 项目愿景与核心定位

本项目专注于**开源大模型（LLM / VLM / MoE）的模型网络结构、数学原理、核心算法与代码实现**。
通过统一规范的目录组织、清晰精炼的最小实现（PyTorch / NumPy）以及深度架构对比，建立一个系统化、高置信度的大模型架构演进知识库与复现代码库。

### 核心目标
1. **结构严谨**：严格遵循 `开源公司/系列/版本/`（`vendor/series/version/`）层级，结构清晰可扩展。
2. **算法透彻**：不仅记录模型配置参数，更剖析核心注意力机制（MHA/GQA/MQA/MLA/Sliding Window）、位置编码（RoPE/YaRN/Dual Chunk）、激活与归一化（SwiGLU/RMSNorm/LayerNorm）、MoE 路由（TopK/Shared Expert/Lossless）等算法细节。
3. **代码自洽**：每个具体模型版本包含核心模块的独立 PyTorch 最小化参考实现（Standalone Reference Implementation）与架构卡片，便于单步调试与推导。
4. **横向沉淀**：共性算法算子沉淀在 `common/` 模块，形成跨模型对比与演进矩阵。

---

## 2. 目录规范与组织架构

### 2.1 目录组织原则
项目严格遵循三级模型分类树：
```text
llm-architectures/
├── <开源公司或机构 (Vendor)>/
│   └── <模型系列 (Series)>/
│       └── <具体版本 (Version)>/
│           ├── README.md               # 该版本的架构详解卡片（配置、算法特性、网络拓扑）
│           ├── modeling.py             # 最小化独立 PyTorch 模型实现（核心 Transformer / Attention / MoE）
│           └── config.json (可选)      # 典型超参数配置
├── common/                             # 通用算子与算法组件库
│   ├── attention/                      # MHA, MQA, GQA, MLA, Sliding Window, Linear Attention
│   ├── rope/                           # RoPE, YaRN, RoPE Scaling, Dynamic NTK
│   ├── norm/                           # RMSNorm, LayerNorm, DeepNorm
│   ├── ffn_moe/                        # SwiGLU, DeepSeekMoE, Mixtral MoE
│   └── quantization/                   # FP8, AWQ, GPTQ 概念与结构适配
├── .agents/skills/                     # 项目自动化与情报追踪技能体系
│   └── model-intelligence-crawler/     # 全模态权威排行榜追踪、站点探活与自我演进 Skill
└── docs/                               # 论文精读、演进矩阵与横向对比
    ├── comparisons/                    # 架构横向对比分析与最新 SOTA 榜单
    └── templates/                      # 架构卡片标准模版与编写指南
```

### 2.2 命名规范
- **公司目录名（小写字母+数字）**：如 `meta`, `deepseek`, `alibaba`, `mistralai`, `google`, `microsoft`, `01ai`, `baichuan`, `tii` 等。
- **系列目录名（小写字母+下划线）**：如 `llama`, `deepseek_v2`, `qwen`, `mistral`, `gemma`, `phi` 等。
- **版本目录名（小写字母+下划线）**：如 `llama2`, `llama3`, `llama3_1`, `v2`, `v3`, `r1`, `mistral_7b`, `mixtral_8x7b` 等。

---

## 3. 模型版本目录必须包含的内容 (Contract)

对任何一个具体的模型版本（如 `deepseek/deepseek_v3/v3/` 或 `meta/llama/llama3/`），标准交付物包括：

### 3.1 `README.md`（架构卡片）
须包含以下小节：
1. **基本概览**：开源机构、开源时间、基座规模（参数量、激活参数量）、上下文窗口长度、官方论文/技术报告链接。
2. **核心架构亮点**：例如引入了何种 Attention（GQA/MLA）、何种 MoE（Top-K / 共享专家 / 路由偏置）、何种归一化（RMSNorm）、何种激活函数（SwiGLU）。
3. **架构拓扑与张量流转图**：ASCII 或 Mermaid 图表展示 Block 级前向流程。
4. **核心超参数对照表**：`vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size` 等。
5. **相比前序版本的演进要点**：改动动机、效果及工程考量。

### 3.2 `modeling.py`（最小化实现）
- 必须基于原生 PyTorch 实现（避免重度依赖第三方庞大框架），代码具备可读性、模块化和完整的类型标注与张量 Shape 注释。
- 至少实现：
  - Attention 模块（包含位置编码计算）
  - FFN / MoE 路由模块
  - Decoder Block 模块
  - 顶层 Model / CausalLM 类
- 附带一个可运行的 `if __name__ == "__main__":` 测试用例，构造 Dummy Input 验证 Forward pass 输出维度与数值稳定性。

---

## 4. Agent 执行工作流与行为规范

当 Agent 受托添加或更新模型结构时，必须遵循以下步骤：

1. **查证官方资料与权威来源**：优先查阅官方技术报告（arXiv）、官方 GitHub 官方仓库开源建模代码（或 HuggingFace `transformers/src/transformers/models/...` 中的官方提交）。严禁主观凭空臆测超参数与模块连接。
2. **检查与遵循目录规范**：在 `vendor/series/version/` 确切路径下作业。如果属于通用算法，应下沉或引用 `common/` 模块。
3. **测试自洽性**：编写的 `modeling.py` 必须能够独立运行并输出预期的 Tensor Shape，无未捕获异常。
4. **更新全局索引**：完成新模型添加后，同步更新根目录 `README.md` 的模型演进矩阵与导航索引。
5. **Git 规范**：
   - 提交信息遵循 Conventional Commits 规范，例如：
     - `feat(meta/llama3): add llama3 architecture notes and minimal modeling`
     - `feat(deepseek/v3): implement mla and deepseekmoe block`
     - `docs(common/rope): add yarn and dynamic ntk analysis`

---

## 5. 重点跟踪模型与演进规划表

| 公司 / 机构 | 系列 | 版本 | 核心算法关注点 |
|---|---|---|---|
| **DeepSeek** | `deepseek_v2`, `deepseek_v3`, `deepseek_r1` | `v2`, `v3`, `r1` | MLA (Multi-head Latent Attention), DeepSeekMoE (共享专家+路由偏置), 无辅助损失负载均衡, Multi-Token Prediction (MTP) |
| **Meta** | `llama` | `llama1`, `llama2`, `llama3`, `llama3_1`, `llama3_2` | MHA -> GQA 演进, RoPE Base 缩放 (8M), 128k 上下文, 1B/3B 蒸馏与轻量化 |
| **Alibaba** | `qwen` | `qwen1`, `qwen1_5`, `qwen2`, `qwen2_5`, `qwq` | QK-Norm 稳定性优化, Dual Chunk Attention, 极深极宽 MoE 架构 |
| **Mistral AI** | `mistral`, `mixtral` | `mistral_7b`, `mixtral_8x7b`, `mixtral_8x22b` | Sliding Window Attention (SWA), 稀疏门控 MoE Top-2 |
| **Google** | `gemma` | `gemma1`, `gemma2` | 局部与全局滑动注意力交替、Logit Soft-capping、Post-Norm 与 Pre-Norm 双重归一化 |
| **Microsoft** | `phi` | `phi1`, `phi2`, `phi3`, `phi4` | 针对合成数据调优的轻量架构, SuScaledRoPE, 紧凑 Block 设计 |

---
*本规范随着开源社区最新架构的发展持续迭代。*
