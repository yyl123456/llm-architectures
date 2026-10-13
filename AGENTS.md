# AGENTS.md - LLM Architectures 协作与架构规范

本文档是 **LLM Architectures（开源大模型结构与算法全景学习库）** 的唯一全局治理规范与工作指南。面向人类开发者以及所有在此代码库工作的 AI Agent。

---

## 0. 当前时间线锚点与时效性铁律 (Timeline & Temporal Context)

> ⚠️ **全局强制时间线声明 (MANDATORY)**：  
> **当前现实基准时间是：`2026 年 10 月`**。
> 
> 大模型技术与开源生态正在以极高速度迭代演进，**严禁使用陈旧过时的历史认知来“考古”！**
> 1. **对齐最新一至两年 (2025 ~ 2026 年)**：本项目的所有调研、排行榜跟踪、架构拆解与技术选型，必须**严格锚定在 2025 年至 2026 年（重点是 2026 年近期的前沿成果）**；
> 2. **杜绝历史代际倒退**：2024 年及更早期的旧模型（如早期的 LLaMA-1/2/3、老旧的 Whisper、早期 FLUX.1、早期 Qwen 等）早已属于历史基础，**绝对不允许当作当前前沿或最新 SOTA 推荐**；
> 3. **最新成果聚焦**：重点跟踪 2026 年的全新突破（如 DeepSeek-V4.1 系列、阿里 Wan 3.0 与 Qwen3.8 全系列、GLM-5.3、Kimi K3、MiniCPM5-2B、MiniMax H3、FLUX.3 Action、Gemma 4、Granite 4.2/5.0 等）；
> 4. **前沿门禁线**：所有列入前沿 SOTA 矩阵的模型，**发布时间必须在 2026 年 5 月之后（Release Date ≥ 2026-05-01）**，早于此时间的全部作为历史归档，坚决保持代码库的前沿性与高置信度。

---

## 1. 项目愿景与核心定位

本项目专注于**开源大模型（LLM / VLM / DiT / MoE）的模型网络结构、数学原理、核心算法与代码实现**。
通过统一规范的目录组织、清晰精炼的最小实现（PyTorch / NumPy）以及深度架构对比，建立一个系统化、高置信度的大模型架构演进知识库与复现代码库。

在调研和复现过程中，深度吸收并借鉴全球顶级模型架构分析站点（如 **Sebastian Raschka 的 LLM Architecture Gallery** 架构图解与并排 Diff 对比、**Frontier ML Open Model Index** 硬件工程与显存参数索引、**zhanglpg 的 LLM & Multimodal Architecture Atlas** 全模态图谱，以及 **LLM Architectures Explained** 核心算子数学推导）的最佳实践与拓扑表达规范。

### 核心目标
1. **结构严谨**：严格遵循 `开源公司/系列/版本/`（`vendor/series/version/`）层级，结构清晰可扩展。
2. **算法透彻**：不仅记录模型配置参数，更剖析核心注意力机制（MHA/GQA/MQA/MLA/Sliding Window）、位置编码（RoPE/YaRN/Dual Chunk）、激活与归一化（SwiGLU/RMSNorm/LayerNorm）、MoE 路由（TopK/Shared Expert/Lossless）等算法细节。
3. **代码自洽**：每个具体模型版本包含核心模块的独立 PyTorch 最小化参考实现（Standalone Reference Implementation）与架构卡片，便于单步调试与推导。
4. **横向沉淀**：共性算法算子沉淀在 `common/` 模块，形成跨模型对比与演进矩阵。

---

## 2. 目录规范与组织架构

### 2.1 目录组织原则
项目严格遵循四级模型分类树（所有开源公司统一收敛在 `models/` 目录下）：
```text
llm-architectures/
├── models/                             # 全球开源模型主目录
│   └── <开源公司或机构 (Vendor)>/
│       └── <模型系列 (Series)>/
│           └── <具体版本 (Version)>/
│               ├── README.md           # 该版本的架构详解卡片（配置、算法特性、网络拓扑）
│               ├── modeling.py         # 最小化独立 PyTorch 模型实现（核心 Transformer / Attention / MoE）
│               └── config.json (可选)  # 典型超参数配置
├── common/                             # 通用算子与算法组件库
│   ├── attention/                      # MHA, MQA, GQA, MLA, Sliding Window, Linear Attention
│   ├── rope/                           # RoPE, YaRN, RoPE Scaling, Dynamic NTK
│   ├── norm/                           # RMSNorm, LayerNorm, DeepNorm
│   ├── ffn_moe/                        # SwiGLU, DeepSeekMoE, Mixtral MoE
│   └── quantization/                   # FP8, AWQ, GPTQ 概念与结构适配
├── scripts/                            # 质量保障与自动化工具
│   └── verify_index.py                 # 全局索引检测脚本（提交前必跑）
├── .agents/skills/                     # 项目自动化与情报追踪技能体系
│   └── model-intelligence-crawler/     # 全模态权威排行榜追踪、抗渲染穿透与自更新 Skill
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

### 3.1 `README.md`（架构卡片与技术文档组织）
须包含以下小节：
1. **基本概览**：开源机构、开源时间、基座规模（参数量、激活参数量）、上下文窗口长度、官方论文/技术报告链接。
2. **核心架构亮点**：例如引入了何种 Attention（GQA/MLA）、何种 MoE（Top-K / 共享专家 / 路由偏置）、何种归一化（RMSNorm）、何种激活函数（SwiGLU）。
3. **架构拓扑与张量流转图**：ASCII 或 Mermaid 图表展示 Block 级前向流程，并内嵌本地架构图（`assets/architectures/`）。
4. **核心超参数对照表**：`vocab_size`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size` 等。
5. **相比上一小版本的演进要点 (Delta Analysis vs Immediate Predecessor)**：
   - ⚠️ **严禁跨越远古代际对比**；
   - **必须直接对齐紧邻的上一小版本**，逐个字段对比参数改动数值与动机。
6. **技术模块化分拆规范 (Modular Topic Separation)**：
   - ⚠️ **严禁将所有长篇推导和细节全部塞进单一 README.md**；
   - 复杂、独立的前沿技术（如 Engram 词法外挂、DSpark 稀疏引擎、分层 KV 压缩、Hyper-Connections 残差流、小版本 Diff 等），**必须在模型目录下创建 `docs/` 子目录并分拆为独立的专题 Markdown 文档**；
   - 模型主目录的 `README.md` 承担**总导航卡片 (Architecture Hub)** 角色，通过超链接索引并直达各个技术文档，保持主文档清爽、结构分明、易于展开阅读。

### 3.2 `modeling.py`（最小化实现）
- 必须基于原生 PyTorch 实现（避免重度依赖第三方庞大框架），代码具备可读性、模块化和完整的类型标注与张量 Shape 注释。
- 至少实现：
  - Attention 模块（包含位置编码计算）
  - FFN / MoE 路由模块
  - Decoder Block 模块
  - 顶层 Model / CausalLM 类
- 附带一个可运行的 `if __name__ == "__main__":` 测试用例，构造 Dummy Input 验证 Forward pass 输出维度与数值稳定性。

---

## 4. Agent 执行工作流与行为规范 (Core Protocols)

当 Agent 受托添加、调研或更新模型结构时，必须严格执行以下七大刚性法则：

### 4.1 深入源码与抗渲染数据穿透 (Anti-Rendering Ground Truth)
1. **穿透 SPA / 客户端渲染**：
   - 现代榜单（如 Artificial Analysis、LiveCodeBench、SWE-bench）普遍采用 Next.js App Router 或 React SPA，直接发起常规 HTTP 请求常遇空白或 `Loading...` 假死；
   - **严禁因抓取阻碍而退回大模型先验记忆或编造伪造数据**；
   - 必须优先使用内建工具：Jina Reader 代理（`https://r.jina.ai/<url>`）、逆向 Next.js `self.__next_f.push` 动态数据切片，或直连官方 Hugging Face Hub API 提取原始 JSON/CSV 底层真实数值。
2. **拒绝粗糙概括，严查层级异构性 (Heterogeneous Layer Specialization)**：
   - 严禁将深层 Transformer 笼统当成“均一雷同的重复结构（Block × N）”；
   - 必须通过官方推理代码（如 `inference/model.py`）核实各层的特殊分工：确认哪些层是 KV 压缩动力源（Compressor）、哪些层是跳跃索引源（Indexer）、哪些层外挂了超大词表记忆（如 Engram 16M 词表）、哪些层启用了专用动态稀疏（如 DSpark）以及残差超连接机制（Hyper-Connections）。

### 4.2 严时限门禁与前沿模型淘汰准则 (Strict Post-2026-05 Rule)
1. **时效性铁律**：所有作为前沿 SOTA 收录的模型，必须标明确切发布时间（`Release Date`）；
2. **强制硬性淘汰**：凡发布时间**早于 2026 年 5 月 1 日（< 2026-05-01）的老模型一律 PASS 剔除**，作为历史基准归档，绝不允许作为当前最新 SOTA 推荐。

### 4.3 紧邻上一小版本精准差分 (Delta Analysis vs Immediate Predecessor)
1. **严禁跨古代际比较**：例如分析 V4.1 时，严禁越级去和两年前的远古版本（如 V3）比较；
2. **必须直接对齐紧邻的前一个小版本**（如 V4.1-Flash 直接对比 V4-Flash-0731）：
   - 逐个字段对比隐藏层宽度、层数增减、路由专家数、压缩分桶、投机预测层数等关键参数的精准差值；
   - 阐明小版本迭代背后的具体工程权衡与收益。

### 4.4 架构可视化图解本地化移植 (Architecture Assets Localization)
1. 深度借鉴权威专业站点（**Sebastian Raschka 的 LLM Architecture Gallery**、**zhanglpg 的 Model Atlas**、**Frontier ML**）的最佳拓扑表达；
2. 遇到权威架构示意图时，**必须将高清原图下载保存至 `assets/architectures/` 本地目录**，杜绝外部链接失效或防盗链拦截；
3. 在模型 `README.md` 中以相对路径形式规范内嵌（说明：使用 Markdown 图片语法链接至 `assets/architectures/` 相应文件）。

### 4.5 代码自洽性与测试验证
- 编写的 `modeling.py` 必须基于纯 PyTorch 原生实现，且必须在本地通过 `python3 -m py_compile` 与独立单元测试验证，确保输出 Tensor 形状无误。

### 4.6 全局索引联动与提交前死链检测 (Zero Broken Links Gate)
- 新增模型后，必须同步更新：
  1. 根目录 `README.md` 中的目录树与导航索引；
  2. `models/README.md` 中的结构化模型导航总表（去除重复公司/系列冗余名，保持层级箭头视觉清爽）；
- ⚠️ **提交前硬性门禁**：**必须运行 `python3 scripts/verify_index.py`**，确认全库 Markdown 相对路径 100% 存在且无死链，方可执行 `git commit`。

### 4.7 规范化 Git 提交 (Conventional Commits)
- 提交信息遵循规范，如 `feat(models): ...`、`docs(deepseek): ...`、`style(models): ...`，并及时推送到 GitHub 远端仓库。

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
