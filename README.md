# 开源大模型结构与算法全景学习 (LLM Architectures)

> **专门用于系统化学习、深度剖析与复现主流开源大模型（LLM / VLM / MoE）网络结构与核心算法的代码知识库。**

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)]()

---

## 1. 为什么创建本项目？

开源大语言模型发展迅猛，从最初的标准 Transformer Decoder 架构，迅速演进出大量具有革命性的算法与结构创新：
- **注意力机制**：从传统 MHA（多头注意力）演进至 MQA、GQA（分组查询注意力）、MLA（多头潜在注意力）以及滑动窗口注意力（SWA）；
- **位置编码**：从绝对位置编码演进至 RoPE（旋转位置编码）、线性插值、Dynamic NTK、YaRN、Dual Chunk Attention；
- **计算块与归一化**：从 Post-LN 到 Pre-LN、RMSNorm、DeepNorm、QK-Norm 以及 Logit Soft-capping；
- **稀疏激活 (MoE)**：从标准 Dense FFN 到 SwiGLU，再到细粒度专家划分（Fine-grained MoE）、共享专家隔离（Shared Expert）与无辅助损失负载均衡（Lossless Load Balancing）。

不同机构、不同系列、不同代际的模型在设计权衡（推理吞吐、KV Cache 内存占用、训练稳定性、上下文扩展能力）上各有千秋。本项目旨在提供清晰的**分层目录结构**与**极简自洽的独立实现**，帮助算法工程师与系统开发者透彻掌握每一个架构细节。

---

## 2. 目录体系：开源公司 / 系列 / 版本

项目遵循统一严格的目录组织规范：
`models / [开源机构或公司] / [模型系列] / [系列版本] /`

```text
llm-architectures/
├── models/                                 # 全球开源模型全景库
│   ├── alibaba/                            # 阿里巴巴 (Alibaba Cloud / 通义实验室)
│   │   ├── wan/                            # Wan (通义万相) 视频生成系列
│   │   │   ├── wan3_0/                     # Wan 3.0 (2026-09, 原生音视频一体化生成, 30秒连续)
│   │   │   └── wan2_2/                     # Wan 2.2 Animate 系列 (人物高保真动态生成)
│   │   └── qwen/                           # 通义千问 (Qwen) 系列
│   │       ├── qwen3_8/                    # Qwen 3.8-27B (2026-08, 64层原生多模态, SWE 61.7%)
│   │       ├── qwen3_8_flash/              # Qwen 3.8-Flash-Next (2026-08, 512细粒度专家, 极速响应)
│   │       ├── qwen3_8_moe/                # Qwen 3.8-2.4T-A95B (2026-08, 2.4万亿超大MoE基座)
│   │       ├── qwen3_asr/                  # Qwen3-ASR 1.7B / ForcedAligner (2026-06, RTFx 835.6)
│   │       ├── qwen_image/                 # Qwen-Image-2.1 (2026-09, 流匹配 DiT 汉字排版)
│   │       ├── qwen1/ & qwen1_5/ & qwen2/  # 早期代际归档
│   │       ├── qwen2_5/                    # Qwen-2.5: QK-Norm 训练稳定化 + 128k 强大基座
│   │       └── qwq/                        # QwQ: 强化学习长思维链推理模型
│   ├── deepseek/                           # 深度求索 (DeepSeek)
│   │   ├── deepseek_v4/                    # 2026 前沿旗舰 MoE 系列 (>= 2026-05)
│   │   │   ├── v4_1_flash/                 # V4.1 Flash (2026-09, 384细粒度MoE + 1M窗口, Codeforces 3471)
│   │   │   ├── v4_pro/                     # V4 Pro 0813 (2026-08, 1.6T 万亿MoE旗舰)
│   │   │   ├── v4_flash_dspark/            # V4 Flash DSpark (2026-07, 动态稀疏推理与马尔可夫排序)
│   │   │   └── v4_flash_vision/            # V4 Flash Vision (2026-08, 原生多模态统一流)
│   │   ├── deepseek_llm/                   # DeepSeek-LLM 稠密基础系列
│   │   ├── deepseek_moe/                   # DeepSeek-MoE: 细粒度专家与共享专家首发
│   │   ├── deepseek_v2/                    # DeepSeek-V2 / Lite: MLA (KV Cache 极度压缩)
│   │   ├── deepseek_v3/                    # DeepSeek-V3: 671B 极致 MoE + MLA + 无辅助 Loss
│   │   └── deepseek_r1/                    # DeepSeek-R1: 强化学习推理架构与长思维链
│   ├── zhipu/                              # 智谱 AI (Zhipu AI)
│   │   └── glm/glm5_3/                     # GLM-5.3 Max / Flash (2026-08, AA 智能指数 44.8)
│   ├── moonshot/                           # 月之暗面 (Moonshot AI)
│   │   └── kimi/k3/                        # Kimi K3 Max (2026-07, AA 智能指数 43.6, 长思考链)
│   ├── openbmb/                            # 面壁智能 (OpenBMB)
│   │   └── minicpm/                        # MiniCPM 系列 (端侧小钢炮)
│   │       ├── minicpm5/                   # MiniCPM5-2B (2026-09, ≤4B 官方榜首, 端侧长思考)
│   │       └── minicpm_v4_6/               # MiniCPM-V 4.6 1.3B (2026-05, 1B极速端侧多模态)
│   ├── minimax/                            # MiniMax (稀宇科技)
│   │   ├── minimax_video/h3/               # MiniMax H3 (2026-07, AA Video Arena 开源榜首 1137.4 Elo)
│   │   └── minimax_m/m3/                   # MiniMax-M3 (2026-06, 通用大模型)
│   ├── lightricks/                         # Lightricks
│   │   └── ltx_video/ltx_2_5/              # LTX-2.5 Pro / Fast (2026-08, 22B 超高帧率开源视频生成)
│   ├── black_forest_labs/                  # Black Forest Labs (BFL)
│   │   └── flux_3/flux_3_action/           # FLUX-3-Action (2026-09, 具身智能动作生成 DiT)
│   ├── google/                             # Google DeepMind
│   │   ├── gemma/gemma4/                   # Gemma 4 12B/31B (2026-06, 原生音视频文字一体多模态)
│   │   ├── diffusiongemma/diffusiongemma_26b/ # DiffusionGemma 26B-A4B (2026-06, 首个 MoE 扩散模型)
│   │   ├── embeddinggemma/embeddinggemma_2/ # EmbeddingGemma-2 (2026-09, 最新长文本向量化)
│   │   └── gemma/gemma1/ & gemma2/         # 历史代际归档
│   ├── ibm/                                # IBM Research
│   │   └── granite/                        # Granite 系列
│   │       ├── granite_4_2/                # Granite 4.2 3B (2026-08, ≤4B 企业级紧凑端侧)
│   │       └── granite_speech_5/               # Granite-Speech-5.0 (2026-10, 工业高速抗噪 ASR)
│   ├── nvidia/                             # NVIDIA
│   │   └── nemotron/nemotron_3_5/          # Nemotron 3.5 Lightning (2026-08, Tensor Core 极致对齐)
│   ├── meta/                               # Meta (llama1, llama2, llama3, llama3_1, llama3_2)
│   ├── mistralai/                          # Mistral AI (leanstral_1_5, shieldstral_1, mistral_7b, mixtral)
│   ├── microsoft/                          # 微软 (phi1, phi2, phi3, phi4)
│   ├── tii/                                # TII (falcon_v1, falcon_v2)
│   ├── 01ai/                               # 零一万物 (yi_v1, yi_1_5)
│   └── baichuan/                           # 百川智能 (baichuan1, baichuan2)
├── common/                                 # 通用算子与算法组件库
│   ├── attention/                          # MHA, MQA, GQA, MLA, Sliding Window
│   ├── rope/                               # RoPE, YaRN, Linear/NTK Scaling
│   ├── norm/                               # RMSNorm, LayerNorm, DeepNorm
│   ├── ffn_moe/                            # SwiGLU, TopK MoE, DeepSeekMoE
│   └── quantization/                       # FP8, AWQ, GPTQ 核心原理
├── scripts/                                # 自动化与工程辅助脚本
│   └── verify_index.py                     # 全局索引、死链与相对路径自动化检测器
├── .agents/skills/                         # 项目自动化与情报追踪技能体系
│   └── model-intelligence-crawler/         # 全模态权威排行榜追踪、抗渲染穿透与自更新 Skill
└── docs/                                   # 理论研究与横向对比
    ├── comparisons/                        # 跨模型横向对比矩阵、SOTA 排行榜分析
    └── templates/                          # 统一模型卡片模版
```

---

## 3. 开源前沿 SOTA 梯队导航 (2026-05 后前沿专属)

本仓库严格执行**时效性硬性门禁（发布时间 Release Date ≥ 2026-05-01，早于此时间的旧代际模型一律 PASS 淘汰）**，直连 11 大赛道权威排行榜真实数据层：

- 👑 **通用 LLM / 推理 SOTA**：[`MiMo-V2.6-Pro (2026-09)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`GLM-5.3 Max (2026-08)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`DeepSeek-V4.1-Flash (2026-09)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md)（AA Open Source 智能指数 44+~46+，384 细粒度 MoE + 1M 窗口）
- 🚀 **端侧极小钢炮 (≤4B)**：[`K2 Horizon 3.7B (2026-09)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`MiniCPM5-2B (2026-09)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`Granite 4.2 3B (2026-08)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md)（AA 官方认证 ≤4B 智能密度最高）
- 💻 **代码与工程 Agent SOTA**：[`DeepSeek-V4.1-Flash (2026-09)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (Terminal-Bench 90.6%，DeepSWE 解决率 74.2%) / [`Qwen 3.8-27B (2026-08)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (SWE-bench Pro 61.7%)
- 👁️ **多模态视觉 (VLM) SOTA**：[`DeepSeek-V4-Flash-Vision (2026-08)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`Ling-3.0-flash-VL (2026-09)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`MiniCPM-V 4.6 1.3B (2026-05)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md)
- 🎨 **文生图 (Text-to-Image) SOTA**：[`Qwen-Image-2.1 (2026-09)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`DiffusionGemma 26B (2026-06)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`HiDream-O1-Image (2026-06)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md)
- 🎬 **文生视频 (T2V) SOTA**：[`MiniMax H3 (2026-07)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (AA Video Arena 开源第 1，Elo 1137.4) / [`LTX-2.5 Pro (2026-08)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (Elo 945.9)
- 🎙️ **语音识别 (ASR) SOTA**：[`granite-speech-5.0 (2026-10)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`Qwen3-ASR-1.7B (2026-06)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (WER 5.75%，RTFx 吞吐高达 835.6)
- 🔍 **语义检索 (Embedding) SOTA**：[`UME-R1 (2026-07)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`VultronRetriever-Qwen3.5 (2026-06)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (MTEB 思考型向量模型)

完整 11 大赛道最新数据与穿透脚本，详见 📑 [docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md)。

---

## 4. 核心大模型架构横向对比速查表

| 模型架构 | 开发者 | 注意力机制 | 位置编码 (PE) | 归一化 (Norm) | 激活函数 | 专家机制 (MoE) | 上下文长度 |
|---|---|---|---|---|---|---|---|
| **LLaMA 3.1** | Meta | GQA (8 KV heads) | RoPE (base=500k) | RMSNorm | SwiGLU | Dense | 128k |
| **DeepSeek-V3** | DeepSeek | **MLA** (低秩投影) | Decoupled RoPE | RMSNorm | SwiGLU | **DeepSeekMoE** (共享+细粒度路由) | 128k |
| **Qwen 2.5** | Alibaba | GQA + **QK-Norm** | RoPE (base=1M) | RMSNorm | SwiGLU | Dense / MoE (细粒度) | 128k |
| **Gemma 2** | Google | GQA (交替滑动窗口) | RoPE | RMSNorm (双重) | GeGLU | Dense | 8k |
| **Mixtral 8x7B** | Mistral AI | GQA (SWA) | RoPE | RMSNorm | SwiGLU | Top-2 Gating (8 专家) | 32k |
| **Phi-3** | Microsoft | GQA | SuScaledRoPE | RMSNorm | SwiGLU | Dense | 128k |

---

## 4. 协作与开发指南

每个模型版本目录均包含：
1. `README.md`：架构解析卡片（参数表、架构流转图、核心创新点、数学推导）；
2. `modeling.py`：纯 PyTorch 编写的**自包含独立参考实现**，包含可直接运行的单元验证代码；
3. `config.json`：该版本的官方标准配置参数超参定义。

详细的 Agent 协作契约与贡献工作流，请参考 [AGENTS.md](AGENTS.md)。

---

## 5. 快速上手

运行任意模型的独立架构实现并验证前向传播：

```bash
# 验证索引与相对链接完好性
python3 scripts/verify_index.py

# 验证 DeepSeek-V3 核心 MLA + DeepSeekMoE 架构
python models/deepseek/deepseek_v3/v3/modeling.py

# 验证 LLaMA-3 架构
python models/meta/llama/llama3/modeling.py

# 验证 Qwen-2.5 (QK-Norm + GQA) 架构
python models/alibaba/qwen/qwen2_5/modeling.py
```
