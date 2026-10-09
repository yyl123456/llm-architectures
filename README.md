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
`[开源机构或公司] / [模型系列] / [系列版本] /`

```text
llm-architectures/
├── meta/                                   # Meta (Facebook)
│   └── llama/                              # LLaMA 系列
│       ├── llama1/                         # LLaMA-1: RMSNorm + SwiGLU + RoPE 基石
│       ├── llama2/                         # LLaMA-2: 引入 34B/70B GQA
│       ├── llama3/                         # LLaMA-3: 8B 全面采用 GQA, 128k 词表
│       ├── llama3_1/                       # LLaMA-3.1: 128k 长上下文, RoPE Base=500k, 405B 旗舰
│       └── llama3_2/                       # LLaMA-3.2: 1B/3B 紧凑模型与 Vision 架构
├── deepseek/                               # 深度求索 (DeepSeek)
│   ├── deepseek_llm/                       # DeepSeek-LLM 稠密基础系列
│   ├── deepseek_moe/                       # DeepSeek-MoE: 细粒度专家与共享专家首发
│   ├── deepseek_v2/                        # DeepSeek-V2 / Lite: MLA (KV Cache 极度压缩)
│   ├── deepseek_v3/                        # DeepSeek-V3: 671B 极致 MoE + MLA + 无辅助 Loss
│   └── deepseek_r1/                        # DeepSeek-R1: 强化学习推理架构与长思维链
├── alibaba/                                # 阿里巴巴 (Alibaba Cloud)
│   └── qwen/                               # 通义千问 (Qwen) 系列
│       ├── qwen1/                          # Qwen-1: Untied Embedding + NTK 感知插值
│       ├── qwen1_5/                        # Qwen-1.5: 架构标准化与广泛尺寸支持
│       ├── qwen2/                          # Qwen-2: 全尺寸 GQA + Tie-Word-Embeddings 优化
│       ├── qwen2_5/                        # Qwen-2.5: QK-Norm 训练稳定化 + 128k 强大基座
│       └── qwq/                            # QwQ: 强化学习长思维链推理模型
├── mistralai/                              # Mistral AI
│   ├── mistral/                            # Mistral Dense 系列
│   │   ├── mistral_7b/                     # Mistral-7B: Sliding Window Attention (SWA)
│   │   └── mistral_large/                  # Mistral-Large: 工业级通用大模型
│   └── mixtral/                            # Mixtral 稀疏 MoE 系列
│       ├── mixtral_8x7b/                   # Mixtral 8x7B: Top-2 Router 稀疏专家
│       └── mixtral_8x22b/                  # Mixtral 8x22B: 大规模 MoE
├── google/                                 # Google
│   └── gemma/                              # Gemma 系列
│       ├── gemma1/                         # Gemma-1: GeGLU + RoPE + RMSNorm
│       └── gemma2/                         # Gemma-2: 交替 SWA/全注意力 + Logit Soft-capping + 双重 Norm
├── microsoft/                              # 微软 (Microsoft)
│   └── phi/                                # Phi 系列 (小钢炮轻量模型)
│       ├── phi1/                           # Phi-1 / 1.5: 教科书级高质量数据小模型
│       ├── phi2/                           # Phi-2: 2.7B Dense 架构
│       ├── phi3/                           # Phi-3: SuScaledRoPE 128k 长窗口 + 紧凑 Block
│       └── phi4/                           # Phi-4: 14B 高性能合成数据推理模型
├── tii/                                    # TII (阿布扎比技术创新研究所)
│   └── falcon/                             # Falcon 系列 (并行注意力+MLP)
│       ├── falcon_v1/                      # Falcon-7B/40B: Multi-Query Attention (MQA)
│       └── falcon_v2/                      # Falcon-2 11B: 视觉融合与 MoE 探索
├── 01ai/                                   # 零一万物 (01.AI)
│   └── yi/                                 # Yi 系列
│       ├── yi_v1/                          # Yi-34B: 200k 超长上下文预训练
│       └── yi_1_5/                         # Yi-1.5: 提升代码与数学能力
├── baichuan/                               # 百川智能 (Baichuan)
│   └── baichuan/                           # Baichuan 系列
│       ├── baichuan1/                      # Baichuan-7B (RoPE) / 13B (ALiBi)
│       └── baichuan2/                      # Baichuan2: 预训练稳定化优化 (NormHead)
├── common/                                 # 通用算子与算法组件库
│   ├── attention/                          # MHA, MQA, GQA, MLA, Sliding Window
│   ├── rope/                               # RoPE, YaRN, Linear/NTK Scaling
│   ├── norm/                               # RMSNorm, LayerNorm, DeepNorm
│   ├── ffn_moe/                            # SwiGLU, TopK MoE, DeepSeekMoE
│   └── quantization/                       # FP8, AWQ, GPTQ 核心原理
└── docs/                                   # 理论研究与横向对比
    ├── comparisons/                        # 跨模型横向对比矩阵、SOTA 排行榜分析
    └── templates/                          # 统一模型卡片模版
```

---

## 3. 开源前沿 SOTA 梯队导航 (11 大赛道全模态直连实测)

本仓库持续直连全球 11 大权威评测基准（涵盖通用推理、端侧小模型、代码对抗、工程 Agent、多模态 VLM、视频理解、文生图、文生视频、ASR 语音以及向量检索 MTEB）底层数据，跟踪各领域最新开源顶流：

- 👑 **通用 LLM / 推理 SOTA**：[`MiMo-V2.6-Pro`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`GLM-5.3 (Max)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`DeepSeek-V4 Pro`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md)（AA Open Source 智能指数 44+~46+，细粒度 MoE + 大尺度 RL）
- 🚀 **端侧极小钢炮 (≤4B)**：[`K2 Horizon 3.7B`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`MiniCPM5-2B`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`Granite 4.2 3B`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md)（AA Tiny Models 认证最高智能密度）
- 💻 **代码动态对抗 SOTA**：[`DeepSeek-R1-0528`](./deepseek/deepseek_r1/r1/) (Pass@1 达 **84.4%**) / [`OpenReasoning-Nemotron-32B`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (**81.0%**) / [`Qwen3-235B-A22B`](./alibaba/qwen/) (**80.4%**)（LiveCodeBench 真实全量评测均分）
- 🛠️ **软件工程 Agent SOTA**：[`DeepSeek-R1 (Agentic)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (SWE-bench 49%~55%) / [`Qwen 3.8-27B`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (SWE-bench Pro 61.7%)
- 👁️ **多模态视觉 (VLM) SOTA**：[`InternVL-Chat-V1.5 (26B/78B)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`Qwen2.5-VL 72B`](./alibaba/qwen/qwen2_5/)（OpenVLM 官方综合评分前列）
- 🎨 **文生图 (Text-to-Image) SOTA**：[`FLUX.2 [dev]`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (Elo 1000.0) / [`HiDream-O1-Image`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (Elo 956.6) / [`FLUX.2 [klein] 9B`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (AA Image Arena 开源榜首)
- 🎬 **文生视频 (T2V) SOTA**：[`MiniMax H3 (768p)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (Elo 1153.3) / [`LTX-2.5 Pro`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (Elo 963.4)（AA Video Arena 物理连续性与真实度之星）
- 🎙️ **语音识别 (ASR) SOTA**：[`Phi-4-multimodal-instruct`](./microsoft/phi/phi4/) (WER **5.02%**) / [`Granite-Speech-3.3`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (WER **5.26%**) / [`Qwen3-ASR-1.7B`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) (RTFx **835.6**)（Open ASR 官方数据直连）
- 🔍 **语义检索与重排 (Embedding)**：[`UME-R1 (7B/2B)`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`VultronRetriever-Qwen3.5`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md) / [`BGE-M3`](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md)（MTEB 思考型与高维度检索最新演进）

完整 11 大赛道底层数据逆向解析报告，详见 📑 [docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md](./docs/comparisons/SOTA_OPEN_LLM_LEADERBOARD.md)。

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
# 验证 DeepSeek-V3 核心 MLA + DeepSeekMoE 架构
python deepseek/deepseek_v3/v3/modeling.py

# 验证 LLaMA-3 架构
python meta/llama/llama3/modeling.py

# 验证 Qwen-2.5 (QK-Norm + GQA) 架构
python alibaba/qwen/qwen2_5/modeling.py
```
