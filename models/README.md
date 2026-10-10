# 全球开源大模型索引与架构导航 (Models Index & Architecture Hub)

> 本目录收录并维护全球主流开源大模型（涵盖通用语言/推理、端侧轻量小模型、代码工程 Agent、多模态视觉 VLM、视频理解、文生图、文生视频、语音 ASR 以及语义检索 Embedding）。  
> 目录严格遵循 **四级层级：`models / [开源机构或公司] / [模型系列] / [具体版本] /`**。

---

## 1. 2026 前沿开源旗舰模型导航 (Post-2026-05 SOTA)

> 严格执行 **Release Date ≥ 2026-05-01** 时效性门禁，收录当前全球各大开源领军机构最新的登顶模型。

| 所属机构 / 公司 | 模型系列 | 代表版本 / 模型 | 准确发布时间 | 架构核心亮点 | 架构卡片直达 |
|---|---|---|:---:|---|:---:|
| **阿里巴巴 (Alibaba)** | **wan** (通义万相) | **Wan 3.0** | 2026-09-04 | 原生音视频一体化多模态生成，最长 30 秒连续物理仿真 | [README](alibaba/wan/wan3_0/README.md) |
| **阿里巴巴 (Alibaba)** | **qwen** (通义千问) | **Qwen 3.8-27B** | 2026-08-05 | 64 层原生统一音视频文本流，SWE-bench Pro 61.7% | [README](alibaba/qwen/qwen3_8/README.md) |
| **阿里巴巴 (Alibaba)** | **qwen** (通义千问) | **Qwen 3.8-Flash-Next** | 2026-08-24 | 512 细粒度路由专家 (Top-10 Router)，极速首字响应 | [README](alibaba/qwen/qwen3_8_flash/README.md) |
| **阿里巴巴 (Alibaba)** | **qwen** (通义千问) | **Qwen 3.8-2.4T-A95B** | 2026-08-08 | 2.4 万亿超大 MoE 基座，92 层极深 Transformer 架构 | [README](alibaba/qwen/qwen3_8_moe/README.md) |
| **阿里巴巴 (Alibaba)** | **qwen** (通义千问) | **Qwen3-ASR 1.7B** | 2026-06-26 | 工业高速语音识别 (RTFx 835.6, WER 5.75%) 与毫秒对齐 | [README](alibaba/qwen/qwen3_asr/README.md) |
| **阿里巴巴 (Alibaba)** | **qwen** (通义千问) | **Qwen-Image-2.1** | 2026-09-14 | 流匹配 DiT (Rectified Flow)，突破中英文字符排版 | [README](alibaba/qwen/qwen_image/README.md) |
| **深度求索 (DeepSeek)**| **deepseek_v4** | **DeepSeek-V4.1-Flash** | 2026-09-10 | 384 细粒度专家 + 1M 上下文，Terminal-Bench 90.6% | [README](deepseek/deepseek_v4/v4_1_flash/README.md) |
| **深度求索 (DeepSeek)**| **deepseek_v4** | **DeepSeek-V4-Pro (1.6T)**| 2026-08-13 | 1.6 万亿参数 MoE 旗舰，SimpleQA / SuperGPQA 开源第 1 | [README](deepseek/deepseek_v4/v4_pro/README.md) |
| **深度求索 (DeepSeek)**| **deepseek_v4** | **V4 Flash DSpark** | 2026-07-04 | 动态稀疏推理与马尔可夫排序 (DSpark / Sinkhorn 最优传输) | [README](deepseek/deepseek_v4/v4_flash_dspark/README.md) |
| **智谱 AI (Zhipu AI)** | **glm** | **GLM-5.3 Max / Flash** | 2026-08-18 | AA Open Source 智能指数 44.8，长程工作流规划优化 | [README](zhipu/glm/glm5_3/README.md) |
| **月之暗面 (Moonshot)**| **kimi** | **Kimi K3 Max** | 2026-07-16 | AA Open Source 智能指数 43.6，端到端长思维链慢思考 | [README](moonshot/kimi/k3/README.md) |
| **面壁智能 (OpenBMB)** | **minicpm** | **MiniCPM5-2B** | 2026-09-07 | ≤4B 官方认证最高智能密度，端侧原生强化学习推理链 | [README](openbmb/minicpm/minicpm5/README.md) |
| **稀宇科技 (MiniMax)** | **minimax_video** | **MiniMax H3 (768p)** | 2026-07-28 | AA Video Arena 开源视频榜首 (1137.4 Elo)，3D 物理一致性 | [README](minimax/minimax_video/h3/README.md) |
| **Lightricks** | **ltx_video** | **LTX-2.5 Pro (22B)** | 2026-08-11 | AA Video Arena 第 2 名 (945.9 Elo)，超高生成帧率与模块化 IC-LoRA | [README](lightricks/ltx_video/ltx_2_5/README.md) |
| **Black Forest Labs** | **flux_3** | **FLUX-3-Action** | 2026-09-22 | 具身智能动作生成与空间物理交互 DiT | [README](black_forest_labs/flux_3/flux_3_action/README.md) |
| **Google DeepMind** | **gemma** | **Gemma 4 12B/31B** | 2026-06-03 | 原生音视频文字一体多模态，Dual RMSNorm 极深架构 | [README](google/gemma/gemma4/README.md) |
| **Google DeepMind** | **diffusiongemma** | **DiffusionGemma 26B** | 2026-06-09 | 业界首个大参数量 MoE 稀疏扩散生成模型 (激活 4B) | [README](google/diffusiongemma/diffusiongemma_26b/README.md) |
| **IBM Research** | **granite** | **Granite 4.2 3B** | 2026-08-25 | ≤4B 企业级紧凑模型，专注代码审计与工具调用 | [README](ibm/granite/granite_4_2/README.md) |
| **NVIDIA** | **nemotron** | **Nemotron 3.5 Lightning**| 2026-08-11 | 硬件感知设计，对齐 GPU Tensor Core FP4/FP8 极限吞吐 | [README](nvidia/nemotron/nemotron_3_5/README.md) |

---

## 2. 经典基石架构模型导航 (Foundational Baselines)

> 包含大模型发展史上具有划时代技术意义的经典架构实现与对比卡片。

| 机构 / 公司 | 模型系列 | 代表版本 | 核心技术里程碑 | 架构卡片直达 |
|---|---|---|---|:---:|
| **Meta AI** | **llama** | **LLaMA-3** | 8B 标配 GQA + 128k 大词表 + RMSNorm/SwiGLU | [README](meta/llama/llama3/README.md) |
| **深度求索 (DeepSeek)**| **deepseek_v3** | **DeepSeek-V3** | MLA (多头潜在注意力) + DeepSeekMoE (共享+细粒度) | [README](deepseek/deepseek_v3/v3/README.md) |
| **阿里巴巴 (Alibaba)** | **qwen** | **Qwen-2.5** | QK-Norm 训练稳定化关键技术 + 100万 RoPE Base | [README](alibaba/qwen/qwen2_5/README.md) |
| **Google DeepMind** | **gemma** | **Gemma-2** | 交替滑动窗口注意力 + Logit Soft-capping 软截断 | [README](google/gemma/gemma2/README.md) |
| **Mistral AI** | **mixtral** | **Mixtral 8x7B** | 经典 Top-2 Router 稀疏门控 MoE | [README](mistralai/mixtral/mixtral_8x7b/README.md) |
| **微软 (Microsoft)** | **phi** | **Phi-3** | SuScaledRoPE 128k 长窗口 + 教科书级数据精调 | [README](microsoft/phi/phi3/README.md) |

---

## 3. 维护规范说明 (Maintenance Contract)

根据 [`../AGENTS.md`](../AGENTS.md) 规定：
1. **新增模型必更新**：当开发者或 AI Agent 在 `models/` 下新增或归档任何模型版本时，**必须同步在本文档对应的表格中增加一行**；
2. **时效性校验**：新增前沿模型必须核对 `Release Date`（必须 ≥ 2026-05-01）；
3. **提交前死链校验**：完成编辑后，提交前必须运行 `python3 scripts/verify_index.py` 确保本文件内所有相对链接 100% 有效。
