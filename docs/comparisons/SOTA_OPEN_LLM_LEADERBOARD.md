# 全模态开源大模型前沿排行榜与 SOTA 梯队全景 (Omni-Modal Open AI Leaderboard & SOTA Matrix)

> **覆盖模态**：通用语言/推理、端侧小模型、代码/软件工程 Agent、多模态视觉语言 (VLM)、视频理解、文生图、文生视频/图生视频、语音识别 (ASR)、向量嵌入与重排 (Embedding / Reranker)。  
> **数据基准**：严格对应 [`model-intelligence-crawler`](../../.agents/skills/model-intelligence-crawler/SKILL.md) 技能中的 11 个权威活体评测站点。

---

## 1. 全模态开源 SOTA 矩阵总览 (Omni-Modal SOTA Overview)

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                               全球开源 AI 各领域顶流王者 (SOTA Snapshot)                           │
├─────────────────────────┬────────────────────────────┬────────────────────────┬──────────────────┤
│ 领域 / 模态             │ 当前 SOTA 开源模型         │ 所属机构               │ 核心权威评测基准 │
├─────────────────────────┼────────────────────────────┼────────────────────────┼──────────────────┤
│ 通用 LLM / 旗舰推理     │ DeepSeek-V3 / DeepSeek-R1  │ 深度求索 (DeepSeek)    │ AA Open Source   │
│ 端侧轻量小模型 (≤4B)    │ Qwen2.5-3B / Llama-3.2-3B  │ 阿里 Qwen / Meta       │ AA Tiny Models   │
│ 代码生成与竞赛编程      │ Qwen2.5-Coder-32B-Inst     │ 阿里巴巴 (Qwen)        │ LiveCodeBench    │
│ 软件工程 Agent          │ DeepSeek-R1 / Qwen2.5-Coder│ DeepSeek / 阿里 Qwen   │ SWE-bench        │
│ 多模态视觉语言 (VLM)    │ Qwen2.5-VL-72B / InternVL2.5│ 阿里 Qwen / OpenGVLab  │ Open VLM (HF)    │
│ 视频理解 (Video-LLM)    │ Qwen2.5-VL / MiniCPM-V 2.6 │ 阿里 Qwen / 面壁智能   │ OpenCompass Video│
│ 图像生成 (Text-to-Image)│ FLUX.1 [dev]               │ Black Forest Labs      │ AA Image Arena   │
│ 视频生成 (Text-to-Video)│ Wan2.1-14B / HunyuanVideo  │ 阿里通义万相 / 腾讯混元 │ AA Video / VBench│
│ 语音识别 (ASR)          │ Whisper Large-v3 Turbo     │ OpenAI / 社区开源权重  │ Open ASR (HF)    │
│ 语义向量与重排 (MTEB)   │ BGE-M3 / NV-Embed-v2       │ 智源 BAAI / NVIDIA     │ MTEB Leaderboard │
└─────────────────────────┴────────────────────────────┴────────────────────────┴──────────────────┘
```

---

## 2. 细分赛道权威排行榜与代表模型解析

### 2.1 通用 LLM 与长思维链推理 (General & Reasoning LLMs)
* 权威评测源：**[AA Open Source Leaderboard](https://artificialanalysis.ai/models/open-source)**
* 重点关注：综合智能质量评分 (Quality Index)、推理性价比、生成吞吐 (Tokens/s) 与上下文窗口

| 排名梯队 | 模型名称 | 机构 | 架构形态 | 激活/总参数 | 关键优势与技术亮点 |
|:---:|---|---|---|---|---|
| 👑 **综合 SOTA** | **DeepSeek-V3** | DeepSeek | 细粒度 MoE | 37B / 671B | 质量跑分超越多款顶尖闭源模型；MLA 潜在注意力将 KV Cache 压缩为原先 1/5；无辅助 Loss 动态偏置路由 |
| 🧠 **推理 SOTA** | **DeepSeek-R1** | DeepSeek | 细粒度 MoE | 37B / 671B | 纯大规模强化学习（RL）激发出自主纠错与长链反思能力，AIME 2024 与 MATH-500 超越 o1-preview |
| 🐘 **稠密天花板**| **Llama 3.1 405B** | Meta AI | Dense | 405B / 405B | 全球最大纯开源稠密模型，世界级多语言知识储备与工业级小模型蒸馏教师 |
| ⚡ **70B 主力** | **Qwen 2.5 72B-Inst** | 阿里 Qwen | Dense | 72B / 72B | 全面标配 QK-Norm 训练稳定架构，128k 极强文本与数学综合表现，企业私有化部署最稳基座 |

---

### 2.2 小型端侧轻量 LLM（≤4B）
* 权威评测源：**[AA Tiny Models](https://artificialanalysis.ai/models/open-source/tiny)**
* 重点关注：≤4B 极小体积下的逻辑密度、移动端/单卡运行效率 (RAM/VRAM < 6GB)

| 排名梯队 | 模型名称 | 机构 | 参数量 | 上下文 | 关键优势与技术亮点 |
|:---:|---|---|---|---|---|
| 👑 **SOTA** | **Qwen 2.5 3B / 1.5B** | 阿里 Qwen | 3.09B / 1.54B | 32k/128k | 3B 尺寸具备超越老一代 13B 稠密模型的代码和逻辑能力，支持 GQA 与高压缩词表 |
| 🥈 **Top Tier** | **Llama 3.2 3B / 1B** | Meta AI | 3.21B / 1.23B | 128k | Meta 专为移动端与本地轻量化优化的紧凑模型，配合剪枝与全量蒸馏策略 |
| 🥉 **Top Tier** | **Gemma 2 2B** | Google DeepMind | 2.6B | 8k | 继承 27B 教师模型的蒸馏成果，带滑动窗口注意力与 Logit Soft-capping 约束 |

---

### 2.3 代码生成与动态编程 (Code Generation)
* 权威评测源：**[LiveCodeBench Leaderboard](https://livecodebench.github.io/leaderboard.html)**
* 重点关注：**动态无污染新题**执行正确率 (Pass@1, 严格防训练集泄露)

| 排名梯队 | 模型名称 | 机构 | 参数量 | Pass@1 表现 | 架构亮点 |
|:---:|---|---|---|---|---|
| 👑 **开源 SOTA** | **Qwen 2.5 Coder 32B** | 阿里 Qwen | 32B Dense | **65%+** (超越 GPT-4o 早期基准) | 专为代码和算法精调，支持 128k 超大项目文件，32B 尺寸即可在常规 GPU 上达到顶级编程表现 |
| 🥈 **Top Tier** | **DeepSeek-Coder-V2** | DeepSeek | 21B/236B MoE | **60%+** | 基于 DeepSeekMoE + MLA 架构，支持 338 种编程语言与 128k 上下文 |
| 🥉 **轻量王者** | **Qwen 2.5 Coder 7B** | 阿里 Qwen | 7.6B Dense | **50%+** | 7B 小尺寸中编程性能最高，开发者日常 Copilot/本地 IDE 插件首选 |

---

### 2.4 软件工程智能体 (Software Engineering Agent)
* 权威评测源：**[SWE-bench Official Leaderboard](https://www.swebench.com/)**
* 重点关注：真实大型 GitHub 仓库 Issue 的自动化代码定位、修改并一次性通过所有单元测试的比例

| 排名梯队 | 模型与 Agent 框架 | 机构 / 提交方 | 评测子集 | 解决率 (Resolved %) |
|:---:|---|---|---|---|
| 👑 **开源 SOTA** | **DeepSeek-R1 (with Agent Scaffolds)** | DeepSeek / 社区 | SWE-bench Verified | **49% ~ 55%** |
| 🥈 **Top Tier** | **Qwen 2.5 Coder 32B (via Aider / OpenCode)** | 阿里 Qwen | SWE-bench Verified | **40% ~ 43%** |
| 🥉 **Top Tier** | **Llama 3.1 405B (Agentic)** | Meta AI | SWE-bench Lite | **38% ~ 41%** |

---

### 2.5 多模态视觉语言大模型 (Vision-Language Models / VLM)
* 权威评测源：**[Open VLM Leaderboard (Hugging Face)](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)**
* 重点关注：图像多模态推理、复杂图表文档 OCR、原生分辨率缩放、细粒度目标定位 (Grounding)

| 排名梯队 | 模型名称 | 机构 | 视觉编码与基座架构 | 核心特性 |
|:---:|---|---|---|---|
| 👑 **开源 SOTA** | **Qwen 2.5-VL-72B** | 阿里 Qwen | 动态分辨率 NaViT + Qwen2.5 语言基座 | 支持任意长宽比图像原生输入与视频动态切片，高精细 OCR、几何图表分析与长视频时序问答全面拔群 |
| 🥈 **Top Tier** | **InternVL 2.5-78B** | OpenGVLab / 上海 AI 实验室 | 动态高分辨率视觉编码器 + InternLM2.5 基座 | 跨分辨率切片拼接机制，多模态综合学术榜单常年前三 |
| 🥉 **端侧王座** | **MiniCPM-V 2.6 (8B)** | 面壁智能 (OpenBMB) | 端侧 8B 高效统一架构 | 8B 尺寸在 OCR、图表理解上越级战胜 70B 老模型，单张 3090/4090 或 iPad 端侧流畅运行 |
| 🏅 **统一多模态** | **Janus-Pro-7B** | 深度求索 (DeepSeek) | 解耦视觉编码与生成解码器 | 既支持多模态图像理解，又支持文生图，架构统一度极高 |

---

### 2.6 视频理解与时序推理 (Video-LLM)
* 权威评测源：**[OpenCompass Multi-modal Video Benchmarks](https://huggingface.co/opencompass)**
* 重点关注：多帧长时序记忆、时序事件定位 (Temporal Localization)、动作因果关系推断

| 排名梯队 | 模型名称 | 机构 | 架构机制 | 典型能力 |
|:---:|---|---|---|---|
| 👑 **SOTA** | **Qwen 2.5-VL (72B/7B)** | 阿里 Qwen | 3D-RoPE 时空动态位置编码 | 原生将视频帧视为连续三维数据流，支持数小时长视频精准时间戳检索 |
| 🥈 **Top Tier** | **MiniCPM-V 2.6** | 面壁智能 | 密集帧采样与时序池化 (Temporal Pooling) | 8B 参数支持长视频流式理解与实时视频问答 |

---

### 2.7 文生图模型 (Text-to-Image Generation)
* 权威评测源：**[AA Image Arena (Artificial Analysis)](https://artificialanalysis.ai/embed/text-to-image-leaderboard/leaderboard/text-to-image)**
* 重点关注：人类偏好 Elo 对齐分、复杂长 Prompt 指令遵循度、文字排版渲染能力、写实真实度

| 排名梯队 | 模型名称 | 机构 | 模型架构 | 参数规模 | 关键优势 |
|:---:|---|---|---|---|---|
| 👑 **开源绝对王座** | **FLUX.1 [dev]** | Black Forest Labs (原 SD 核心团队) | 12B 整流流匹配 (Rectified Flow DiT) | **12B** | 人类偏好盲测超越所有传统扩散模型；照片级真实皮肤纹理与光影；完美支持复杂英文字符排版 |
| 🥈 **极致速度** | **FLUX.1 [schnell]** | Black Forest Labs | 步数蒸馏 DiT (1~4 步极速生成) | **12B** | Apache 2.0 协议，4 步即可生成商用级画质，单次生成延迟低于 1 秒 |
| 🥉 **传统架构旗舰** | **SD 3.5 Large (8B)** | Stability AI | MMDiT (Multimodal Diffusion Transformer) | **8B** | 改进的解耦文本与图像双流注意力，艺术风格多样性好 |

---

### 2.8 视频生成模型 (Text-to-Video & Image-to-Video)
* 权威评测源：**[AA Video Arena](https://artificialanalysis.ai/embed/text-to-video-leaderboard/leaderboard/text-to-video)** / **[VBench Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard)**
* 重点关注：物理规律仿真度 (Fluid/Gravity)、前后帧时序一致性 (Temporal Consistency)、镜头运动幅度与画质

| 排名梯队 | 模型名称 | 机构 | 核心架构 | 参数规模 | 关键优势 |
|:---:|---|---|---|---|---|
| 👑 **最新开源双雄** | **Wan2.1 (通义万相)** | 阿里巴巴 (Wan-AI) | 3D 因果 VAE + Flow-Matching DiT | **14B / 1.3B** | 2025-2026 年度开源视频生成标杆；14B 物理真实感与运镜逼近 Sora 级表现；1.3B 亲民小模型支持消费级显卡极速生成 |
| 👑 **电影级画质 SOTA**| **HunyuanVideo** | 腾讯混元 (Tencent) | 双流与单流融合 DiT (Dual-stream to Single-stream) | **13B** | 全球首个完全开源的顶级影视级 DiT 视频生成架构；超大 3D 因果卷积压缩比，支持原生 720p/1080p |
| 🥉 **高性价比先驱** | **CogVideoX-5B** | 智谱 AI / 清华大学 | 3D-VAE + 专家 Transformer | **5B** | 开创开源视频 DiT 先河，社区生态适配极其完备 (ComfyUI / Diffusers) |

---

### 2.9 语音识别与音频理解 (Automatic Speech Recognition - ASR)
* 权威评测源：**[Open ASR Leaderboard (Hugging Face)](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)**
* 重点关注：词错误率 (WER)、多语种通用泛化、环境噪音抗干扰与实时推断因子 (RTF)

| 排名梯队 | 模型名称 | 机构 | 架构特性 | 核心表现 |
|:---:|---|---|---|---|
| 👑 **工业界事实标准** | **Whisper Large-v3 Turbo** | OpenAI (开源权重) | 剪枝加速 Transformer Encoder-Decoder | 参数缩减至原版近一半（809M），推理速度提升 4~8 倍，保持极佳的多语言通用 WER |
| 🥈 **多模态语音王者** | **SenseVoice-Small** | 阿里开源 (FunAudioLLM) | 非自回归流式语音模型 | 音频推理速度极快（延迟数十毫秒），并具备丰富的情绪检测 (Emotion) 与声音事件检测能力 |
| 🥉 **多语种极低 WER** | **Canary-1B** | NVIDIA NeMo | 混合 Conformer-Transducer 架构 | 英语与多欧洲主流语言 WER 处于顶尖水平 |

---

### 2.10 向量表示与语义重排 (Embedding & Reranker)
* 权威评测源：**[MTEB (Massive Text Embedding Benchmark)](https://leaderboard.mteb.org/)**
* 重点关注：海量多任务检索 (Retrieval)、分类、聚类、重排得分，向量维度与显存开销

| 排名梯队 | 模型名称 | 机构 | 类型 | 核心指标与亮点 |
|:---:|---|---|---|---|
| 👑 **通用检索王者** | **BGE-M3** | 智源研究院 (BAAI) | 统一多功能 Embedding | 单模型支持密集向量 (Dense)、稀疏词袋 (Lexical Sparse)、多向量交互 (Multi-vector ColBERT) 三合一，原生 8192 上下文与 100+ 语言 |
| 👑 **MTEB 绝对高分** | **NV-Embed-v2** | NVIDIA | 大语言模型级 Embedding | 基于 Mistral/LLaMA 基座微调，MTEB 全球榜单综合前二，高精度企业级知识库召回首选 |
| 🥈 **重排 SOTA** | **bge-reranker-large / v2-m3** | 智源研究院 (BAAI) | 交叉注意力 (Cross-Encoder) Reranker | 配合 BGE-M3 两阶段召回，Top-K 精排序准度极高 |
| 🥉 **百亿长文本向量**| **gte-Qwen2-7B-instruct** | 阿里巴巴 | LLM-based Embedding | 支持 32k 超长文档单次向量化，检索深度语义匹配出众 |

---

## 3. 全模态开源模型选型总结指南

```
【文本逻辑与工程】
  ├── 极致智商 / 复杂推理 / 论文推导 ──► DeepSeek-R1 (671B MoE)
  ├── 综合业务对话 / 企业通用底座    ──► Qwen 2.5 72B / Llama 3.3 70B
  ├── 真实项目代码修复 / Copilot     ──► Qwen 2.5 Coder 32B / 7B
  └── 端侧设备 / 树莓派 / 本地轻量    ──► Qwen 2.5 3B / Llama 3.2 3B

【多模态感知 (视觉 / 视频 / 语音)】
  ├── 图像高精 OCR / 几何推理 / 视频QA──► Qwen 2.5-VL 72B / MiniCPM-V 2.6 (8B)
  └── 语音转写 / 音视频会议纪要      ──► Whisper Large-v3 Turbo / SenseVoice

【多模态创作生成】
  ├── 商业级写实文生图 / 字符海报    ──► FLUX.1 [dev] / FLUX.1 [schnell]
  └── 电影级长镜头运镜 / 物理仿真视频──► Wan2.1-14B (通义万相) / HunyuanVideo

【知识库与 RAG 检索】
  ├── 检索阶段 (Bi-Encoder)          ──► BGE-M3 / NV-Embed-v2
  └── 精排阶段 (Cross-Encoder)       ──► BGE-Reranker-large
```
