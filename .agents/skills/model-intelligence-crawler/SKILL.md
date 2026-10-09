---
name: model-intelligence-crawler
description: 专门用于在大模型全模态前沿生态中搜寻、抓取、交叉核验与维护最新模型信息与权威排行榜的技能规范。具备自我健康检查与动态更迭机制，能够自动甄别有效/可信站点，剔除失效、停更或过时的榜单，并将最新 SOTA 模型情报沉淀入知识库。
---

# 大模型情报检索与权威排行榜跟踪规范 (Model Intelligence & Leaderboard Crawler)

本技能定义了如何高效、精准、可信地搜寻与跟踪全球开源及前沿大模型信息，并实现权威评测站点的**自动维护、健康验证与自我动态更新**。

---

## 1. 核心目标与原则

1. **多模态全景覆盖**：涵盖通用 LLM、推理模型、端侧轻量小模型（≤4B）、代码工程 Agent、多模态 VLM、视频理解、图像/视频生成、ASR 语音识别、Embedding/Reranker 等全方位赛道。
2. **权威性与防污染原则**：优先采纳具备**真实执行（Execution-based）、动态对抗防污染（LiveCodeBench）、人类双盲对齐（Arena Elo）、真实工业 issue 验证（SWE-bench）**机制的排行榜；警惕仅凭静态学术提问（易发生训练集污染）的榜单。
3. **自我演进与健康维护机制**：
   - 定期使用内建探测器进行可用性与活跃度扫描。
   - **收录标准**：数据在 6 个月内持续更新、访问稳定、社区公信力强、透明度高。
   - **剔除标准**：连续多次 HTTP 探测失败（404/410/连接超时）、超过 12 个月停止纳入新模型、被社区证实存在严重刷榜刷分且未治理。

---

## 2. 权威榜单基准知识库 (Authoritative Registry)

| 模型类别 / 赛道 | 首选排行榜 | 重点关注维度与指标 | 权威入口 URL |
|---|---|---|---|
| **通用 LLM / 推理模型** | **AA Open Source** | 综合智能能力、质量评分、生成吞吐、成本与开源协议 | [https://artificialanalysis.ai/models/open-source](https://artificialanalysis.ai/models/open-source) |
| **小型 LLM（≤4B）** | **AA Tiny Models** | 端侧 ≤4B 轻量模型智能密度、显存占用与推理速度 | [https://artificialanalysis.ai/models/open-source/tiny](https://artificialanalysis.ai/models/open-source/tiny) |
| **代码生成与编程** | **LiveCodeBench** | 动态对抗 LeetCode/Codeforces 新题执行正确率 (Pass@1) | [https://livecodebench.github.io/leaderboard.html](https://livecodebench.github.io/leaderboard.html) |
| **软件工程 Agent** | **SWE-bench** | 真实 GitHub Issue 缺陷全自动化修复率 (Verified / Pro) | [https://www.swebench.com/](https://www.swebench.com/) |
| **多模态 / VLM** | **Open VLM Leaderboard** | 图像理解、多图推理、复杂高分辨率 OCR、视觉数学逻辑 | [https://huggingface.co/spaces/opencompass/open_vlm_leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) |
| **视频理解模型** | **OpenCompass** | 视频问答 (Video-QA)、多帧时序因果理解、长视频长程记忆 | [https://huggingface.co/opencompass](https://huggingface.co/opencompass) |
| **图像生成 (T2I)** | **AA Image Arena** | 文生图生成质量、人类盲测偏好 Elo、语义遵循与构图美学 | [https://artificialanalysis.ai/embed/text-to-image-leaderboard/leaderboard/text-to-image](https://artificialanalysis.ai/embed/text-to-image-leaderboard/leaderboard/text-to-image) |
| **视频生成 (T2V)** | **AA Video Arena** | 文生视频/图生视频画质、人类偏好 Elo、物理规律连续性 | [https://artificialanalysis.ai/embed/text-to-video-leaderboard/leaderboard/text-to-video](https://artificialanalysis.ai/embed/text-to-video-leaderboard/leaderboard/text-to-video) |
| **视频生成客观指标** | **VBench** | 16 个细分客观维度（视频时序一致性、动态运镜、主体忠实度等） | [https://huggingface.co/spaces/Vchitect/VBench_Leaderboard](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) |
| **语音识别 ASR** | **Open ASR** | 单词错误率 (WER)、多语种识别能力、噪音抗干扰鲁棒性 | [https://huggingface.co/spaces/hf-audio/open_asr_leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) |
| **Embedding / Reranker**| **MTEB** | 跨语言海量向量检索召回、重排相关性、显存维度与尺寸权衡 | [https://leaderboard.mteb.org/](https://leaderboard.mteb.org/) |

---

## 3. 榜单生命周期与自我更新流程 (Self-Updating Protocol)

当 Agent 启动前沿模型情报搜集或维护任务时，执行以下三步闭环：

```text
[触发情报搜寻 / 榜单更新]
          │
          ▼
[步骤 1: 自动化健康探活] ───► 运行 python3 scripts/benchmark_prober.py
          │
          ├─► 存在 HTTP 404 / 5xx / 废弃 ──► 标记为待剔除 / 寻找社区最新替代
          │
          ▼
[步骤 2: 数据时效性与信度审查]
          │ 检查最近更新时间是否停滞 (>12 个月)
          │ 检查是否有新设立的高信度权威榜单 (如行业最新评测空间)
          │
          ▼
[步骤 3: 动态更新 Skill 知识库]
          │ 同步更新 sources.json 与 SKILL.md
          │ 更新根目录 README.md / comparisons 报告
```

### 3.1 运行健康探活脚本
```bash
python3 .agents/skills/model-intelligence-crawler/scripts/benchmark_prober.py
```
若出现任何状态码异常，脚本将直接报警并输出时延与错误报文，指导 Agent 进行 URL 修订或标记淘汰。

### 3.2 发现并增补新榜单准则
当社区出现全新模型模态（例如 3D 生成、具身智能 Embodied AI、实时语音交互 Omnimodal）时：
1. 确认该评测源是否具备开源透明的 Harness 或具有广泛行业背书；
2. 在 `sources.json` 中追加该源对象，包含 `id`, `category`, `name`, `focus`, `url`, `evaluation_type`；
3. 同步将最新行更新入 `SKILL.md` 的表格。

### 3.3 剔除失效与低信度榜单准则
- **永久失效**：站点停止服务超过 30 天，无镜像可用；
- **过度污染**：若某一基准已被大模型厂商通过训练集过度过拟合而丧失横向对比意义（如早期未经防泄漏处理的初级单选题库），应主动降级其权重或移除。

---

## 4. 获取具体模型架构参数的标准作业流程 (SOP)

当在权威榜单上发现新晋 SOTA 模型后，按以下链路获取其第一手权威架构数据：

1. **锁定官方 Hugging Face 仓库**：
   - 获取官方卡片与配置：`https://huggingface.co/<org>/<model>/raw/main/config.json`
2. **提取核心张量拓扑参数**：
   - 提取 `model_type`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`；
   - 检查是否为稀疏 MoE：`n_routed_experts`, `num_experts_per_tok`, `n_shared_experts`；
   - 检查注意力机制：是否为 MLA (`kv_lora_rank`, `qk_rope_head_dim`) 或带 `sliding_window`；
   - 检查归一化与稳定性：是否开启 `q_norm`/`k_norm` (QK-Norm) 或 soft-capping。
3. **沉淀入仓库**：
   - 按照 `[开源公司]/[系列]/[版本]/` 规范归档入 `llm-architectures/` 对应目录。
