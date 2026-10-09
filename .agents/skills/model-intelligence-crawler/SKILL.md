---
name: model-intelligence-crawler
description: 专门用于在大模型全模态前沿生态中搜寻、抓取、交叉核验与维护最新模型信息与权威排行榜的技能规范。具备抗动态渲染穿透机制（Jina Reader / Next.js chunks / 官方 API 逆向）与硬性时间窗口过滤器（强制排除 2026 年 5 月前的一切过时模型）。
---

# 大模型情报检索与动态排行榜自更新规范 (Model Intelligence & Crawler Protocol)

本技能定义了如何高效、精准、可信地搜寻与跟踪全球开源及前沿大模型信息，并实现权威评测站点的**自动维护、健康验证、抗渲染数据穿透与强时效性自我更新**。

---

## 1. 核心铁律与时效性门禁 (Time Gate & Quality Rules)

### 1.1 强制硬性时间过滤 (The Strict Post-2026-05 Rule)
- ⚠️ **时间铁律**：所有录入与展示的开源模型**必须标明发布时间 (Release Date)**。
- ❌ **绝对淘汰标准**：凡是**发布时间早于 2026 年 5 月 1 日（< 2026-05-01）的模型全部 PASS 剔除**，严禁作为当前 SOTA 推荐给用户（例如 2024 年的 LLaMA-3、旧版 Whisper、早期 FLUX.1、Qwen2.5 早期版等已属历史代际，全部淘汰归档）。
- 唯一的保留标准是当且仅当该赛道官方榜单近期完全无更新时，须明确标注 `[该赛道近期无 >= 2026-05 新模型发布，标记需寻找新榜单]`，坚决杜绝“拿旧充新”或“脑补伪造”。

### 1.2 穿透动态客户端渲染的三大工具化解法 (Anti-Rendering Solutions)
现代 AI 排行榜（如 Artificial Analysis、LiveCodeBench、Hugging Face Spaces、VBench）普遍采用 Next.js App Router、React SPA 或 Gradio 异步加载，直接使用普通 `urllib` 或 `curl` 会遭遇空白页面或 `Loading...` 假死。
**严禁因为抓取遇到障碍就退回使用大模型先验记忆或编造假数据**。必须按以下三步梯度工具化解决：

1. **方案 A：利用 Jina Reader 代理穿透客户端渲染**：
   - 规则：在目标 URL 前增加 `https://r.jina.ai/`。
   - 效果：Jina Reader 在服务端自动完成 Headless Chromium 渲染、等待异步网络请求并输出完整格式化的 Markdown 文本。
   - 示例：`curl -sL "https://r.jina.ai/https://www.swebench.com/"`
2. **方案 B：逆向 Next.js App Router Chunks**：
   - 规则：对于 Next.js 页面（如 Artificial Analysis），数据被序列化存储在内嵌脚本 `self.__next_f.push(...)` 中。
   - 效果：通过 Python 正则清洗转义字符 `\"`，直接提取包含 `releaseDate`、`intelligenceIndex`、`openSourceCategorization` 的原始 JSON 字典，获得 100% 官方底层数值。
3. **方案 C：直连 Hugging Face Hub 官方 API 与底层 CSV/JSON 资产**：
   - 规则：调用 `https://huggingface.co/api/models/<model_id>` 验证 `createdAt` 时间戳；读取 Space 绑定的原始提交文件（如 `english_short_latest.csv`）。

---

## 2. 11 大赛道权威榜单注册与活跃状态 (Authoritative Sources Registry)

| 模型类别 / 赛道 | 首选权威排行榜 | 真实底层数据源 / 接入方式 | 重点关注指标 |
|---|---|---|---|
| **1. 通用 LLM / 推理模型** | [AA Open Source](https://artificialanalysis.ai/models/open-source) | Next.js chunks (`intelligenceIndex`, `releaseDate`) | 综合质量跑分、细粒度 MoE 激活开销 |
| **2. 小型 LLM（≤4B）** | [AA Tiny Models](https://artificialanalysis.ai/models/open-source/tiny) | Next.js chunks (端侧 ≤4B 专项) | 极小参数逻辑密度、端侧推理延迟 |
| **3. 代码生成与编程** | [LiveCodeBench](https://livecodebench.github.io/leaderboard.html) | `performances_generation.json` 全量评测数据 | 动态防污染新题 Pass@1 执行均分 |
| **4. 软件工程 Agent** | [SWE-bench](https://www.swebench.com/) | Jina Reader 穿透渲染表格数据 | Verified / Pro 解决率、真实 Issue 修复 |
| **5. 多模态 / VLM** | [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) | `http://opencompass.openxlab.space/assets/OpenVLM.json` | MMMU_VAL、多图推理、动态 OCR |
| **6. 视频理解模型** | [OpenCompass](https://huggingface.co/opencompass) | OpenCompass 多模态/视频系列评测套件 | 3D 时空建模、长视频时序问答 |
| **7. 文生图 (Text-to-Image)**| [AA Image Arena](https://artificialanalysis.ai/embed/text-to-image-leaderboard/leaderboard/text-to-image) | `leaderboard/text-to-image` chunks (`elo`, `openWeightsUrl`) | 人类盲测 Elo Rating、复杂排版 |
| **8. 文生视频 (Text-to-Video)**| [AA Video Arena](https://artificialanalysis.ai/embed/text-to-video-leaderboard/leaderboard/text-to-video) | `leaderboard/text-to-video` chunks (`elo`, `openWeightsUrl`) | 人类盲测 Elo、真实物理模拟 |
| **9. 视频生成客观指标** | [VBench](https://huggingface.co/spaces/Vchitect/VBench_Leaderboard) | `results.csv` / `vbench_leaderboard_submission` | 16 维客观时序/主体一致性 |
| **10. 语音识别 ASR** | [Open ASR](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) | `hf-audio/open-asr-leaderboard-results/english_short_latest.csv` | 真实短文本 WER (词错误率)、RTFx 吞吐 |
| **11. Embedding / Reranker** | [MTEB](https://leaderboard.mteb.org/) | `spaces/mteb/leaderboard/raw/main/models.py` | 检索召回、长文本向量化、思考型向量 |

---

## 3. 标准化执行脚本与自检工具

本技能内建了自动化检测与时间过滤工具：

### 3.1 运行严时限 (>= 2026-05) 实时抓取器
```bash
python3 .agents/skills/model-intelligence-crawler/scripts/fetch_latest_sota.py
```
该脚本自动执行以下操作：
1. 连线各个站点，逆向提取最新数据；
2. **强制剔除所有 `< 2026-05-01` 的老模型**；
3. 输出经过时间戳审查的合法模型清单与指标。

### 3.2 运行网络健康与连通性探活
```bash
python3 .agents/skills/model-intelligence-crawler/scripts/benchmark_prober.py
```

---

## 4. 榜单动态演进与自我更新协议 (Self-Updating Protocol)

当 Agent 或协作者发现以下情况时，必须更新本技能：
1. **站点失效或改版**：若某一 URL 连续 3 次返回 404 或前端结构重构，必须用 Jina Reader 探寻新入口，并更新 `sources.json` 与 `SKILL.md`；
2. **基准防刷机制失效**：若某一榜单不再更新动态题目，导致严重过拟合，需降级该榜单并引入新基准；
3. **硬时间窗口推进**：随着时间推移，不断将 `CUTOFF_DATE` 向前推进，确保知识库永远保持在最新的前沿水平。
