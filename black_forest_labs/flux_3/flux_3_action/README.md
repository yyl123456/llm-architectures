# FLUX.3 Action 系列架构预研卡片

---

## 1. 基本概览

| 属性 | 说明 / 指标 |
|---|---|
| **模型名称** | FLUX-3-Action-Base / SO101 / Droid |
| **开源机构** | Black Forest Labs (BFL) |
| **发布时间** | **2026-09-22** |
| **模型定位** | 具身智能动作生成与视觉物理交互扩散模型 (Vision-Action DiT) |
| **开源状态** | 开放权重 (Open Weights) |
| **官方代码库** | [black-forest-labs/flux-3-action-base](https://huggingface.co/black-forest-labs/flux-3-action-base) |

---

## 2. 核心架构亮点与算法突破

1. **从纯画面生成向“具身动作生成 (Action Generation)”跨越**：
   - BFL 将 FLUX 顶尖的 Rectified Flow DiT 架构引入机器人与具身空间控制；
   - 能够同时输出高拟真连续动作轨迹预测与物理交互因果视觉渲染。
2. **沉淀研究方向**：
   - 动作 Token 与潜在图像特征联合去噪的 MMDiT 架构剖析。
