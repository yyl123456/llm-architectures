# Hyper-Connections (HC) 4 路残差超连接机制

> 源码锚点：`inference/model.py:Block`, `inference/model.py:hc_split_sinkhorn`

---

## 1. 传统残差流 vs 超连接 (HC) 拓扑演进

```text
传统 Transformer 残差流:
  x = x + Sublayer(RMSNorm(x))  ──► 仅有一路 [batch, seq_len, dim] 向量累加

DeepSeek-V4.1 的 4 路并行超连接 (hc_mult = 4):
  stream = [batch, seq_len, 4, 5120]  (4 个通道并行流淌)
             │
             ├──► hc_mixes(): 计算 20 轮 Sinkhorn 双随机转移矩阵
             ├──► hc_pre(): 塌缩 4 路残差为单路 [b, s, 5120] 作为 Attn/FFN 输入
             ├──► 前向计算 Sublayer (Attention 或 MoE)
             └──► hc_post(): 将单路输出与 4 路残差通过转移矩阵重投影，恢复为 4 路流
```

---

## 2. 关键工程收益

1. **彻底消除梯度饱和**：为 40 层深层网络提供了 4 倍于传统的梯度反传通道，避免了极深 MoE 网络的表征退化；
2. **残差特征去污染**：浅层语言信息与深层逻辑可以在不同的并行残差通道中独立保存，无需强行挤入同一条一维向量中。
