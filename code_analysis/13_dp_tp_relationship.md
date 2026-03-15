# DP 与 TP 的关系：从原理到工业实例

> **核心问题**：DP 和 TP 都是"多卡"方案，它们的本质区别是什么？
> DP=2、TP=4 时，8 张 GPU 分别在做什么？数据怎么流动？

---

## 一、先说清楚本质区别

### 1.1 一句话区分

```
TP（张量并行）：切分模型权重
  → 一份数据，由多卡协作完成一次矩阵乘法
  → 多卡"合力"处理同一批 token

DP（数据并行）：切分输入数据
  → 多份数据，每份交给一个独立的模型副本
  → 多卡"分头"处理不同批次的 token
```

**打个比方**：

```
任务：批改 100 份试卷（= 处理 100 条 sequences）

DP：你有 2 个老师，每人批改 50 份（各自独立，互不干扰）
    批改完汇总"哪些题型大家都错了"（梯度 AllReduce）

TP：你有一份试卷（= 一批 token），4 个老师同时批改这份卷子的不同部分
    每人负责某些题目（= 权重的不同分片）
    批完后把分数加起来（AllReduce）才是完整成绩

DP + TP：2 组老师，每组 4 人；
    组1批前 50 份（每组内 4 人 TP 协作）
    组2批后 50 份（每组内 4 人 TP 协作）
    最后两组交流"哪些题型都错了"（DP 梯度同步）
```

---

### 1.2 关键对比表

| 维度 | TP | DP |
|------|----|----|
| **切分对象** | 模型权重（W 的行/列）| 输入数据（batch 按样本切分）|
| **每卡持有** | 权重的 1/TP | 完整模型（或 ZeRO 后的分片）|
| **每卡输入** | **完整的** token batch | **不同的** token batch |
| **通信时机** | 每层内（AllReduce，阻塞）| 每步结束（AllReduce，可重叠）|
| **通信量/步** | 2 × layers × 2 × B × H | 2 × 模型参数量 |
| **通信对延迟的影响** | 直接增加单步延迟（阻塞）| 可与计算重叠（隐藏）|
| **适用硬件** | 节点内 NVLink（高带宽）| 节点间 InfiniBand（可接受）|
| **解决的问题** | 单卡显存不够放一层权重 | 训练吞吐不够（需要更多算力）|

---

## 二、DP=2 TP=4：8 张 GPU 的完整布局

### 2.1 GPU 分组

```
总共 8 张 GPU（例如：2 台服务器，每台 4 张 A100）

物理布局：
  ┌──────────────────────────────────────────────────────────────────────────┐
  │  机器 0（节点 0）                  机器 1（节点 1）                     │
  │  ┌────────────────────┐            ┌────────────────────┐               │
  │  │ GPU0  GPU1  GPU2  GPU3 │         │ GPU4  GPU5  GPU6  GPU7 │            │
  │  │  NVLink（~600GB/s）   │         │   NVLink（~600GB/s）   │            │
  │  └────────────────────┘            └────────────────────┘               │
  │         节点间 InfiniBand（~25GB/s）                                    │
  └──────────────────────────────────────────────────────────────────────────┘

逻辑分组：
  TP Group 0：[GPU 0, GPU 1, GPU 2, GPU 3]  ← 机器 0 的 4 卡，NVLink 互联
  TP Group 1：[GPU 4, GPU 5, GPU 6, GPU 7]  ← 机器 1 的 4 卡，NVLink 互联

  DP Group 0（相同 TP rank 0 的卡）：[GPU 0, GPU 4]  ← 跨机器，IB 互联
  DP Group 1（相同 TP rank 1 的卡）：[GPU 1, GPU 5]  ← 跨机器，IB 互联
  DP Group 2（相同 TP rank 2 的卡）：[GPU 2, GPU 6]
  DP Group 3（相同 TP rank 3 的卡）：[GPU 3, GPU 7]
```

用矩阵表示 rank 布局（代码中的 `all_ranks[dp][tp]`）：

```
          TP rank 0   TP rank 1   TP rank 2   TP rank 3
DP rank 0:  GPU 0       GPU 1       GPU 2       GPU 3
DP rank 1:  GPU 4       GPU 5       GPU 6       GPU 7

读法：
  横向（同一行）：同一个 DP rank，做 TP 协作 → 是一个 TP Group
  纵向（同一列）：同一个 TP rank，做 DP 梯度同步 → 是一个 DP Group
```

---

### 2.2 每张 GPU 持有什么权重？

以 LLaMA-3-8B（hidden=4096，FFN=14336，32层）为例：

```
完整模型（单卡）的主要权重：
  q_proj:       [4096, 4096]  →  TP=4 切分后：[1024, 4096]（行切分，每卡 1/4 头）
  k_proj:       [1024, 4096]  →  TP=4 切分后：[256,  4096]（GQA，每卡 2 个 KV 头）
  v_proj:       [1024, 4096]  →  TP=4 切分后：[256,  4096]
  o_proj:       [4096, 4096]  →  TP=4 切分后：[4096, 1024]（列切分）
  gate_proj:    [14336,4096]  →  TP=4 切分后：[3584, 4096]
  up_proj:      [14336,4096]  →  TP=4 切分后：[3584, 4096]
  down_proj:    [4096,14336]  →  TP=4 切分后：[4096, 3584]

机器 0（DP rank 0 的 TP group）：
  GPU 0（TP rank 0）：持有所有层的 q_proj[0:1024,:], gate_proj[0:3584,:], ...
  GPU 1（TP rank 1）：持有所有层的 q_proj[1024:2048,:], gate_proj[3584:7168,:], ...
  GPU 2（TP rank 2）：持有所有层的 q_proj[2048:3072,:], gate_proj[7168:10752,:], ...
  GPU 3（TP rank 3）：持有所有层的 q_proj[3072:4096,:], gate_proj[10752:14336,:], ...

机器 1（DP rank 1 的 TP group）：
  GPU 4 ~ GPU 7：持有与 GPU 0 ~ GPU 3 **完全相同形状、完全相同数值**的权重分片！
  （DP 的核心：每个 DP rank 有完整的模型副本，只是切成 TP 分片后数值相同）
```

---

### 2.3 输入数据如何划分

```
Global Batch（一个 Step 的全部数据）：
  128 条 sequences，每条 2048 tokens

DP 切分（按样本切分）：
  DP rank 0（机器 0，GPU 0-3）：处理 sequences [0:64]   ← Batch A
  DP rank 1（机器 1，GPU 4-7）：处理 sequences [64:128] ← Batch B

注意！TP 不切分数据：
  TP Group 0（GPU 0-3）的 4 张卡同时处理相同的 Batch A（64 条 sequences）
  每张卡看到的输入 X 形状：[64, 2048, 4096]  ← 完全相同！
  但每张卡对 X 做的矩阵乘法，用的是权重的不同分片
```

---

## 三、完整数据流：DP=2 TP=4 的一次 Step

### 3.1 前向传播（以一个 Decoder Layer 为例）

```
初始状态：
  GPU 0-3（Batch A）：X_A = [64, 2048, 4096]（完整激活，每卡都一样）
  GPU 4-7（Batch B）：X_B = [64, 2048, 4096]（完整激活，每卡都一样）

  （以下只展示 GPU 0-3 对 Batch A 的处理；GPU 4-7 完全对称，并行进行）

─── Step 1：QKV 投影（列并行，无通信）───────────────────────────────

  GPU 0：Q_0 = X_A @ q_proj[0:1024,:]ᵀ    → [64, 2048, 1024]（head 0-7）
         K_0 = X_A @ k_proj[0:256, :]ᵀ    → [64, 2048, 256] （KV head 0-1）
         V_0 = X_A @ v_proj[0:256, :]ᵀ    → [64, 2048, 256]

  GPU 1：Q_1 = X_A @ q_proj[1024:2048,:]ᵀ → [64, 2048, 1024]（head 8-15）
         K_1 = X_A @ k_proj[256:512, :]ᵀ  → [64, 2048, 256] （KV head 2-3）
         V_1 = ...

  GPU 2、3 类似，负责各自的 head 分片

  ★ 无通信！因为是列并行（每卡独立输出不同列）

─── Step 2：Flash Attention（各卡独立计算自己负责的头）──────────────

  GPU 0：对 head 0-7 独立计算 Attention → attn_out_0: [64, 2048, 1024]
  GPU 1：对 head 8-15 独立计算 Attention → attn_out_1: [64, 2048, 1024]
  GPU 2、3 类似

  ★ 无通信！每个头的注意力计算只依赖自己的 Q/K/V

─── Step 3：O Proj（行并行）→ AllReduce ────────────────────────────

  GPU 0：P_0 = attn_out_0 @ o_proj[:,0:1024]  → [64, 2048, 4096]（部分和）
  GPU 1：P_1 = attn_out_1 @ o_proj[:,1024:2048] → [64, 2048, 4096]（部分和）
  GPU 2：P_2 = ...
  GPU 3：P_3 = ...

  ★ AllReduce（NVLink，~0.05ms）：
    X_out = P_0 + P_1 + P_2 + P_3 → [64, 2048, 4096]（完整激活）
    通信只在 GPU 0-3 内部！不涉及 GPU 4-7

  + 残差：X = X_A + X_out → [64, 2048, 4096]（每卡持有完整副本）

─── Step 4：Gate/Up Proj（列并行，无通信）──────────────────────────

  GPU 0：gate_0 = X @ gate_proj[0:3584,:]ᵀ   → [64, 2048, 3584]
         up_0   = X @ up_proj[0:3584,:]ᵀ     → [64, 2048, 3584]
         h_0    = SiLU(gate_0) ⊙ up_0         → [64, 2048, 3584]

  GPU 1-3 类似，各自的 FFN 维度分片

─── Step 5：Down Proj（行并行）→ AllReduce ─────────────────────────

  GPU 0：P_0 = h_0 @ down_proj[:,0:3584]  → [64, 2048, 4096]（部分和）
  GPU 1：P_1 = h_1 @ down_proj[:,3584:7168] → ...
  GPU 2：P_2 = ...
  GPU 3：P_3 = ...

  ★ AllReduce（NVLink，~0.05ms）：
    FFN_out = P_0 + P_1 + P_2 + P_3 → [64, 2048, 4096]（完整激活）

  + 残差：X = X + FFN_out → [64, 2048, 4096]（进入下一层）

─── 重复 32 层 ─────────────────────────────────────────────────────

  最终输出：
  GPU 0-3 各自持有 Batch A 的 logits: [64, 2048, 32000]
  GPU 4-7 各自持有 Batch B 的 logits: [64, 2048, 32000]
```

---

### 3.2 损失计算与反向传播

```
损失计算（每组独立）：
  GPU 0-3：loss_A = cross_entropy(logits_A, labels_A)  → scalar
  GPU 4-7：loss_B = cross_entropy(logits_B, labels_B)  → scalar

反向传播（每组独立，类比前向的逆过程）：
  GPU 0-3：计算 Batch A 的梯度 ∂loss_A/∂W（每个权重分片）
  GPU 4-7：计算 Batch B 的梯度 ∂loss_B/∂W（每个权重分片）

  ★ TP 反向通信：
    与前向对称，行并行层的输入梯度也需要 AllReduce
    GPU 0-3 内部 AllReduce（NVLink）
    GPU 4-7 内部 AllReduce（NVLink）
```

---

### 3.3 DP 梯度同步（跨机器通信）

```
反向传播结束后，每个 GPU 持有对应权重分片的梯度：

  GPU 0：∂loss_A/∂q_proj[0:1024,:]  （Batch A 贡献的梯度）
  GPU 4：∂loss_B/∂q_proj[0:1024,:]  （Batch B 贡献的梯度）

  ★ DP AllReduce（DP Group 0：GPU 0 和 GPU 4，跨机器 IB，~1ms）：
    ∂loss/∂q_proj[0:1024,:] = (∂loss_A + ∂loss_B) / 2

  同样地：
  GPU 1 ↔ GPU 5（DP Group 1）：同步 q_proj[1024:2048,:] 的梯度
  GPU 2 ↔ GPU 6（DP Group 2）：同步 q_proj[2048:3072,:] 的梯度
  GPU 3 ↔ GPU 7（DP Group 3）：同步 q_proj[3072:4096,:] 的梯度

  重要：4 个 DP AllReduce 可以同时进行！（互不依赖）

同步后：
  GPU 0 和 GPU 4 持有相同的梯度 → 参数更新后权重相同（DP 一致性）
  GPU 1 和 GPU 5 持有相同的梯度 → 同上
  ...
```

---

### 3.4 参数更新

```
Adam 优化器（每卡本地执行，无通信）：
  GPU 0：用 ∂loss/∂q_proj[0:1024,:] 更新 q_proj[0:1024,:] ← 仅更新本卡的分片
  GPU 1：用 ∂loss/∂q_proj[1024:2048,:] 更新 q_proj[1024:2048,:] ← 仅更新本卡的分片
  ...
  GPU 4：用相同梯度更新相同分片（与 GPU 0 更新结果一致）
  ...

更新后：
  GPU 0 和 GPU 4 的 q_proj[0:1024,:] 完全相同  ✓（DP 一致性维持）
  进入 Step N+1，用新参数继续训练
```

---

### 3.5 完整时间线（一个 Step 的通信事件）

```
时间 →  0    2    4    6    8    10   12   14   16   18   20   22   24   26  ms
        ├────前向传播（32层）────────────────────────────────┤
        │ 每层 2次 AllReduce（TP，NVLink）                   │
        │  ↑ ~0.1ms/次，32层×2次 ≈ 6ms（与 GEMM 重叠后更少）│
        │                                                    ├──Loss──┤
        │                                                             ├────反向传播（32层）──────┤
        │                                                             │ 每层 2次 AllReduce（TP） │
        │                                                             │  ← 与 GEMM 计算重叠    │
        │                                                             │                         │
        │                                                             │    DP AllReduce（IB）   │
        │                                                             │    ← 与反向计算重叠     │
        │                                                             │                        ├──Optim──┤
                                                                                              参数更新
                                                                                              Step 结束

TP AllReduce：机器内，NVLink，~0.05ms/次，可与 GEMM 重叠（理想情况）
DP AllReduce：机器间，IB，~1ms/次，只发生 1次（与最后几层反向重叠）
```

---

## 四、工业界实例

### 实例 1：LLaMA-3-8B 微调（单机 8 卡）

**场景**：公司内部用 8 张 A100-80G 微调 8B 模型

**配置**：`TP=1, DP=8`

```
为什么 TP=1？
  8B 模型（BF16）= 16GB，单卡 80G 放得下
  → 不需要 TP！

为什么 DP=8？
  8 张卡都有完整模型副本
  → 每卡处理不同的数据，吞吐是单卡的 8 倍

GPU 布局（8卡）：
          TP rank 0（唯一）
DP rank 0:  GPU 0   ← 完整模型，处理 batch[0:micro]
DP rank 1:  GPU 1   ← 完整模型，处理 batch[micro:2*micro]
...
DP rank 7:  GPU 7   ← 完整模型，处理 batch[7*micro:8*micro]

数据流：
  完全没有 TP 通信（TP=1）
  反向结束后 AllReduce 梯度（8卡全参与）

性能特点：
  ✓ 极简，没有 TP 通信开销
  ✓ 适合 SFT/LoRA/RLHF 微调
  训练吞吐 ≈ 单卡的 7.8 倍（受 DP AllReduce 轻微影响）
```

---

### 实例 2：LLaMA-3-70B 推理（单机 8 卡）

**场景**：推理服务，用 8 张 A100-80G 部署 70B 模型

**配置**：`TP=8, DP=1`

```
为什么 TP=8？
  70B 模型（BF16）= 140GB > 80GB（单卡放不下！）
  → 必须用 TP 切分权重
  140GB / 8卡 = 17.5GB/卡 → 能放下

为什么 DP=1？
  推理只需要一份模型就够（在线推理，continuous batching）
  → 不需要多个副本

GPU 布局（8卡，单机）：
  TP Group（唯一）：[GPU 0, GPU 1, GPU 2, GPU 3, GPU 4, GPU 5, GPU 6, GPU 7]
  所有卡在 NVLink 域内（DGX A100 满足）

  每卡持有：
  q_proj: [4096/8, 8192]（70B hidden=8192）= [512, 8192]
  ...

数据流：
  所有请求的 batch（如 32 条序列）同时进入所有 8 卡
  每层 2 次 AllReduce（NVLink，~0.1ms/次）
  64层 × 2次 = 128 次 AllReduce/step

性能特点：
  ✓ 支持 70B 模型推理
  ✓ NVLink 带宽充足，TP=8 通信开销可接受
  延迟 ≈ 单卡的 1.2 倍（多了 AllReduce 开销）
  vs 如果跨机器：延迟会增加 5-10 倍
```

---

### 实例 3：LLaMA-3-70B 训练（32 台机器，256 卡）

**场景**：从头预训练 70B 模型

**配置**：`TP=8, PP=4, DP=8`

```
为什么这样配置？
  70B 模型太大，需要 3D 并行

  TP=8（机器内）：
    每台机器 8 卡做 TP，70B / 8 = 8.75GB/卡（只是激活的参数部分）
    加上激活值和 Adam 状态，需要 ~60GB/卡 → A100-80G 刚好

  PP=4（机器间）：
    80 层 / 4 Stage = 20 层/Stage
    PP 用 InfiniBand，P2P 通信量小（每 Stage 边界传一次激活）

  DP=8（机器间）：
    32 台机器 / (TP=1 machine × PP=4 stages) = 8 个 DP replica
    → 8 个相同的 TP+PP 流水线，各处理不同 batch

GPU 布局（256卡 = 32台 × 8卡）：

  DP rank 0，PP Stage 0：机器 0  [GPU  0-7]  TP group（Layer 0-19）
  DP rank 0，PP Stage 1：机器 1  [GPU  8-15] TP group（Layer 20-39）
  DP rank 0，PP Stage 2：机器 2  [GPU 16-23] TP group（Layer 40-59）
  DP rank 0，PP Stage 3：机器 3  [GPU 24-31] TP group（Layer 60-79）
  （以上 4 台机器组成 1 条 PP 流水线，是 1 个 DP replica）

  DP rank 1，PP Stage 0：机器 4  [GPU 32-39] ← 与 rank 0 相同的权重分片
  DP rank 1，PP Stage 1：机器 5  [GPU 40-47]
  ...

  共 8 条 PP 流水线，每条 4 台机器

通信格局：
  机器内（NVLink，600GB/s）：TP AllReduce（频繁，每层2次）
  机器间同 PP Stage（IB）：DP AllReduce（每步1次）
  机器间相邻 PP Stage（IB）：PP P2P（每 microbatch 1次）

  TP 通信 → 必须在机器内（NVLink）完成
  PP 通信 → 跨机器（IB），但 P2P 且可流水线化
  DP 通信 → 跨机器（IB），与反向计算重叠
```

---

### 实例 4：DeepSeek-V3 训练（2048 卡）

**场景**：DeepSeek 的 671B MoE 模型预训练

**配置**：`TP=1, PP=16, EP=64, DP=ZeRO-1`（官方技术报告）

```
关键点：TP=1！没有张量并行！

为什么 671B MoE 模型可以不用 TP？
  MoE 的激活参数是 37B（每步只有 37B 参数在计算）
  + MLA 大幅压缩 KV Cache 显存
  → 单条 PP 流水线（16 台机器）每台约处理 42B/16 ≈ 2.6B 参数
  → 每台机器 8 卡平均 2.6B/8 ≈ 325M 参数 → 单卡放得下！

  代价：
  如果用 TP，每层需要 2 次 AllReduce
  671B，61 个 Transformer 层 × 2 次 = 122 次 AllReduce/step
  每次 ~0.1ms（NVLink）→ 12ms/step 的纯 TP 通信开销
  → 对于 671B 级别的训练，12ms 是不可忽视的开销

  DeepSeek 选择：宁可不用 TP，通过架构（MLA + MoE）降低显存需求

GPU 布局（以 128 卡为例说明原理）：
  PP rank 0：  机器 0  [GPU 0-7]   ← 层 0-3（每 stage 约 4 层）
  PP rank 1：  机器 1  [GPU 8-15]  ← 层 4-7
  ...
  PP rank 15： 机器 15 [GPU120-127]← 层 60-61 + lm_head

  每台机器 8 卡做 EP（Expert Parallelism）：
  256 专家 / 64 卡（EP=64，跨 8 台机器）= 4 专家/卡

  DP：剩余的 2048/128 = 16 个 PP+EP 组做 DP
  ZeRO-1 分片优化器状态（不再 AllReduce 完整梯度，用 ReduceScatter）
```

---

### 实例 5：GPT-4 式推理服务（vLLM，多机多卡）

**场景**：提供 API 推理服务，追求吞吐与延迟的平衡

**配置（假设 100B 模型）**：`TP=4, DP=N（弹性扩展）`

```
每个推理实例：
  1台机器，4张 GPU，TP=4（机器内 NVLink）
  → 100B/4 = 25GB/卡 → 一台机器 4 卡可以服务一份模型

多实例水平扩展（DP 类似概念）：
  流量大时：启动更多推理实例（每实例 4 卡）
  10个实例 = 40张卡 = 可以并行处理 10倍的请求

vLLM 的 DP+TP 模式（最新版本）：
  --tensor-parallel-size 4    ← TP=4（单实例内）
  --data-parallel-size 2       ← DP=2（2个实例）
  → 共 8 卡，分成 2 组，各自处理不同请求

  DP Group 0（GPU 0-3）：服务 requests [A, B, C, D]
  DP Group 1（GPU 4-7）：服务 requests [E, F, G, H]

  优势：DP 不增加单请求延迟（两组完全独立）
        TP=4 保证大模型可以装下
```

---

### 实例 6：Megatron-LM 530B 训练（NVIDIA）

**场景**：NVIDIA 训练 MT-NLG 530B 模型

**配置**：`TP=8, PP=35, DP=35`，共 2240 张 A100

```
计算验证：
  8 × 35 × 35 = 9800？

  实际：TP=8, PP=35, DP=8  → 8 × 35 × 8 = 2240 ✓

GPU 布局（2240 卡）：
  每台 DGX A100（8卡）= 1 个 TP Group
  8卡 × 35 stage × 8 DP = 2240 卡 = 280 台 DGX

  每个 PP 流水线 = 35 台机器（35 个 Stage，每 Stage 8卡 TP）
  共 8 条并行流水线（DP=8）

  PP stage 划分：
  530B 参数 / 35 PP stage = 15B 参数/Stage
  15B / TP=8 ≈ 1.9B 参数/卡 × 2bytes(FP16) ≈ 3.8GB
  加上梯度/激活 → ~20-30GB/卡（A100-80G 够用）

  通信格局：
  机器内（NVLink）：TP AllReduce（频繁）← 最关键
  机器间（InfiniBand）：PP P2P + DP AllReduce
```

---

### 实例 7：生产推理 — 百川/通义/Kimi 的大模型部署

**场景**：服务 100B 级别的 Dense 模型，低延迟要求

**典型配置**：`TP=8, PP=1, DP=弹性扩展`

```
为什么 PP=1？
  推理时 PP 增加延迟（每个 Stage 需要等上一 Stage 完成）
  → 只要内存够，推理首选 TP 而不是 PP

一个推理集群（例如 32 卡服务 100B 模型）：

  实例 0（GPU 0-7，单机 8 卡）：TP=8，服务 1 份模型
  实例 1（GPU 8-15，单机 8 卡）：TP=8，服务 1 份模型
  实例 2（GPU 16-23，单机 8 卡）：TP=8
  实例 3（GPU 24-31，单机 8 卡）：TP=8

  Load Balancer：将 incoming requests 均匀分配给 4 个实例

  → 4 个实例 = 相当于 DP=4 的推理
  → 每个实例独立，互不通信（不同于训练的 DP AllReduce）
  → 实例可以随流量弹性增减（云原生）

推理时 DP vs 训练时 DP 的关键区别：
  训练 DP：需要 AllReduce 梯度（实例间有通信）
  推理 DP：实例完全独立（无通信），只需要负载均衡
```

---

## 五、理解 DP 和 TP 关系的三个核心规则

### 规则 1：TP 和 DP 正交组合

```
DP 和 TP 是独立的两个维度：

  world_size = TP × DP（× PP × EP）

  TP 决定：一份模型被切成多少片（机器内）
  DP 决定：模型有多少个副本（机器间）

  可以任意组合：
  TP=1, DP=8：8 卡各有完整模型，处理不同数据
  TP=8, DP=1：8 卡合力持有一份模型，同时处理相同数据
  TP=4, DP=2：2 个 4 卡 TP 组，各处理不同数据
  TP=2, DP=4：4 个 2 卡 TP 组，各处理不同数据

  （所有配置都使用 8 卡，但通信模式和内存布局完全不同）
```

### 规则 2：TP 在卡内，DP 在卡间

```
                     TP=4（卡内 NVLink，高带宽）
                  ┌─────────────────────────┐
  DP rank 0 →     │ GPU0  GPU1  GPU2  GPU3  │ ← 同一台机器
                  └─────────────────────────┘
                            ↕ DP AllReduce（跨机器 IB，低带宽）
  DP rank 1 →     ┌─────────────────────────┐
                  │ GPU4  GPU5  GPU6  GPU7  │ ← 另一台机器
                  └─────────────────────────┘

  TP 通信（AllReduce）：在机器内走 NVLink
  DP 通信（AllReduce）：在机器间走 InfiniBand

  → TP 必须在 NVLink 域内（因为每层都要通信，带宽要求高）
  → DP 可以跨机器（每步只通信一次，带宽要求相对低）
```

### 规则 3：TP 解决显存问题，DP 解决吞吐问题

```
模型太大，单卡放不下？
  → 增大 TP（把一层权重切到更多卡上）
  → 代价：更多 TP 通信，更高带宽需求

训练太慢，想更快？
  → 增大 DP（更多副本并行处理更多数据）
  → 代价：更多 DP 通信（梯度同步），线性增加总算力

两者同时需要？
  → TP + DP（先 TP 保证能放下，再 DP 扩展吞吐）

决策流程：
  1. 模型能放进一张卡？→ TP=1，只用 DP
  2. 需要 N 卡 TP？→ TP=N（机器内），再用 DP 扩展
  3. 机器内 GPU 数不够 TP？→ TP 不超过机器内 GPU 数，PP 做跨机切分
```

---

## 六、一张图总结

```
                        Global Batch（所有数据）
                               │
                    DP 切分（按样本数）
                   ╱                       ╲
           Batch A                          Batch B
    （DP rank 0 处理）                （DP rank 1 处理）
           │                                    │
   ┌───────┴───────┐                    ┌───────┴───────┐
   │  TP Group 0   │                    │  TP Group 1   │
   │ GPU0 GPU1     │                    │ GPU4 GPU5     │
   │ GPU2 GPU3     │                    │ GPU6 GPU7     │
   │（协作完成      │                    │（协作完成      │
   │  Batch A 的   │                    │  Batch B 的   │
   │  前向/反向）   │                    │  前向/反向）   │
   └───────────────┘                    └───────────────┘
           │                                    │
           └──────── DP AllReduce ──────────────┘
                    （梯度平均，步结束后）
                           │
                    参数更新（两组相同）
```

---

*参考资料：*
- *[Megatron-LM：Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053)*
- *[3D Parallelism（Narayanan et al. 2021）](https://arxiv.org/abs/2104.04473)*
- *[DeepSeek-V3 技术报告（训练配置章节）](https://arxiv.org/abs/2412.19437)*
- *[vLLM DP+TP 推理文档](https://docs.vllm.ai/en/stable/serving/distributed_serving.html)*
- *[MT-NLG 530B 训练报告（NVIDIA+Microsoft）](https://arxiv.org/abs/2201.11990)*
*更新：2026-03*
