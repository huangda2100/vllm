# 什么是 3D 并行？

> **3D 并行 = TP（张量并行）× PP（流水线并行）× DP（数据并行）**
>
> 三个维度各自解决一个独立的问题，组合在一起才能训练超大规模模型。

---

## 一、为什么需要"3D"？

### 1.1 单一并行方式的瓶颈

每种并行方式单独使用时，都有无法克服的上限：

```
仅用 TP：
  TP=8 → 一台机器 8 卡，权重切分到 8 卡
  TP=16 → 必须跨机器，NVLink 失效，IB 带宽太低
  → TP 的上限 ≈ 单机 GPU 数（通常 8 卡）

仅用 PP：
  PP=N → 模型被切成 N 段，每段一台机器
  缺点：Pipeline Bubble（气泡），即使有 microbatch 优化，效率损失仍达 10-30%
  → PP 可以跨机，但气泡是内在代价

仅用 DP：
  DP=N → 每台机器有完整模型副本
  前提：一台机器能放下完整模型！
  → 模型太大时 DP 无法单独使用
```

**结论**：训练千亿参数模型时：
- TP 单独用：上限 8 卡，装不下千亿参数
- PP 单独用：气泡效率损失大，且每个 Stage 还是要装数十亿参数
- DP 单独用：单机内存不足

**解法**：把三个维度叠加起来。

---

### 1.2 三个维度各司其职

```
┌─────────────────────────────────────────────────────────────────────┐
│                    3D 并行的职责划分                                  │
│                                                                      │
│  TP（张量并行）：解决"单层太大"                                       │
│    → 把一层的权重矩阵切分到多卡（机器内，NVLink）                      │
│    → 粒度：层内（intra-layer）                                        │
│                                                                      │
│  PP（流水线并行）：解决"层数太多/总参数太大"                            │
│    → 把不同层分配到不同机器（机器间，InfiniBand）                      │
│    → 粒度：层间（inter-layer）                                        │
│                                                                      │
│  DP（数据并行）：解决"训练太慢/吞吐不够"                               │
│    → 复制多份模型（TP+PP 的整体），处理不同数据                        │
│    → 粒度：步间（inter-step，梯度同步）                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 二、3D 并行的 GPU 布局

### 2.1 四维 Rank 矩阵

以 `TP=2, PP=2, DP=2`（共 8 卡）为例，GPU 的逻辑布局是一个三维矩阵：

```
all_ranks[dp][pp][tp]

dp=0, pp=0: [GPU 0, GPU 1]   ← TP Group A（PP Stage 0，DP Replica 0）
dp=0, pp=1: [GPU 2, GPU 3]   ← TP Group B（PP Stage 1，DP Replica 0）
dp=1, pp=0: [GPU 4, GPU 5]   ← TP Group C（PP Stage 0，DP Replica 1）
dp=1, pp=1: [GPU 6, GPU 7]   ← TP Group D（PP Stage 1，DP Replica 1）
```

用三维图形表示：

```
                        DP 维度（复制模型）
                       ╱                 ╲
                  DP=0                    DP=1
                 ╱    ╲                  ╱    ╲
            PP=0      PP=1          PP=0      PP=1
           ╱    ╲   ╱    ╲        ╱    ╲   ╱    ╲
         TP=0  TP=1 TP=0 TP=1  TP=0  TP=1 TP=0 TP=1
         GPU0  GPU1 GPU2 GPU3  GPU4  GPU5 GPU6 GPU7

各组通信：
  TP 组（横向，同 DP 同 PP）：{GPU0,GPU1}, {GPU2,GPU3}, {GPU4,GPU5}, {GPU6,GPU7}
  PP 组（纵向，同 DP 同 TP）：{GPU0,GPU2}, {GPU1,GPU3}, {GPU4,GPU6}, {GPU5,GPU7}
  DP 组（深度，同 PP 同 TP）：{GPU0,GPU4}, {GPU1,GPU5}, {GPU2,GPU6}, {GPU3,GPU7}
```

**每张 GPU 同时属于三个不同的通信组！**

---

### 2.2 大规模实例：TP=8, PP=4, DP=8（256 卡）

```
总 GPU = 8 × 4 × 8 = 256 张

物理部署（以 DGX A100 为例，每台 8 卡）：
  每台机器内 8 卡 = 1 个 TP Group（机器内 NVLink）
  4 台机器组成 1 条 PP 流水线（4 个 Stage）
  8 条 PP 流水线（DP=8 个副本）
  → 共 4 × 8 = 32 台机器

机器编号与职责：

  DP Replica 0（机器 0-3）：
    机器 0（GPU 0-7）：   PP Stage 0，层 0-N/4，       8卡做 TP
    机器 1（GPU 8-15）：  PP Stage 1，层 N/4-N/2，     8卡做 TP
    机器 2（GPU 16-23）： PP Stage 2，层 N/2-3N/4，    8卡做 TP
    机器 3（GPU 24-31）： PP Stage 3，层 3N/4-N，      8卡做 TP

  DP Replica 1（机器 4-7）：与 Replica 0 相同的权重，不同数据
  DP Replica 2（机器 8-11）：...
  ...
  DP Replica 7（机器 28-31）：...

通信路径：
  机器内（NVLink 600GB/s）：  TP AllReduce（每层 ×2，频繁）
  机器间同 PP 位置（IB）：    DP AllReduce（每步 ×1，与反向重叠）
  机器间相邻 PP Stage（IB）： PP P2P（每 microbatch 传激活值）
```

---

## 三、3D 并行中的数据流

以 LLaMA-3-70B 训练（TP=8, PP=4, DP=2，共 64 卡）为例。

### 3.1 模型切分方式

```
LLaMA-3-70B：80 层，hidden=8192，FFN=28672

PP=4 切分：
  PP Stage 0（机器 0）：Embedding + Layer 0~19
  PP Stage 1（机器 1）：Layer 20~39
  PP Stage 2（机器 2）：Layer 40~59
  PP Stage 3（机器 3）：Layer 60~79 + Final Norm + LM Head

每个 Stage 内，TP=8（机器内 8 卡）：
  q_proj:   [8192, 8192] → 每卡 [1024, 8192]（8192/8=1024 行）
  k_proj:   [1024, 8192] → 每卡 [128, 8192] （GQA，8KV头/8=1个KV头/卡）
  down_proj:[8192,28672] → 每卡 [8192, 3584]（28672/8=3584 列）

每卡持有参数量：
  70B / 4（PP）/ 8（TP）= 2.19B 参数/卡
  × 2bytes (BF16) = 4.37 GB
  加激活值 + Adam 状态 ≈ 30-40 GB/卡 → 80GB A100 可容纳
```

### 3.2 前向传播流（1F1B 调度）

```
Global Batch = 64 sequences（每条 2048 tokens）

DP 切分（DP=2）：
  DP Replica 0（机器 0-7）：处理 Batch A（sequence 0-31）
  DP Replica 1（机器 8-15）：处理 Batch B（sequence 32-63）

PP microbatch（Batch A 内继续切分，假设 4 个 microbatch）：
  μbatch 0: sequence 0-7
  μbatch 1: sequence 8-15
  μbatch 2: sequence 16-23
  μbatch 3: sequence 24-31

时间线（Replica 0 的 1F1B 调度）：

时刻 →   1    2    3    4    5    6    7    8    9    10   11   12
机器0:   F0   F1   F2   F3   B0   B1   B2   B3
         │    │    │    │    ↑    ↑    ↑    ↑
         └─Send─┘ └─Send─┘  └─Recv─┘└─Recv─┘ （PP P2P 传激活值/梯度）

机器1:   [W]  F0   F1   F2   F3   B0   B1   B2   B3
              │    │    │    ↑    ↑    ↑    ↑
              └─Send─┘ └─Send─┘ └─Recv─┘└─Recv─┘

机器2:   [W] [W]  F0   F1   F2   F3   B0   B1   B2   B3
                   │    │    │    ↑    ↑    ↑    ↑

机器3:   [W] [W] [W]  F0   F1   F2   F3  loss B0   B1   B2   B3

F = Forward（前向），B = Backward（反向）
[W] = Wait（气泡，Bubble）
数字 = microbatch 编号

气泡比例 = (PP-1)/num_microbatches = 3/4 → 仅 4 个 microbatch 效率低！
实际训练通常 microbatch 数 >> PP，如 32 microbatch：气泡 = 3/32 ≈ 9%
```

### 3.3 每个时刻的层内 TP 通信

```
在机器 0 执行 F0（microbatch 0 的前向）时，机器 0 的 8 张卡在做：

  t=0: GPU 0-7 接收相同的输入 X: [8, 2048, 8192]
  t=1: [TP 层内]
       GPU 0: Q₀ = X @ q_proj[0:1024,:].T
       GPU 1: Q₁ = X @ q_proj[1024:2048,:].T
       ...（列并行，无通信）
  t=2: [TP 层内]
       Flash Attention（各卡独立，无通信）
  t=3: [TP 层内]
       O_proj 行并行 → 各卡得部分和 P₀, P₁,...P₇
       ★ AllReduce（NVLink，GPU 0-7 内部）
       → 完整激活值 [8, 2048, 8192]
  t=4: [TP 层内]
       FFN Gate/Up 列并行（无通信）
       FFN Down 行并行 → AllReduce（NVLink）
       ★ AllReduce（NVLink，GPU 0-7 内部）

  ×20 层后，机器 0 的 Layer 0-19 计算完成
  → P2P Send：激活值 [8, 2048, 8192] → 机器 1（PP 通信）

  机器 1 收到激活值 → 开始 Layer 20-39 的计算（同样 TP 内部 AllReduce）
  ...
  机器 3 计算完 → loss → 开始反向
```

### 3.4 反向传播与梯度同步

```
反向传播（对称于前向，方向相反）：
  机器 3（PP Stage 3）计算 loss，开始反向
  → P2P Send：梯度 ∂L/∂h₅₉ → 机器 2
  → 机器 2 反向传播，得到 ∂L/∂h₃₉ → P2P Send → 机器 1
  → ...

机器内 TP 反向通信（与前向对称）：
  每层 ×2 次 AllReduce（行并行层的输入梯度同步）

反向完成后，DP AllReduce（步内最后一步通信）：
  DP Group 0（相同 TP rank，相同 PP stage）：
    机器 0 的 GPU 0 ↔ 机器 8 的 GPU 0（DP Replica 0 vs 1 的 TP rank 0）
    AllReduce：∂L/∂q_proj[0:1024,:]（Batch A vs Batch B 的梯度平均）

  这个 AllReduce 在反向传播的最后几层计算时就异步发起，与计算重叠
```

---

## 四、3D 并行的通信量分析

### 4.1 三种通信各自的量与频率

以 LLaMA-3-70B，TP=8，PP=4，DP=2，tokens=1024/microbatch，BF16 为例：

```
──── TP AllReduce（机器内，每层 2 次，每次前向）────────────────────────

每次 AllReduce 数据量：
  = [tokens, hidden] × 2 bytes
  = 1024 × 8192 × 2 = 16 MB

每层 2 次，80 层：
  = 2 × 80 × 16 MB = 2560 MB ≈ 2.5 GB（前向）
  前向 + 反向 ≈ 5 GB

NVLink 带宽 600 GB/s（双向）：
  Ring-AllReduce 实际带宽 = 600 × (TP-1)/TP = 600 × 7/8 = 525 GB/s
  耗时 ≈ 5 GB / 525 GB/s ≈ 9.5 ms（可与 GEMM 重叠后 < 3ms）

──── PP P2P（机器间，每 microbatch 每 Stage 边界）──────────────────────

每次 P2P 数据量：
  = [tokens, hidden] × 2 bytes
  = 1024 × 8192 × 2 = 16 MB

PP=4 有 3 个 Stage 边界，4 个 microbatch，前向 + 反向：
  = 2 × 3 × 4 × 16 MB = 384 MB ≈ 0.4 GB

IB 带宽 25 GB/s（单向）：
  耗时 ≈ 0.4 GB / 25 GB/s ≈ 16 ms（可与流水线计算重叠）

──── DP AllReduce（机器间，每步 1 次）──────────────────────────────────

每次 AllReduce 数据量：
  = 该 TP rank 持有的参数量 × 2 bytes（梯度，BF16）
  = 70B / 4(PP) / 8(TP) × 2 bytes = 2.19B × 2 = 4.37 GB

ZeRO-1 用 ReduceScatter 替代（每卡只收 1/DP 的梯度）：
  实际通信量 ≈ 4.37 GB（总量不变，但内存减少）

IB 带宽 25 GB/s：
  耗时 ≈ 4.37 GB / 25 GB/s ≈ 175 ms
  （与反向传播的后半段重叠，实际延迟 << 175ms）
```

### 4.2 通信与计算的时间占比

```
一个 Step 的时间构成（估算，LLaMA-3-70B，256 卡）：

  计算时间（GEMM + Attention）：~400 ms
  TP 通信（NVLink，隐藏在计算内）：~3 ms（重叠后）
  PP 通信（IB P2P，流水线隐藏）：~5 ms（重叠后）
  DP 通信（IB AllReduce，与反向重叠）：~15 ms（重叠后）

  总 Step 时间 ≈ 420 ms
  有效计算占比 ≈ 400/420 ≈ 95%（理想情况，实际会低一些）
```

---

## 五、3D 并行的最优配置原则

### 5.1 三个维度的排列顺序

**Megatron-LM 论文给出的配置原则（从内到外）**：

```
配置优先级：TP > PP > DP

1. 先定 TP（最内层）：
   TP ≤ 单机 GPU 数（保证在 NVLink 域内）
   TP 以能把最大的权重矩阵切开为目标

2. 再定 PP（中间层）：
   PP = 剩余显存需求 / 单机能持有的层数
   PP 尽量小（减少 Bubble），但要保证每 Stage 显存够用

3. 最后定 DP（最外层）：
   DP = 总 GPU 数 / (TP × PP)
   DP 越大，训练吞吐越高（线性扩展）
```

### 5.2 为什么是这个顺序？

```
TP 放最内层：
  TP 通信（AllReduce）最频繁（每层 2 次，阻塞），必须用最高带宽的 NVLink
  NVLink 只在机器内，所以 TP 必须是"最小的并行单元"

PP 放中间层：
  PP 通信（P2P）比 TP 少（每 microbatch 一次），且可以流水线化
  适合跨机器（IB），但仍然是单向点对点，比 AllReduce 快

DP 放最外层：
  DP 通信（梯度 AllReduce）每步只有一次，可以完全与计算重叠
  即使带宽最低，也几乎不影响训练时间（只要 batch 够大）
  适合跨多台机器甚至跨数据中心
```

---

## 六、工业界 3D 并行配置实例

### 6.1 NVIDIA Megatron-LM：GPT-3 175B

```
集群：1024 张 A100-80G
配置：TP=8, PP=16, DP=8（8 × 16 × 8 = 1024 ✓）

物理布局：
  每台 DGX A100（8卡）= 1 个 TP Group
  16 台机器组成 1 条 PP 流水线
  8 条 PP 流水线（DP=8）
  总机器数：16 × 8 = 128 台

模型切分：
  96 层 / 16 PP Stage = 6 层/Stage
  参数量/卡 = 175B / 8 / 16 ≈ 1.37B 参数 ≈ 2.7 GB（BF16）
  加激活/Adam：~50 GB/卡（A100-80G 可容纳）

训练效率：
  MFU（Model FLOP Utilization）≈ 50%
  Pipeline Bubble ≈ 15/16 ÷ microbatch_count
  TP AllReduce 是主要延迟来源（每层 2 次，96 层 = 192 次/step）
```

### 6.2 NVIDIA MT-NLG 530B（Microsoft + NVIDIA）

```
集群：2240 张 A100-80G（280 台 DGX）
配置：TP=8, PP=35, DP=8（8 × 35 × 8 = 2240 ✓）

模型切分：
  105 层 / 35 PP Stage = 3 层/Stage（每 Stage 非常薄！）
  参数量/卡 = 530B / 8 / 35 ≈ 1.89B 参数 ≈ 3.8 GB

PP=35 的代价：
  Pipeline Bubble = (35-1)/microbatch_count
  需要 microbatch ≥ 140 才能把 Bubble 压到 < 20%

训练吞吐：
  论文报告：126 TFLOP/s/GPU（A100 理论值 312 TFLOP/s）
  MFU ≈ 40%（PP 深度增加了 Bubble 代价）
```

### 6.3 DeepSeek-V3（671B MoE）

```
集群：2048 张 H800
配置：TP=1, PP=16, EP=64, ZeRO-1 DP

特殊之处：完全没有 TP！

为什么？
  MoE 模型每步激活参数 = 37B（不是 671B）
  MLA 压缩 KV → 单 Stage 显存大幅降低
  → 16 个 Stage，每 Stage 平均 ~42B 参数 / 8 卡 ≈ 1.3B 参数/卡
  → 单卡 80GB 放得下（只需 ~2.6GB 权重，其他是 KV Cache 和 Adam）

  TP=1 的好处：
  消除所有 TP AllReduce（节省 ~12ms/step）
  → 替换为更高效的 EP AllToAll（可被 DualPipe 重叠）

  DualPipe：新型 PP 调度算法
  → PP Bubble ≈ 0（双向流水线填充气泡）
  → EP AllToAll 与 PP P2P 时间上互相掩盖
```

### 6.4 Google TPU Pod：PaLM 540B

```
集群：6144 张 TPU v4
配置：DP=12（多路复制），每路 512 TPU 做 TP+模型切分

TPU 的特殊性：
  TPU Pod 内部有超快的专用网络（ICI，Inter-Chip Interconnect）
  带宽远高于 GPU InfiniBand
  → 可以使用更大的 TP（TP 跨多个 TPU 芯片也可以接受）

  PaLM 的配置：
  TP-like 切分（Pathways 系统）跨 512 TPU
  类似 DP=12 的全局复制
  MFU ≈ 46%（在超大规模下的理论极限）
```

### 6.5 LLaMA-3-70B 训练（Meta，估算）

```
集群（推测）：~2000 张 H100
配置：TP=8, PP=4（或 PP=8），DP=~60

物理：
  每台 DGX H100（8卡）= 1 TP Group
  4 台机器 = 1 PP 流水线
  ~60 条并行流水线（DP）
  总机器 ≈ 4 × 60 = 240 台（1920 卡）

全局批大小：~4M tokens/step
训练数据：15T tokens
总 Step ≈ 15T / 4M ≈ 3.75M 步（估算）
```

---

## 七、3D 并行的关键权衡

### 7.1 增大各维度的收益与代价

```
增大 TP（从 4 → 8）：
  收益：每卡权重减半，支持更大模型
  代价：AllReduce 次数不变，但每次通信卡数翻倍（Ring 步骤 ×2）
        TP=8 时，每次 AllReduce 需要 7 步（TP=4 时 3 步）
        → 每步通信时间 ≈ 翻倍

增大 PP（从 4 → 8）：
  收益：每卡层数减半，支持更大模型（或节省更多显存）
  代价：Pipeline Bubble 增大（(pp-1)/microbatch）
        需要更多 microbatch 才能把 Bubble 压到可接受水平

增大 DP（从 8 → 16）：
  收益：吞吐线性提升（近似），每步处理 ×2 的数据
  代价：梯度 AllReduce 数据量不变，但通信节点翻倍
        Ring-AllReduce 总量 = 2×(dp-1)/dp×params ≈ 2×params（dp 大时）
        不随 dp 增大，通信时间几乎不变！
  → DP 扩展是三种方式中"最便宜"的
```

### 7.2 通信类型与硬件的匹配

```
通信类型       频率          适合的硬件
─────────────────────────────────────────────────────
TP AllReduce  每层 ×2（高）  NVLink（600 GB/s），机器内
PP P2P        每 μbatch ×1  IB（25-100 GB/s），机器间
DP AllReduce  每步 ×1（低）  IB（25-100 GB/s），机器间

对应到 3D 并行布局的层次：
  TP = 机器内的 GPU 组（NVLink 域，最快）
  PP = 相邻机器组（IB P2P，中速）
  DP = 所有机器（IB AllReduce，最慢但可隐藏）
```

### 7.3 微批次数（microbatch count）对性能的影响

```
PP 的 Bubble 比例 = (PP - 1) / microbatch_count

microbatch_count = global_batch_size / (micro_batch_size × TP × DP)

例：global_batch = 1024 sequences，micro_batch = 4，TP=8，DP=8
  microbatch_count = 1024 / (4 × 8 × 8) = 4

  PP=4 时，Bubble = 3/4 = 75%（效率极低！）
  PP=4，增加 microbatch 数到 32：Bubble = 3/32 ≈ 9%（可接受）

  所以：PP 越深，需要越大的 global_batch 来填充 Bubble
  大 batch → 需要调整学习率（LR Scaling）
  → PP 深度与 batch size 是耦合的！
```

---

## 八、3D 并行 vs 其他并行方案

### 8.1 与 ZeRO 的关系

```
ZeRO（Zero Redundancy Optimizer）是 DP 的"内存优化版本"，
可以与 3D 并行结合使用：

  标准 DP（每卡完整参数 + 完整梯度 + 完整 Adam 状态）
  ZeRO-1（分片 Adam 状态）
  ZeRO-2（分片 Adam + 梯度）
  ZeRO-3（分片 Adam + 梯度 + 参数）

  3D 并行 + ZeRO-1（DeepSeek-V3 的方案）：
  TP=1, PP=16, EP=64 + ZeRO-1 DP
  → 参数不分片（每卡完整 PP Stage 权重），但 Adam 状态分片
  → 节省 Adam 状态显存（约 1/3 的显存优化）

  3D 并行 + ZeRO-3（极端省内存）：
  TP=1, PP=1 + ZeRO-3（DeepSpeed 常用配置）
  → 相当于用 ZeRO-3 替代 TP，牺牲通信频率换取更大模型支持
```

### 8.2 与 Sequence Parallelism（SP）的关系

```
SP 是 TP 的增强版（Megatron-LM 2022 年提出）：

  纯 TP：AllReduce 后所有卡都保存完整激活 [tokens, hidden]
  TP+SP：AllReduce → Reduce-Scatter，每卡只保存 1/tp 的序列
         → 激活显存节省 tp 倍

  SP 通常与 TP 一起使用：
  "TP=8+SP" 是 DeepSeek-V3 推理的配置

  3D 并行实际上是 TP+SP+PP+DP 四维：
  有时写作"TP+SP+PP+DP"而不是"3D 并行"
```

### 8.3 与 Context Parallelism（CP）的关系

```
CP（上下文并行）：为超长序列设计的第四维度

  问题：序列长度 128K，Attention 计算的显存是 O(S²)
  CP 解法：把序列切分到多卡
    CP=4 → 每卡只处理 32K tokens 的 Attention

  CP 与 SP 的区别：
  SP：切分非 Attention 层（LayerNorm 等）的序列维度
  CP：切分 Attention 本身（用 Ring Attention 等算法）

  4D 并行 = TP + CP + PP + DP（Google 长上下文训练使用）
```

---

## 九、总结：3D 并行的核心心智模型

```
把模型想象成一栋大楼：

  楼层（Layers）= 模型的层
  楼内的房间（Layer 内部）= 权重矩阵

  PP（流水线并行）= 把楼切成几段，每段给一台机器
  TP（张量并行）  = 每段内部，把大房间切开，分给同台机器的多张卡
  DP（数据并行）  = 复制整栋楼，同时接待多批"客人"（数据）

  客人（数据）的旅程：
  → 进入 DP Replica 0 的楼（DP 决定去哪栋楼）
  → 经过 PP Stage 0（机器 0 的楼层 0-19）
    → 机器 0 内 TP 协作处理（机器内 8 卡协作）
  → P2P 传递到 PP Stage 1（机器 1 的楼层 20-39）
    → 机器 1 内 TP 协作处理
  → ...经过所有 PP Stage
  → 得到 logits，计算 loss，反向传播
  → 梯度 AllReduce 到所有 DP 副本（各栋楼同步经验）
  → 所有楼参数同步更新，开始接待下一批客人
```

| 并行维度 | 解决的问题 | 切分对象 | 通信方式 | 硬件要求 |
|---------|----------|---------|---------|---------|
| TP | 单层权重太大 | 权重矩阵（行/列）| AllReduce（每层）| NVLink（机器内）|
| PP | 总层数太多 | 模型层（按 Stage）| P2P Send/Recv | IB（机器间）|
| DP | 训练吞吐不够 | 输入数据（按样本）| AllReduce（每步）| IB（机器间）|
| **3D** | **三者之和** | **全部** | **三种通信** | **NVLink+IB** |

---

*参考资料：*
- *[Megatron-LM（原始 TP 论文）：Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053)*
- *[3D 并行论文（Narayanan et al. 2021）](https://arxiv.org/abs/2104.04473)*
- *[Sequence Parallelism（Korthikanti et al. 2022）](https://arxiv.org/abs/2205.05198)*
- *[PaLM 540B 训练（Chowdhery et al. 2022）](https://arxiv.org/abs/2204.02311)*
- *[MT-NLG 530B（Smith et al. 2022）](https://arxiv.org/abs/2201.11990)*
- *[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)*
- *[ZeRO（Rajbhandari et al. 2020）](https://arxiv.org/abs/1910.02054)*
*更新：2026-03*
