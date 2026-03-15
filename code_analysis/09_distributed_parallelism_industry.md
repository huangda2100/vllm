# 工业界 TP/PP/DP/EP 决策原理：以 DeepSeek 为例

> **官方文献依据**：
> - [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)
> - [DeepSeek-V2 Technical Report (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434)
> - [DeepSeek Open-Infra-Index（官方开源推理系统概览）](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)
> - [DualPipe 开源代码库](https://github.com/deepseek-ai/DualPipe)

---

## 一、DeepSeek 的实际并行配置（官方证据）

### 1.1 DeepSeek-V3 训练配置

**模型规模**：671B 总参数，每个 token 激活 37B（MoE）

官方技术报告原文（Section 3.3 Training Framework）：

> *"We meticulously optimize our memory usage, making it possible to train DeepSeek-V3 without using costly Tensor Parallelism."*
>
> *"For the computation graph of DeepSeek-V3, we employ 16-way Pipeline Parallelism (PP), 64-way Expert Parallelism (EP) across 8 nodes, combined with ZeRO-1 Data Parallelism."*

| 并行维度 | 取值 | 说明 |
|---------|------|------|
| **TP** | **1（不使用）** | 通过内存优化彻底消除 TP 需求 |
| **PP** | **16** | DualPipe 双向流水线，16个 Stage |
| **EP** | **64**（跨 8 节点）| MoE 专家分布在 64 张 GPU |
| **DP** | ZeRO-1 | 梯度分片，进一步节省优化器内存 |
| **GPU 集群** | **2048 × H800** | 节点内 NVLink，节点间 InfiniBand |

**为什么训练不用 TP？**

V3 技术报告明确指出，H800 的 NVLink 带宽约 160 GB/s（有效），InfiniBand 约 40 GB/s。TP 的 AllReduce 需要在**每一个 Transformer 层都同步**（每层 2 次），无法异步。与 DeepSeek V2 相比，V3 通过以下手段消除了 TP 的必要性：

1. **MLA（Multi-head Latent Attention）**：用低秩压缩 KV，将 KV Cache 从原始的 `[L, 2*H]` 压缩到 `[L, compressed_dim]`，大幅减少每卡内存占用
2. **DeepSeekMoE 分组**：每个 MoE 层只激活 37B 参数，单卡内存远小于 Dense 模型
3. **精心的显存管理**：激活重计算 + 优化器 ZeRO-1 分片

---

### 1.2 DeepSeek-V3 推理配置（官方分离部署策略）

官方文档（Technical Report Section 3.4 + Open-Infra-Index）明确将 **Prefill 和 Decode 阶段分离部署**：

#### Prefill 阶段（计算密集）

官方原文：
> *"The minimum deployment unit of the prefilling stage consists of 4 nodes with 32 GPUs. The attention part employs TP4 with Sequence Parallelism (SP), combined with DP8. Its small TP size of 4 limits the overhead of TP communication. For the MoE part, we use EP32."*

| 组件 | 配置 | 逻辑 |
|------|------|------|
| Attention (MLA) | **TP4 + SP + DP8** | 小 TP 限制 AllReduce 开销；DP8 提高吞吐 |
| MoE Expert | **EP32** | 保证每个 Expert 有足够 batch size |
| 最小部署单元 | **4节点 × 8卡 = 32 GPU** | 一个 Prefill 实例 |

#### Decode 阶段（内存密集）

官方原文：
> *"The minimum deployment unit of the decoding stage consists of 40 nodes with 320 GPUs. The attention part employs TP4 with SP, combined with DP80, while the MoE part uses EP320. For the MoE part, each GPU hosts only one expert."*

| 组件 | 配置 | 逻辑 |
|------|------|------|
| Attention (MLA) | **TP4 + SP + DP80** | 同 Prefill |
| MoE Expert | **EP320（每卡一个 Expert）** | 极限并行，最小化 KV Cache + Expert 显存 |
| 通信方式 | **点对点 IB + IBGDA** | 低延迟，避免 AllToAll 集合通信开销 |
| 最小部署单元 | **40节点 × 8卡 = 320 GPU** | 一个 Decode 实例 |

**另：官方 Open-Infra-Index 最新生产配置（2025年2月）：**

```
Prefill：EP32，每 GPU 9个路由专家 + 1个共享专家，32个冗余路由专家
         跨 4 节点部署

Decode：EP144，每 GPU 2个路由专家 + 1个共享专家，32个冗余路由专家
        跨 18 节点部署
```

---

### 1.3 DeepSeek-V2 的对比（演进视角）

**模型规模**：236B 总参数，每个 token 激活 21B

官方技术报告原文：
> *"For each layer, routed experts are uniformly deployed on D=8 devices. For device-limited routing, each token will be sent to at most M=3 devices."*

| 并行维度 | V2 | V3 | 变化原因 |
|---------|----|----|--------|
| EP | EP8（每层） | EP64 | V3 更多专家（256 vs 64），需要更大 EP |
| PP | 未明确 | PP16（DualPipe）| V3 层数更多，PP 进一步加深 |
| TP | 未使用（推断） | 不使用 | MLA 内存优化消除需求 |
| 路由限制 | 最多 3 节点 | 最多 4 节点 | 更多节点可扩展性 |

---

## 二、如何理解 TP 组

### 2.1 TP 组的定义

**TP 组（Tensor Parallel Group）** 是一组 GPU 的集合，它们**共同协作计算同一个矩阵乘法**。

在代码层面（参考 `vllm/distributed/parallel_state.py`）：

```python
# all_ranks 的四维布局：[external_dp, dp, pp, tp]
all_ranks = torch.arange(world_size).reshape(
    external_dp_size,
    data_parallel_size,
    pipeline_model_parallel_size,
    tensor_model_parallel_size,
)

# TP 组：固定 external_dp + dp + pp 维度，沿 tp 维度取所有 rank
# 例如 world_size=16, DP=2, PP=2, TP=4：
# TP group [0,1,2,3] 是第一个 TP 组（机器0的4张卡）
# TP group [4,5,6,7] 是第二个 TP 组（机器1的4张卡）
```

**核心性质**：
- TP 组内所有 GPU 处理**同一批 tokens**（相同输入）
- TP 组内每张 GPU 只持有**权重的 1/TP 份**
- TP 组内必须在每一层 AllReduce（同步结果）

### 2.2 TP 组内的矩阵切分方式

以一个 Transformer 层的 FFN 为例（LLaMA-3-8B，hidden=4096，FFN=14336，TP=4）：

```
完整 FFN（TP=1 时）：
  gate_proj: [14336, 4096]  ← 列并行（Column Parallel）
  down_proj: [4096, 14336]  ← 行并行（Row Parallel）

TP=4 切分后，每个 GPU 持有：
  gate_proj: [3584, 4096]   ← 只有 14336/4 = 3584 行
  down_proj: [4096, 3584]   ← 只有 14336/4 = 3584 列
```

**数据流（以 TP=4 为例）**：

```
输入 X: [tokens, 4096] → 广播给所有 4 张卡（每卡都有完整输入）
                ↓
GPU 0: gate[0:3584] → 输出 [tokens, 3584]
GPU 1: gate[3584:7168] → 输出 [tokens, 3584]
GPU 2: gate[7168:10752] → 输出 [tokens, 3584]
GPU 3: gate[10752:14336] → 输出 [tokens, 3584]

                ↓ 各自独立做 SiLU 激活 + Up 乘法
                ↓ 再经过 down_proj（行并行）

GPU 0: 部分输出 Y₀: [tokens, 4096]
GPU 1: 部分输出 Y₁: [tokens, 4096]
GPU 2: 部分输出 Y₂: [tokens, 4096]
GPU 3: 部分输出 Y₃: [tokens, 4096]

                ↓ AllReduce（求和）
Y = Y₀ + Y₁ + Y₂ + Y₃: [tokens, 4096]  ← 完整结果
```

**每个 Transformer 层发生 2 次 AllReduce**（Attention 的 o_proj 之后 + FFN 的 down_proj 之后）。

### 2.3 TP 组的拓扑约束（为什么不能跨机器）

**AllReduce 是同步阻塞操作**，必须等所有 GPU 都完成才能继续下一层。

```
通信时间计算（单层 AllReduce）：
  数据量 = 2 × tokens × hidden_size × dtype_bytes
         = 2 × 2048 × 4096 × 2 (BF16) = 32 MB

  机内 NVLink（A100）: 32 MB / 600 GB/s ≈ 0.05 ms  ← 可接受
  跨机 InfiniBand：    32 MB /  25 GB/s ≈ 1.3 ms   ← 严重拖累！

一个 80 层模型（LLaMA-3-70B）：
  机内：80 × 2 × 0.05 = 8 ms/step
  跨机：80 × 2 × 1.3  = 208 ms/step  → 延迟增加 26 倍！
```

**结论**：TP 组必须限制在同一台物理机的 NVLink 域内（通常 ≤ 8 卡）。

### 2.4 TP 组与 EP 组、PP 组的关系

```
Rank 映射（world_size=64，TP=4，PP=4，DP=4，以 DeepSeek 类似配置）：

all_ranks[dp][pp][tp] 布局：

DP=0，PP=0：[0,  1,  2,  3]   ← TP group 0（机器0，4卡）
DP=0，PP=1：[4,  5,  6,  7]   ← TP group 1（机器1，4卡）
DP=0，PP=2：[8,  9, 10, 11]   ← TP group 2（机器2，4卡）
DP=0，PP=3：[12,13, 14, 15]   ← TP group 3（机器3，4卡）

DP=1，PP=0：[16,17, 18, 19]   ← TP group 4（机器4，4卡）
...

PP group（流水线，跨机器）：[0, 4, 8, 12]  ← 4个 stage，各在不同机器
DP group（梯度同步）：[0, 16, 32, 48]     ← 相同 PP+TP 位置，不同 DP rank
```

**EP 组与 TP 组不同**：EP 组是 MoE 专家的分布单元，每个 EP rank 持有一部分专家（而非权重的列/行切片）。

---

## 三、分布式通信量的精确计算

### 3.1 TP 的 AllReduce 通信量

**Ring-AllReduce 算法分两阶段**：

```
阶段 1：Reduce-Scatter（聚合）
  每张 GPU 将数据分成 tp 块
  每步发送 1 块（大小 = V/tp），共 (tp-1) 步
  每卡发送量 = (tp-1)/tp × V

阶段 2：AllGather（广播）
  每步发送 1 块（大小 = V/tp），共 (tp-1) 步
  每卡发送量 = (tp-1)/tp × V

总通信量（每卡）= 2 × (tp-1)/tp × V
```

**公式**：

$$\text{AllReduce 通信量} = 2 \times \frac{tp-1}{tp} \times N_{elements} \times \text{dtype\_bytes}$$

**TP 每层实际通信量**（LLaMA-3-8B，tokens=2048，hidden=4096，TP=4）：

| 操作 | 张量大小 | 通信量 |
|------|---------|--------|
| Attention o_proj 后 AllReduce | [2048, 4096] × 2 bytes | 2 × 3/4 × 16 MB = 24 MB |
| FFN down_proj 后 AllReduce | [2048, 4096] × 2 bytes | 2 × 3/4 × 16 MB = 24 MB |
| **每层合计** | — | **48 MB** |
| **32层模型合计（单步前向）** | — | **1536 MB = 1.5 GB** |

**通信时间**（NVLink 600 GB/s）：
```
单层：48 MB / 600 GB/s ≈ 0.08 ms
32层：1.5 GB / 600 GB/s ≈ 2.5 ms（前向）
```

### 3.2 Sequence Parallelism（SP）优化

DeepSeek-V3 推理用的是 **TP + SP**，而非纯 TP。

SP 将 AllReduce 替换为 Reduce-Scatter + AllGather：

```
纯 TP：
  ... → Attention → [tokens, hidden] → AllReduce → ...
                              ↑ 完整 hidden，冗余存储

TP + SP：
  ... → Reduce-Scatter → [tokens/tp, hidden] → LayerNorm → AllGather → Attention ...
                                   ↑ 每卡只存 1/tp 的序列！
```

**SP 的好处**：
- 通信量与纯 TP 相同（AllReduce = Reduce-Scatter + AllGather）
- 激活值显存节省 TP 倍（LayerNorm 层的输入从 `[tokens, H]` → `[tokens/tp, H]`）
- DeepSeek Prefill 用 TP4+SP 节省 4× 激活内存

### 3.3 PP 的 P2P 通信量

PP 只有相邻 Stage 间传输激活值，通信量极小：

```
每步 P2P 通信量（单向）：
  数据 = [tokens, hidden_size] × dtype_bytes
       = 2048 × 4096 × 2 (BF16) = 16 MB

一步完整流水线（前向 + 反向）：
  前向：(pp-1) 次 P2P 发送，每次 16 MB
  反向：(pp-1) 次 P2P 发送，每次 16 MB（梯度）

PP=16，训练 1 个 microbatch：
  总 P2P 量 = 2 × 15 × 16 MB = 480 MB
  对比 TP AllReduce（1.5 GB 前向）：PP 通信量低 3 倍以上
```

**PP 通信特点**：
- 只有 P2P（点对点），不是集合通信（AllReduce）
- 适合跨机器（InfiniBand），因为不需要同步等待所有节点
- 延迟来源是 **pipeline bubble**，而非带宽

**Pipeline Bubble 比例**（1F1B 调度）：

$$\text{Bubble ratio} = \frac{pp-1}{num\_microbatches}$$

```
DeepSeek-V3（PP=16，num_microbatches=？）：
  DualPipe 算法将 bubble 降至接近 0，通过双向流水线填充气泡
  官方：Bubble ratio < 1/pp_size（远优于标准 1F1B 的 (pp-1)/m）
```

### 3.4 DP 的 AllReduce 通信量（训练）

数据并行梯度同步：

```
梯度 AllReduce 通信量（每步）：
  = 2 × (dp-1)/dp × 模型参数量 × dtype_bytes

DeepSeek-V3（ZeRO-1）：
  模型大小 = 671B × 2 bytes (BF16) = 1.34 TB（原始）
  ZeRO-1 将优化器状态分片，但梯度仍然 AllReduce

  实际参数量 per PP rank = 671B / 16 (pp) = 42B
  per TP rank = 42B / 1 (tp=1) = 42B

  梯度 AllReduce 量 = 2 × 42B × 2 bytes ≈ 168 GB（每个 DP 组）

但 ZeRO-1 用 ReduceScatter 替代 AllReduce：
  每个 rank 只收 1/dp 的梯度
  通信量不变，但每卡只需存 1/dp 的优化器状态
```

### 3.5 EP 的 AllToAll 通信量（MoE）

MoE 的 EP 通信与 TP/DP/PP 完全不同，是 **AllToAll** 操作：

```
MoE 层 EP 通信流程：
  1. Dispatch（分发）：将 token 路由到其选择的 Expert 所在 GPU
     每个 token 选 top-K 个 expert（DeepSeek-V3：top-8）

  2. Expert 计算：各 GPU 在本地计算 FFN

  3. Combine（回收）：Expert 输出返回 token 原始 GPU
```

**AllToAll 通信量公式**：

$$\text{EP AllToAll 通信量（单层）} = 2 \times T_{local} \times H \times top\_K \times \text{dtype\_bytes}$$

其中：
- $T_{local}$：每个 GPU 本地 token 数量
- $H$：hidden size
- $top\_K$：每个 token 选择的 Expert 数
- 系数 2：Dispatch + Combine 各一次

**DeepSeek-V3 具体数值**（Prefill，EP32，batch=512 tokens/GPU，H=7168，top-K=8）：

```
单层 AllToAll = 2 × 512 × 7168 × 8 × 2 bytes
              = 2 × 58.7 MB ≈ 117 MB

全模型 MoE 层数（V3 共 61 个 MoE 层）：
  总通信量 ≈ 61 × 117 MB ≈ 7.1 GB（前向）
```

**关键优化：限制每个 token 最多路由到 M 个节点**

DeepSeek 官方：
> *"Each token is sent to at most 4 nodes"（V3：M=4）*

```
若不限制（token 可以到任意 8 个节点）：
  AllToAll 会触发大量跨节点 IB 通信

限制 M=4 后：
  只有 4 个节点的 IB 通道被占用
  其余 expert 在机内 NVLink 完成（快 24 倍）

DeepSeek V3 实现：
  256 个路由专家 → 每节点 32 个（8节点 EP）
  每个 token top-8 中，保证最多 4 个节点参与
  → 每次 AllToAll 最多有 4×8 = 32 个 IB 传输
```

---

## 四、DeepSeek 决策背后的系统性原理

### 4.1 决策框架：通信/计算比（Computation-to-Communication Ratio）

每种并行方式的本质权衡：

```
并行方式    通信特征              计算/通信比              适用场景
─────────────────────────────────────────────────────────────────────
TP          层内 AllReduce        低（每层同步）            机内高带宽
SP          Reduce-Scatter+AG     与 TP 相同               同 TP
PP          层间 P2P              高（批次间异步）          跨机器
DP          步间 AllReduce        最高（可完全重叠）        多机扩展
EP          步内 AllToAll         中等（前向路径）          MoE 专用
```

**为什么 DeepSeek-V3 训练选择 PP16 而非 PP4？**

```
PP=4（4机）时：
  每个 Stage 平均 17 层（671B / 16 experts × 61 MoE 层 / 4）
  Bubble ≈ 3/num_microbatches

PP=16（16机）时：
  每个 Stage 更少层 → 更细粒度流水线
  DualPipe 填充 Bubble：bubble ratio ≈ 0（接近）
  更多节点 → 可以用 ZeRO-1 DP 扩展更大 batch size

权衡：PP 越深，Bubble 越大，但 DualPipe 解决了 Bubble 问题
     → V3 可以用更深的 PP 而不损失效率
```

### 4.2 DeepSeek-V3 为何推理 Prefill 用 TP4 而非 TP8？

官方原文：
> *"Its small TP size of 4 limits the overhead of TP communication."*

```
TP=4 vs TP=8 的通信开销对比（Prefill，tokens=4096，hidden=7168）：

单层 AllReduce 数据量：
  TP=4：2 × 4096 × 7168 × 2 bytes = 118 MB
  TP=8：2 × 4096 × 7168 × 2 bytes = 118 MB（数据量不变！）

通信时间（机内 NVLink 160 GB/s）：
  TP=4：118 / 160 ≈ 0.74 ms × 2 次/层 × 3 层 ≈ 4.4 ms/step
  TP=8：118 / 160 ≈ 0.74 ms（数据量相同但需要 8 卡协调，Ring 步骤更多）

TP=4 的优势：
  ✓ 4 张卡已足够装下 MLA 的 KV Cache（MLA 大幅压缩了 KV 大小）
  ✓ Ring-AllReduce 步骤更少（tp-1=3 步 vs 7 步）
  ✓ 剩余 4 张卡可以用来做 DP（DP8 = 2 组 TP4，吞吐翻倍）
```

### 4.3 为什么 Decode 用 EP320 而非 EP32？

```
Decode 阶段特点：
  每步只有 batch_size 个 token（每请求 1 个）
  例如 batch=320：每个 EP GPU 只有 1 个 token！

EP320 的逻辑：
  320 张 GPU，256 个路由专家 + 64 个冗余专家
  每张 GPU 最多 1 个专家

  → 消除了 AllToAll 通信！（每个 token 的目标 expert 就在某张 GPU 上）
  → 退化为点对点（P2P）通信，延迟极低
  → IBGDA 进一步降低延迟（GPU 直接发起 RDMA，绕过 CPU）

对比 EP32 + Decode：
  EP32 时每张 GPU 有 10 个专家
  token → expert 需要 AllToAll（可能跨 GPU）
  延迟 >> P2P 直连
```

### 4.4 DualPipe：解决 PP 与 EP 的通信冲突

DeepSeek-V3 同时使用 PP16 和 EP64，这两种通信会**竞争 InfiniBand 带宽**：

```
标准 1F1B 调度的问题：
  PP 的 P2P：Stage 间传输激活值，占用 IB
  EP 的 AllToAll：Expert 间传输 token，占用 IB
  → 两者在同一时间段竞争，互相阻塞

DualPipe 的解决方案：
  将每个 chunk 拆分为 4 部分：
    [Attention] [AllToAll Dispatch] [MLP] [AllToAll Combine]

  双向调度：
    当 chunk A 在执行 Attention（无通信）时
    → chunk B 同时执行 AllToAll Dispatch（通信）

    当 chunk A 在执行 AllToAll Combine（通信）时
    → chunk B 同时执行 MLP（纯计算）

  → 计算与通信完美重叠，IB 带宽利用率接近 100%
  → Pipeline Bubble ≈ 0（双向流水消除气泡）
```

```
DualPipe 调度示意（简化）：

时间轴 →
Stage 0: [F1:Attn][F1:Dispatch][F1:MLP][F1:Combine][F2:Attn]...
          ←← 同时 ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
Stage 7: [B7:Attn][B7:Dispatch][B7:MLP][B7:Combine][F7:Attn]...

F=正向 microbatch，B=反向 microbatch，双端同时注入
```

---

## 五、各并行方式通信量汇总对比

### 5.1 公式速查表

| 并行类型 | 通信原语 | 每步通信量（每卡）| 同步性 |
|---------|---------|----------------|--------|
| **TP AllReduce** | AllReduce | $2 \times \frac{tp-1}{tp} \times B \times H \times 2$ bytes（每层×2次）| 同步阻塞 |
| **TP+SP** | RS + AG | 同上（但激活显存减少 TP 倍）| 同步阻塞 |
| **PP P2P** | Send/Recv | $B \times H \times 2$ bytes（每层边界一次）| 流水异步 |
| **DP AllReduce** | AllReduce | $2 \times \frac{dp-1}{dp} \times \frac{params}{pp \times tp} \times 2$ bytes | 步间异步 |
| **EP AllToAll** | AllToAll | $2 \times T_{local} \times H \times top\_K \times 2$ bytes（每 MoE 层）| 前向路径同步 |

变量说明：
- $B$：batch×seq_len（tokens 数）
- $H$：hidden_size
- $T_{local}$：每卡本地 token 数（= B/EP）
- $top\_K$：每 token 选的 Expert 数

### 5.2 DeepSeek-V3 的具体通信量估算

**配置**：训练时 PP=16，EP=64，DP=ZeRO-1，TP=1，hidden=7168，num_layers=61

```
单步训练（1 个 microbatch，tokens=2048）：

1. TP AllReduce：TP=1 → 通信量 = 0（无 AllReduce）

2. PP P2P（前向 + 反向）：
   每个 Stage 边界：2048 × 7168 × 2 bytes = 28 MB（单向）
   PP=16 → 15 个边界 × 2方向 × 2（前向+反向）= 60 次传输
   总 P2P ≈ 60 × 28 MB = 1.68 GB

3. EP AllToAll（每个 MoE 层，61 层）：
   单层：2 × (2048/64) × 7168 × 8（top-8）× 2 bytes
       = 2 × 32 × 7168 × 8 × 2 bytes = 7.3 MB × 2 = 14.6 MB
   61 层：61 × 14.6 MB = 890 MB ≈ 0.9 GB

4. DP AllReduce（ZeRO-1，每步一次）：
   参数量 per PP rank ≈ 671B / 16 = 42B 参数
   梯度通信 ≈ 2 × 42B × 2 bytes = 168 GB
   （但 ZeRO-1 用 ReduceScatter：每卡只收 168GB/DP）

总通信量（前向单步）：~0 + 1.68 + 0.9 ≈ 2.58 GB
（DualPipe 使 EP AllToAll 与 PP P2P 几乎完全重叠，实际等效通信 ≈ max(1.68, 0.9) ≈ 1.68 GB）
```

---

## 六、从硬件到决策的完整因果链

```
H800 集群特性
├── 节点内：NVLink，400 GB/s（双向总带宽）
│   └── → TP 只能在节点内（≤8卡），TP 大小不超过 8
│
├── 节点间：InfiniBand，400 Gbps ≈ 50 GB/s 单向（per NIC）
│   └── → PP 和 EP 的跨节点通信走 IB
│
└── GPU 显存：80 GB HBM3/卡

        ↓ 约束

DeepSeek-V3 模型特性
├── 671B 参数（MoE）→ 单卡装不下，必须多卡
├── MLA 压缩 KV → 显存需求大幅减少，TP 不再必要
├── 256 专家（8 active/token）→ EP 必须足够大
│
└── 训练 global_batch = 15.36M tokens → 需要大 DP

        ↓ 推导

训练并行决策
├── TP=1：MLA 已节省显存，TP AllReduce 代价高，消除之
├── PP=16：671B / 16 = 42B per stage，单节点可装；DualPipe 消除 Bubble
├── EP=64：256专家 / 64 GPU = 4专家/GPU；跨8节点，限制路由≤4节点
└── ZeRO-1 DP：剩余 GPU 做 DP，优化器状态分片

        ↓ 推导

推理并行决策（Prefill）
├── TP=4+SP：限制 AllReduce 开销；4卡 NVLink 内完成
├── EP=32：单节点内 MoE expert 分布，最小化跨节点 AllToAll
└── DP=8：同一节点 2组 TP4，服务更多并发请求

推理并行决策（Decode）
├── TP=4+SP：与 Prefill 相同
├── EP=320（每卡1专家）：AllToAll → P2P，极致降低 Decode 延迟
└── IBGDA：GPU 直接 RDMA，绕过 CPU，进一步降低 P2P 延迟
```

---

## 七、总结：工业级决策规则

| 规则 | 原理 | DeepSeek 实例 |
|------|------|--------------|
| **TP ≤ 节点内 GPU 数** | TP AllReduce 需要低延迟高带宽 | V3 推理 TP=4（节点内 8 卡的一半） |
| **训练优先消除 TP** | 通过架构优化（MLA/MoE）减少显存需求 | V3 训练 TP=1 |
| **PP 用 DualPipe 填充 Bubble** | 双向流水线使 Bubble ≈ 0 | V3 PP=16，接近零 Bubble |
| **EP 按节点边界对齐** | 减少跨节点 AllToAll 比例 | V3 EP=64=8节点×8卡，限路由≤4节点 |
| **Decode 极限 EP** | EP=卡数时，AllToAll 退化为 P2P | V3 Decode EP=320 |
| **Prefill/Decode 分离** | 两个阶段特性完全不同（计算密集 vs 内存密集）| V3 分别 32GPU / 320GPU 实例 |
| **通信与计算重叠** | DualPipe：EP AllToAll 与 PP P2P 时间上错开 | V3 有效通信量 ≈ max(PP, EP) 而非 PP+EP |

---

*参考文献：*
- *[DeepSeek-V3 Technical Report](https://arxiv.org/html/2412.19437v1)*
- *[DeepSeek-V2 Technical Report](https://arxiv.org/html/2405.04434v2)*
- *[DeepSeek Open-Infra-Index 推理系统概览](https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md)*
- *[DualPipe 开源代码库](https://github.com/deepseek-ai/DualPipe)*
- *[Ring-AllReduce 数学推导（OneFlow）](https://oneflow2020.medium.com/how-to-derive-ring-all-reduces-mathematical-property-step-by-step-9951500db96)*
- *[Megatron-LM TP 原理](https://docs.pytorch.org/tutorials/intermediate/TP_tutorial.html)*
- *[NVIDIA MoE EP 通信优化](https://developer.nvidia.com/blog/optimizing-communication-for-mixture-of-experts-training-with-hybrid-expert-parallel/)*
*更新：2026-03*
