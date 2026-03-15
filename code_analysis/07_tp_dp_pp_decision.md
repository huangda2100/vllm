# 分布式训练与推理中 TP/DP/PP 的决策原理

> 涉及文件：`vllm/config/parallel.py`、`vllm/distributed/parallel_state.py`、`vllm/model_executor/layers/linear.py`
> 核心问题：TP、DP、PP 的取值由什么决定？背后的原理是什么？训练和推理有何区别？

---

## 一、为什么需要并行？两个根本约束

在回答"取值由什么决定"之前，先要理解**为什么需要并行**，它来自两个根本性的硬件限制：

### 约束 1：显存容量（Memory Wall）

```
LLaMA-3-70B 的参数量：
  70B × 2 bytes (BF16) = 140 GB

单张 A100：80 GB 显存
→ 140 GB > 80 GB，单卡装不下，必须多卡

训练时还需要：
  参数：   140 GB (BF16)
  梯度：   140 GB (BF16)
  优化器状态（Adam）：
    momentum:  140 GB (FP32)
    variance:  140 GB (FP32)
  总计：  ≈ 700 GB

4卡 A100 (320 GB) 也装不下，需要 ZeRO 或 PP 分散
```

### 约束 2：计算时间（Compute Wall）

```
LLaMA-3-70B 单步前向传播的 FLOP 数：
  ≈ 2 × num_params × seq_len = 2 × 70B × 2048 ≈ 286 TFLOPs

A100 理论峰值：312 TFLOPs (BF16)
→ 即使 100% 利用率，单卡单步 ≈ 1 秒

实际训练需要迭代数百万步 → 必须多卡加速
```

**并行的本质**：用多卡的**显存之和**解决内存瓶颈，用多卡的**算力之和**解决计算瓶颈，用多卡的**带宽之和**解决通信瓶颈。

---

## 二、三种并行方式的本质

### 2.1 数据并行（Data Parallelism，DP）

**核心思想**：模型不切分，数据切分。每张卡持有一份**完整模型**，处理不同的数据。

```
Batch = [样本1, 样本2, ..., 样本N]
            ↓ 切分
GPU 0：[样本1, ..., 样本N/4]  → 独立前向+反向
GPU 1：[样本N/4+1, ..., 样本N/2]  → 独立前向+反向
GPU 2：[样本N/2+1, ..., 样本3N/4]  → 独立前向+反向
GPU 3：[样本3N/4+1, ..., 样本N]  → 独立前向+反向

AllReduce 梯度 → 梯度求平均 → 每卡用相同梯度更新权重
```

**显存需求**：每卡需要存放完整模型 + 梯度 + 优化器状态，**不能解决显存不足问题**。

**计算加速**：吞吐量随 DP 数量线性扩展，`global_batch_size = micro_batch_size × DP`。

**通信代价**：每步一次 `AllReduce`（梯度），通信量 = 参数量 × 2（Reduce + Broadcast）。

```
AllReduce 通信量（Ring-AllReduce）：
  发送 = 接收 = 2 × (dp_size-1)/dp_size × param_bytes
  ≈ 2 × 140 GB ≈ 280 GB  （70B 模型，BF16）

NVLink 带宽：600 GB/s (A100)
→ 通信时间 ≈ 280 / 600 ≈ 0.47 秒
→ 可与计算重叠（overlap），实际开销接近0
```

### 2.2 张量并行（Tensor Parallelism，TP）

**核心思想**：把**单个矩阵乘法**切开，多卡协作完成一次前向。

```
矩阵乘法：Y = X · W
  W: [hidden_size, ffn_size] = [4096, 16384]

TP=4 时按列切分（ColumnParallel）：
  GPU 0: W₀ = W[:, 0:4096]，计算 Y₀ = X · W₀  → [tokens, 4096]
  GPU 1: W₁ = W[:, 4096:8192]，计算 Y₁ = X · W₁  → [tokens, 4096]
  GPU 2: W₂ = W[:, 8192:12288]，计算 Y₂ = X · W₂  → [tokens, 4096]
  GPU 3: W₃ = W[:, 12288:16384]，计算 Y₃ = X · W₃  → [tokens, 4096]

Concat → Y = [Y₀, Y₁, Y₂, Y₃] = [tokens, 16384]
```

**显存节省**：每卡只存 1/TP 的权重，`显存 ∝ 1/TP`。

**通信代价**：
- ColumnParallel：前向无需通信（各自独立），反向需要 `AllReduce`
- RowParallel：前向需要 `AllReduce`（结果求和），反向无需通信
- 每个 Transformer 层：**2次 AllReduce**（Attention 的 o_proj + MLP 的 down_proj）

```
TP AllReduce 通信量（每个 Transformer 层）：
  2 × [tokens, hidden_size] × 2 bytes (BF16)
  = 2 × 2048 × 4096 × 2 ≈ 32 MB

NVLink 带宽：600 GB/s
→ 通信时间 ≈ 32 MB / 600 GB/s ≈ 0.05 ms（极小！）
```

**关键约束**：TP 通信必须**低延迟、高带宽**，因为它在**每一层**都同步通信，无法异步。**TP 只适合在 NVLink 直连的同机 GPU 之间使用**，跨机 TP（InfiniBand）延迟过高。

### 2.3 流水线并行（Pipeline Parallelism，PP）

**核心思想**：把模型的**层**切开，不同 GPU 负责不同阶段，数据像流水线一样流动。

```
32层 Transformer，PP=4：
  GPU 0（Stage 0）：Embedding + Layers 0~7
  GPU 1（Stage 1）：Layers 8~15
  GPU 2（Stage 2）：Layers 16~23
  GPU 3（Stage 3）：Layers 24~31 + LM Head

数据流：
  Micro-batch 1: GPU0 → GPU1 → GPU2 → GPU3 → 输出
                     ↑P2P         ↑P2P         ↑P2P
```

**显存节省**：每卡只存 1/PP 的模型层，**但 Embedding、优化器状态等仍需考虑**。

**通信代价**：Stage 间 P2P 传输 `[tokens, hidden_size]` 的激活值，通信量远小于 TP。

**核心问题：Pipeline Bubble（流水线气泡）**

```
朴素流水线（GPipe）：
  GPU 0：[F1][F2][F3][F4][idle    ][B4][B3][B2][B1]
  GPU 1：    [F1][F2][F3][F4][idle ][B4][B3][B2][B1]
  GPU 2：        [F1][F2][F3][F4][B4][B3][B2][B1]
  GPU 3：            [F1][F2][F3][F4][B4][B3][B2][B1]

  Bubble 比例 = (pp_size - 1) / (num_microbatches)

  PP=4, micro_batches=4：Bubble = 3/4 = 75%！（极大浪费）
  PP=4, micro_batches=16：Bubble = 3/16 ≈ 19%（可接受）
  PP=4, micro_batches=64：Bubble = 3/64 ≈ 5%（良好）
```

**结论**：PP 需要足够多的 micro-batches 来填满流水线，`num_microbatches >> pp_size`。

---

## 三、决定 TP/DP/PP 取值的核心因素

### 因素 1：显存是否装得下（首要约束）

这是**最硬的约束**，其他考虑都是在满足显存约束后的优化。

```python
# 估算所需显存
def estimate_memory(model_params_B, dtype_bytes=2, parallel_config=None):
    # 单个 rank 的模型参数显存
    model_memory = model_params_B * 1e9 * dtype_bytes / (TP * PP)

    # 训练时还需要梯度和优化器状态（ZeRO未启用时）
    if training:
        optimizer_memory = model_params_B * 1e9 * 12  # Adam: 4+4+4 bytes/param (FP32)
        total = model_memory + optimizer_memory
    else:  # 推理
        total = model_memory

    return total / (1024**3)  # GB
```

**实际规则（推理）**：

| 模型规模 | BF16 大小 | 最小显存方案 |
|---------|---------|------------|
| 7B  | 14 GB | 单卡 A100 80GB（还有余量给 KV Cache） |
| 13B | 26 GB | 单卡 A100 80GB（紧张） |
| 34B | 68 GB | 单卡 A100 80GB（非常紧张，TP=1 可行）|
| 70B | 140 GB | TP=2，双卡 A100（各 70GB） |
| 175B | 350 GB | TP=4，4卡 A100；或 TP=8 |
| 405B | 810 GB | TP=8（640GB）+ PP=2，或 TP=16 |

### 因素 2：互联拓扑（决定 TP 上限）

这是 **TP 的核心限制因素**。

```
同机 GPU 互联带宽（NVLink）：
  A100 NVLink 3.0：600 GB/s 双向
  H100 NVLink 4.0：900 GB/s 双向

跨机 GPU 互联带宽（InfiniBand）：
  HDR InfiniBand：200 Gbps ≈ 25 GB/s 单向
  NDR InfiniBand：400 Gbps ≈ 50 GB/s 单向

带宽差距：NVLink / InfiniBand ≈ 600 / 25 = 24x
```

**TP 的通信在每层都发生（同步阻塞）**，而跨机通信带宽是机内的 1/24：

```
单层 TP AllReduce 时间：
  NVLink：32 MB / 600 GB/s = 0.05 ms  → 可接受（计算时间 >> 通信时间）
  InfiniBand：32 MB / 25 GB/s = 1.3 ms → 严重拖累（通信 >> 计算！）
```

**结论**：**TP 必须限制在同一台物理机（NVLink 域）内**。通常：
- 8 卡 DGX/HGX 节点：`TP ≤ 8`（一台机器内）
- 4 卡服务器：`TP ≤ 4`
- 跨机扩展：使用 PP 或 DP，不用 TP

### 因素 3：模型结构的整除约束

TP 要求模型维度能被 TP 整除：

```python
# vllm/model_executor/layers/linear.py
def ensure_divisibility(numerator, denominator):
    assert numerator % denominator == 0

# 具体约束：
num_attention_heads % tp_size == 0       # 注意力头数
num_kv_heads % tp_size == 0              # KV 头数（GQA 时）
ffn_hidden_size % tp_size == 0           # FFN 隐藏层
vocab_size % tp_size == 0                # 词表大小（近似，有 padding）
```

**实例**：

```
LLaMA-3-8B：
  num_heads = 32, num_kv_heads = 8
  合法 TP：1, 2, 4, 8（都能整除）
  非法 TP：3, 5, 6, 7

Mistral-7B-v0.3：
  num_heads = 32, num_kv_heads = 8
  合法 TP：1, 2, 4, 8

Phi-3-small（num_heads=32, num_kv_heads=8）：
  合法 TP：1, 2, 4, 8

某些小模型 num_heads=12：
  合法 TP：1, 2, 3, 4, 6, 12
  TP=8 非法！（12 % 8 ≠ 0）
```

**vLLM 的进一步约束**（GQA）：

```python
# vllm/model_executor/models/llama.py
if self.total_num_kv_heads >= tp_size:
    assert self.total_num_kv_heads % tp_size == 0
else:
    # KV heads < TP：每组 tp_size/num_kv_heads 个 rank 共享同一 KV head
    assert tp_size % self.total_num_kv_heads == 0
```

### 因素 4：Batch Size 与延迟需求

**训练场景**：

```
global_batch_size = micro_batch_size × gradient_accumulation_steps × DP

PP 需要 micro_batches >> pp_size 才能有效填满流水线：
  经验法则：num_microbatches ≥ 4 × pp_size（Bubble < 25%）

如果 micro_batch_size=2, global_batch_size=1024：
  DP × grad_acc × 2 = 1024

  若 PP=4 需要 num_microbatches ≥ 16：
  → grad_acc ≥ 16，DP = 1024 / (16×2) = 32

如果 global_batch_size 较小（如 64），PP 效率很低：
  PP=4, micro_batch_size=2：num_microbatches = 64/8/2 = 4 → Bubble = 75%！
  → 此时 PP 不合适
```

**推理场景（online serving）**：

```
推理通常追求低延迟：
  - 小 batch（实时请求）：更关注 TTFT（Time To First Token）
  - 大 batch（批量推理）：更关注吞吐量

PP 引入的流水线 bubble 在推理中=延迟增加：
  request latency ≈ num_layers/pp_size × layer_time + (pp_size-1) × inter_stage_comm

→ 推理优先用 TP（减少每层延迟），PP 在模型装不下时作为补充
```

### 因素 5：通信计算比（Compute-to-Communication Ratio）

**计算强度**（Arithmetic Intensity）决定了通信开销是否可被掩盖：

```
矩阵乘法计算量：FLOPs = 2 × M × N × K
矩阵乘法内存访问：bytes = (M×K + K×N + M×N) × dtype_bytes

TP AllReduce 通信量：2 × (1 - 1/tp_size) × M × N × dtype_bytes

通信可被掩盖的条件：
  comm_time < compute_time
  → (comm_bytes / bandwidth) < (FLOPs / throughput)
  → comm_bytes × throughput < FLOPs × bandwidth

以单个 FFN Down Projection 为例（hidden=4096, ffn=16384, tokens=2048）：
  FLOPs = 2 × 2048 × 4096 × 16384 ≈ 274 TFLOPs
  Compute Time（A100）= 274 / 312 ≈ 0.88 ms

  AllReduce 通信量 = 2 × 2048 × 4096 × 2 bytes ≈ 32 MB
  NVLink 通信时间 = 32 / 600,000 ms ≈ 0.05 ms

  → 通信时间（0.05 ms）<< 计算时间（0.88 ms）✓
  → TP 通信可以完全重叠（NVLink 下）

但 Decode 阶段 batch=1 时：
  tokens = 1（每步只生成1个 token）
  FLOPs = 2 × 1 × 4096 × 16384 ≈ 134 GFLOPs → 计算时间 ≈ 0.00043 ms
  通信时间 ≈ 0.05 ms >> 计算时间
  → TP 通信主导延迟！decode 时 TP 开销比例极高
```

**这是推理场景下 TP 的核心权衡**：
- TP 减少 KV Cache 占用（分摊到多卡）→ 支持更多并发
- TP 增加通信延迟 → decode 阶段每步都有 AllReduce 开销
- 实践中：TP=2~4 通常是甜点，TP=8 在小 batch 下可能变慢

---

## 四、Rank 映射的底层原理

### vLLM 的四维 Rank 组织

```python
# vllm/distributed/parallel_state.py
all_ranks = torch.arange(world_size).reshape(
    -1,                               # external_dp_size
    data_parallel_size,               # dp_size
    pipeline_model_parallel_size,     # pp_size
    tensor_model_parallel_size,       # tp_size
)
# Shape: [ext_dp, dp, pp, tp]
```

**Layout 顺序：external_dp × DP × PP × TP（最内层 = 同机最快通信的维度）**

这个顺序不是随机的，它对应了典型集群的网络拓扑：

```
物理集群拓扑（一个典型的 8×4GPU 集群）：

机器 0：[GPU 0, 1, 2, 3] ← NVLink 互联（600 GB/s）
机器 1：[GPU 4, 5, 6, 7]
机器 2：[GPU 8, 9,10,11]
机器 3：[GPU 12,13,14,15] ← 机间 InfiniBand（25 GB/s）

TP=4（同机）→ 占用 NVLink（最快）
PP=4（跨机）→ 只有 P2P，占用 InfiniBand（可接受）
DP=2（数据复制）→ 只在梯度更新时通信（可异步）
```

### 具体 Rank 映射示例

配置：`world_size=16, DP=2, PP=2, TP=4`，`all_ranks.shape = [1, 2, 2, 4]`：

```
all_ranks[0] =
  DP=0：
    PP=0：[0,  1,  2,  3]   ← TP group A（机器0的4张卡）
    PP=1：[4,  5,  6,  7]   ← TP group B（机器1的4张卡）
  DP=1：
    PP=0：[8,  9, 10, 11]   ← TP group C（机器2的4张卡）
    PP=1：[12, 13, 14, 15]  ← TP group D（机器3的4张卡）

TP groups（同机，NVLink）：
  [0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]

PP groups（DP0和DP1分别有各自的 PP 流水线）：
  [0,4], [1,5], [2,6], [3,7]      ← DP0的PP流
  [8,12], [9,13], [10,14], [11,15] ← DP1的PP流

DP groups（梯度同步）：
  [0,8], [1,9], [2,10], [3,11],   ← PP0-TP0,1,2,3各自的DP对
  [4,12], [5,13], [6,14], [7,15]
```

### vLLM 中进程组创建的代码原理

```python
# TP group：最内层维度 → 每行4个rank一组
tp_group_ranks = all_ranks.view(-1, tp_size).unbind(0)
# → [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15]]

# PP group：沿 PP 维度切片，固定 DP 和 TP 位置
pp_group_ranks = all_ranks.transpose(2,3).reshape(-1, pp_size).unbind(0)
# all_ranks.transpose(2,3) 将 [ext,dp,pp,tp] → [ext,dp,tp,pp]
# reshape(-1,pp_size) → 每行 = 一条流水线上各 stage 的 rank

# DP group：梯度同步组，相同 PP_rank + TP_rank，不同 DP_rank
dp_group_ranks = all_ranks.transpose(1,3).reshape(-1, dp_size).unbind(0)
```

---

## 五、训练 vs 推理的并行策略差异

### 训练侧的特殊考虑

**ZeRO（零冗余优化器）**：

```
ZeRO-1：分片优化器状态
  DP rank i 只存 1/DP 的优化器状态
  → 优化器状态显存节省 DP 倍

ZeRO-2：分片优化器状态 + 梯度
  → 额外节省 DP 倍的梯度显存

ZeRO-3：分片优化器状态 + 梯度 + 模型参数
  → 全部显存均分，类似 TP 但通信模式不同
  → 每次前向/反向都需要 AllGather 参数
```

ZeRO 本质上是 **DP + 参数分片**，可以代替 TP 的显存节省作用，但通信模式不同（ZeRO 通信量更大但可以流水化）。

**激活重计算（Activation Checkpointing）**：

```
前向传播时只保存部分层的激活值，反向传播时重新计算被丢弃的激活
→ 激活显存降低约 5x（全部重计算）
→ 计算量增加约 33%（额外的前向）

与 PP 的交互：
  每个 stage 需要存 (num_microbatches - 1) 个中间激活
  → 启用激活重计算可以降低 PP 的显存峰值
```

**梯度累积**：

```
num_grad_acc_steps × micro_batch_size = global_batch_size / DP

PP 需要 num_microbatches = num_grad_acc_steps 足够大以填满流水线
→ global_batch_size 越大，PP 效率越高
→ 小 batch 训练（如 RLHF 中）PP 效率很低
```

### 推理侧的特殊考虑

**KV Cache 的显存占用**：

```
KV Cache 大小 = batch_size × seq_len × num_layers × 2 × num_kv_heads × head_dim × 2 bytes

LLaMA-3-70B，batch=32，seq=2048，TP=1：
  = 32 × 2048 × 80 × 2 × 8 × 128 × 2 bytes
  ≈ 86 GB！（远超显存）

TP=4 时，num_kv_heads 被切分为 8/4 = 2：
  KV Cache = 32 × 2048 × 80 × 2 × 2 × 128 × 2 bytes ≈ 21.5 GB（每卡）
  → TP 不仅减少了模型权重显存，也减少了 KV Cache！
```

**推理的 Prefill-Decode 分离**：

```
Prefill（prompt 处理）：
  - Compute-Bound，类似训练的前向传播
  - 大 batch 有利，TP 通信可被计算掩盖
  - PP 的 bubble 在 prefill 中是绝对的延迟增加

Decode（token 生成）：
  - Memory-Bound，每步只有 1 个 token
  - TP 的 AllReduce 开销 >> 实际计算
  - PP 每步都要等 stage 间传输 → 额外延迟

→ 推理更倾向于少用 PP，TP 是主要的多卡并行方式
```

**推理的实时性要求（TTFT/ITL）**：

```
TTFT（Time To First Token）= Prefill 时间
  - TP 缩短 Prefill（每层计算量 ÷ TP）
  - PP 增加 Prefill（流水线 bubble）

ITL（Inter-Token Latency）= Decode 每步时间
  - TP 增加 AllReduce 开销（decode batch 小）
  - TP 减少每卡 KV Cache 占用 → 支持更大 batch → 更高吞吐

→ 延迟优先：使用最小的 TP（但能装下模型）
→ 吞吐优先：适当增大 TP 以支持更大 batch
```

---

## 六、实践决策框架

### 步骤 1：检查显存约束（必须满足）

```
已知：num_GPU, GPU_memory, model_size, batch_size, seq_len

# 推理
min_tp × min_pp ≥ ceil(model_size_GB / (GPU_memory_GB × 0.7))
# 70% 是保守估计，留 30% 给 KV Cache 和其他

# 训练（不用 ZeRO）
min_tp × min_pp ≥ ceil(model_size_GB × 12 / GPU_memory_GB)
# 12× 因子 = 参数(2B) + 梯度(2B) + Adam状态(8B) 总 12 bytes/param

# 训练（用 ZeRO-3，DP 参与显存分片）
min_tp × min_pp ≥ ceil(model_size_GB × 12 / (GPU_memory_GB × DP))
```

### 步骤 2：检查整除约束（硬约束）

```python
assert num_attention_heads % tp_size == 0
assert ffn_intermediate_size % tp_size == 0
assert num_layers % pp_size == 0  # PP 要求层数均匀分配
assert vocab_size % tp_size == 0  # 近似，vLLM 有 padding
```

**TP 合法取值**：`num_attention_heads` 的因子集合

```python
def valid_tp_sizes(num_heads):
    return [i for i in range(1, num_heads+1) if num_heads % i == 0]

valid_tp_sizes(32)  # = [1, 2, 4, 8, 16, 32]
valid_tp_sizes(40)  # = [1, 2, 4, 5, 8, 10, 20, 40]
```

### 步骤 3：检查互联约束（TP 的拓扑约束）

```
tp_size ≤ 同机 GPU 数（NVLink 域内）

典型值：
  DGX A100（8卡）：tp_size ≤ 8
  DGX H100（8卡）：tp_size ≤ 8
  4卡工作站：tp_size ≤ 4
  消费级 GPU（无 NVLink）：实际上 tp_size = 1 最优（PCIe 太慢）
```

### 步骤 4：确定 PP 是否需要（推理）

```
# 只有当模型装不下单台机器时才用 PP
if model_size_GB <= num_GPUs_per_node × GPU_memory_GB × 0.7:
    pp_size = 1  # 不需要 PP
else:
    pp_size = ceil(model_size_GB / (num_GPUs_per_node × GPU_memory_GB × 0.7))
    # PP 跨机器

# 例：405B 模型（810 GB），4台 8×A100（320 GB/机）：
#   810 / 320 = 2.53 → pp_size = 3（或 4 对齐）
#   tp_size = 8（单机内）
#   总 GPU = 8 × 4 = 32
```

### 步骤 5：确定 DP（吞吐 vs 延迟权衡）

```
# 推理：DP 通常不用（各 DP 副本独立服务请求，由负载均衡层处理）
dp_size = 1  # vLLM 实例本身不做 DP

# 训练：满足显存约束后，剩余算力全用 DP
dp_size = total_GPUs / (tp_size × pp_size)

# 需要检查是否满足 batch size 要求
global_batch_size = micro_batch_size × grad_acc_steps × dp_size
assert global_batch_size == target_global_batch_size
```

---

## 七、经典配置案例

### 案例 1：LLaMA-3-8B 推理（A100 80GB × 1）

```
模型大小：16 GB (BF16)
单卡显存：80 GB
KV Cache 需求（32 并发, 8K 上下文）：≈ 20 GB

方案：
  TP=1, PP=1, DP=1（单卡）
  模型：16 GB
  KV Cache：20 GB
  其余：80 - 36 = 44 GB 可用
```

### 案例 2：LLaMA-3-70B 推理（A100 80GB × 8）

```
模型大小：140 GB (BF16)
8卡总显存：640 GB

方案一（低延迟优先）：
  TP=4, PP=1, DP=2
  每卡模型：140/4 = 35 GB
  KV Cache 每卡（TP=4 下 num_kv_heads=2）：≈ 15 GB
  → 每卡使用约 50 GB，有余量 ✓
  → DP=2 提供 2x 吞吐 ✓

方案二（超大 batch 推理）：
  TP=8, PP=1, DP=1
  每卡模型：140/8 = 17.5 GB
  更多显存给 KV Cache → 支持更大并发
  但 decode 时 TP AllReduce 开销增加

推荐：TP=4, PP=1（单机 NVLink）
```

### 案例 3：LLaMA-3-405B 推理（A100 80GB × 32，4台机）

```
模型大小：810 GB (BF16)
32卡总显存：2560 GB

方案：
  TP=8（单机 NVLink，整除 num_heads=128）
  PP=4（4台机器各一个 stage，跨机 InfiniBand）
  DP=1

每台机器：8卡 × 80 GB = 640 GB
每 stage 模型层数：128 / 4 = 32 层
每卡模型权重：810 / 32 = 25.3 GB
  → 剩余 54.7 GB 给 KV Cache ✓
```

### 案例 4：LLaMA-3-70B 训练（H100 80GB × 64，8台机）

```
模型大小：140 GB (BF16)
训练全精度（FP32 Adam）：140 × 6 = 840 GB

方案（无 ZeRO）：
  TP=8（单机 NVLink）
  PP=2（两机 stage）
  DP=4（剩余4组做数据并行）
  64 / (8 × 2) = 4 ✓

每卡显存：
  模型权重：140 / (8×2) = 8.75 GB
  梯度：8.75 GB
  Adam 状态（FP32）：17.5 GB
  激活（需要重计算）：约 5 GB
  合计：≈ 40 GB < 80 GB ✓

Bubble 填充：
  num_microbatches = 16（grad_acc_steps）
  PP=2：Bubble = 1/16 = 6.25% ✓

global_batch_size：
  = micro_batch_size × 16 × DP=4 = 64 × micro_batch_size
  micro_batch_size=2 → global_batch_size=128 tokens/batch
```

---

## 八、常见错误与陷阱

### 陷阱 1：跨机使用 TP

```
错误配置（16卡，2台机器，各8卡）：
  TP=16, PP=1, DP=1  ← TP 跨机！

问题：
  每层 2 次 AllReduce，通信量 = 2 × [tokens, hidden]
  跨机带宽：InfiniBand 25 GB/s
  通信时间：0.05s × (25/600) × ... >> 计算时间

正确配置：
  TP=8 (单机), PP=2 (跨机), DP=1
```

### 陷阱 2：PP 的 micro_batch 不足

```
错误配置（训练）：
  PP=8, global_batch_size=128, micro_batch_size=2
  num_microbatches = 128 / (2 × DP) = 64/DP
  若 DP=4：num_microbatches = 16
  Bubble = (8-1)/16 = 43.75%（几乎一半时间在等待）

正确配置：
  PP=8 时需要 num_microbatches >> 8
  增大 grad_acc_steps 或增大 global_batch_size
```

### 陷阱 3：TP 不整除 GQA 的 KV heads

```
错误配置：
  模型：num_kv_heads=4（如 Llama-3-8B 变体）
  TP=8

问题：
  4 % 8 ≠ 0，且 8 % 4 = 0（可以复制）
  但每个 TP rank 会复制一份 KV head，KV Cache 没有节省
  → 增加了 TP 通信开销，却没有节省 KV Cache 显存

分析：当 tp_size > num_kv_heads 时，TP 对 KV Cache 没有帮助，
      应优先选择 tp_size = num_kv_heads 的因子
```

### 陷阱 4：PP 在推理中被滥用

```
小模型 + PP 的错误：
  LLaMA-3-8B，16 GB，4卡（单台机）
  配置：TP=2, PP=2（没必要，内存完全够用）

问题：
  PP 每步增加 (pp_size-1) × inter_stage_latency 的延迟
  对于推理流式输出（streaming），用户感知到明显的首 token 延迟增加

正确：
  TP=4（充分利用 NVLink）, PP=1（单机够用）
```

---

## 九、vLLM 中的约束代码对照

```python
# vllm/config/parallel.py
world_size = pipeline_parallel_size * tensor_parallel_size
# 注意：DP 不计入 world_size（每个 vLLM 实例独立）

# vllm/distributed/parallel_state.py - initialize_model_parallel
# TP 组：同 DP_rank 和 PP_rank，不同 TP_rank
# PP 组：同 DP_rank 和 TP_rank，不同 PP_rank
# DP 组：同 PP_rank 和 TP_rank，不同 DP_rank

# vllm/model_executor/layers/linear.py
self.output_size_per_partition = divide(output_size, self.tp_size)
# → output_size % tp_size == 0（ColumnParallel）
self.input_size_per_partition = divide(input_size, self.tp_size)
# → input_size % tp_size == 0（RowParallel）

# 运行时 AllReduce（RowParallel forward）：
if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output)
```

---

## 十、总结：决策规则速查表

| 场景 | 首选策略 | 原因 |
|------|---------|------|
| **单卡能装下** | TP=PP=DP=1 | 无需并行，省去通信开销 |
| **同机多卡（NVLink）** | TP=GPU数，PP=DP=1 | NVLink 足够快，TP 效率最高 |
| **跨机扩展（InfiniBand）** | TP=单机GPU数，PP=机器数 | TP 不能跨机，PP 适合跨机 |
| **超大 batch 吞吐** | 增大 DP | DP 线性扩展吞吐，通信可异步 |
| **模型装不下单机** | 增大 PP | PP 跨机通信量少，延迟可接受 |
| **极低延迟推理** | 最小 TP（能装下模型） | 减少 decode 阶段 AllReduce |
| **训练大模型** | TP + PP + ZeRO | 三者结合，各司其职 |
| **GQA 模型** | TP ≤ num_kv_heads | 超过后 KV Cache 无法分片 |

---

*涉及文件：`vllm/config/parallel.py`、`vllm/distributed/parallel_state.py`、`vllm/model_executor/layers/linear.py`*
*更新：2026-03*
