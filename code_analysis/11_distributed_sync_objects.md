# 分布式并行：不同操作同步的对象与同步后的处理

> 核心问题：
> 1. 每种并行方式，**谁和谁同步**（同步对象是什么）？
> 2. **为什么要同步**（不同步会发生什么错误）？
> 3. **同步后拿到的是什么**，接下来怎么用？

---

## 零、前置概念：为什么分布式需要"同步"

分布式计算的根本矛盾：**每张 GPU 只有局部信息，但模型前向/反向需要全局信息**。

```
单卡计算（无需同步）：
  Y = X · W         X 完整，W 完整，Y 完整 ✓

TP（每卡只有 1/4 的 W）：
  GPU 0: Y₀ = X · W[0:1024]   → 只有部分结果（1/4 的列）
  GPU 1: Y₁ = X · W[1024:2048] → 只有部分结果
  GPU 2: Y₂ = X · W[2048:3072] → 只有部分结果
  GPU 3: Y₃ = X · W[3072:4096] → 只有部分结果

  真正的 Y = Y₀ + Y₁ + Y₂ + Y₃  ← 必须同步（AllReduce）才能得到
```

**同步**的本质：把分散在不同 GPU 上的**局部结果**合并成**全局正确结果**。

---

## 一、TP（张量并行）：同步激活值

### 1.1 前向传播同步

#### 同步对象：**部分激活值（Partial Activations）**

每张 GPU 持有权重矩阵的一部分（行或列切分），矩阵乘法的结果是**部分和（Partial Sum）**。

**列并行（Column Parallel）→ 不需要立即同步**

```
列并行：output_dim 切分（q/k/v/gate/up_proj）

GPU 0: X·W_col0 → Y₀: [tokens, 1024]   ← 只有前 1/4 列
GPU 1: X·W_col1 → Y₁: [tokens, 1024]   ← 只有 1/4~2/4 列
GPU 2: X·W_col2 → Y₂: [tokens, 1024]   ← 只有 2/4~3/4 列
GPU 3: X·W_col3 → Y₃: [tokens, 1024]   ← 只有后 1/4 列

concat([Y₀, Y₁, Y₂, Y₃]) = Y: [tokens, 4096]  ← 列拼接，无需通信！

原因：每个 GPU 负责输出的不同"列段"，互不依赖，直接拼接即可
```

**行并行（Row Parallel）→ 必须 AllReduce**

```
行并行：input_dim 切分（o_proj/down_proj）

输入 X（列并行的输出）已经被切分：
  GPU 0 有: X₀: [tokens, 1024]（前 1/4）
  GPU 1 有: X₁: [tokens, 1024]（中 1/4）
  ...

GPU 0: X₀·W_row0 → P₀: [tokens, 4096]  ← 部分和
GPU 1: X₁·W_row1 → P₁: [tokens, 4096]  ← 部分和
GPU 2: X₂·W_row2 → P₂: [tokens, 4096]  ← 部分和
GPU 3: X₃·W_row3 → P₃: [tokens, 4096]  ← 部分和

AllReduce →  Y = P₀ + P₁ + P₂ + P₃: [tokens, 4096]  ← 完整激活值
```

**同步后得到什么**：完整的激活张量 `[tokens, hidden_size]`，继续传入下一层的 LayerNorm。

---

#### 完整的 TP 前向数据流（以一个 Transformer 层为例）

```
输入 X: [tokens, 4096]（每卡都有完整副本）
  │
  ├──────────────────────────────── Attention 路径 ────────────────────────────────┐
  │                                                                                 │
  │ q_proj [4096→1024/卡]（列并行）  k_proj（列并行）  v_proj（列并行）            │
  │ → Q₀,K₀,V₀    → Q₁,K₁,V₁    → Q₂,K₂,V₂    → Q₃,K₃,V₃                      │
  │  （各卡持有 Q/K/V 的 1/4 heads，独立完成 Flash Attention）                     │
  │                                                                                 │
  │ o_proj（行并行）：P₀ + P₁ + P₂ + P₃                                           │
  │         ↑                                                                       │
  │    AllReduce ①  ←←←←  同步对象：o_proj 的部分和                               │
  │                                                                                 │
  └──────────────────────────────────────────────────────────────────────────────→ +残差
  │
  ├──────────────────────────────── FFN 路径 ──────────────────────────────────────┐
  │                                                                                 │
  │ gate_proj [4096→3584/卡]（列并行）  up_proj（列并行）                          │
  │ → gate₀  gate₁  gate₂  gate₃       up₀  up₁  up₂  up₃                       │
  │   （各卡独立做 SiLU(gate) ⊙ up）                                               │
  │                                                                                 │
  │ down_proj（行并行）：P₀ + P₁ + P₂ + P₃                                        │
  │         ↑                                                                       │
  │    AllReduce ②  ←←←←  同步对象：down_proj 的部分和                            │
  │                                                                                 │
  └──────────────────────────────────────────────────────────────────────────────→ +残差

每层 2 次 AllReduce，同步的都是形状 [tokens, hidden_size] 的部分和
```

---

### 1.2 反向传播同步

#### 同步对象：**输入梯度（Input Gradients）**

前向是激活值的部分和需要同步，反向是**梯度的部分和**需要同步（完全对称）。

```
前向（行并行 down_proj）：
  Y = X₀·W₀ + X₁·W₁ + X₂·W₂ + X₃·W₃  →  AllReduce 求和

反向（对输入 X 求梯度）：
  ∂L/∂X₀ = (∂L/∂Y)·W₀ᵀ   → 部分梯度
  ∂L/∂X₁ = (∂L/∂Y)·W₁ᵀ   → 部分梯度
  ...
  真正的 ∂L/∂X = AllReduce(∂L/∂X₀, ∂L/∂X₁, ...) ← 同样需要 AllReduce
```

**同步后得到什么**：完整的输入梯度 `∂L/∂X`，传给上一层继续反向传播。

**对权重的梯度（不需要同步）**：
```
∂L/∂W₀ = X₀ᵀ · (∂L/∂Y)   ← 每张卡只需要更新自己持有的那部分权重
（W₀ 在 GPU 0 上，GPU 0 自己算自己的梯度，无需和别人同步）
```

---

### 1.3 Sequence Parallelism（SP）的优化

SP 把 AllReduce 拆成 **Reduce-Scatter + AllGather**，减少中间激活的显存：

```
纯 TP（每次 AllReduce 后所有卡都有完整激活）：

  ... → [tokens, 4096] →  AllReduce  → [tokens, 4096] → LayerNorm → ...
                             ↑所有卡都持有完整的 [tokens, 4096]，冗余！

TP + SP（Reduce-Scatter + AllGather，每卡只保留 1/tp 的序列）：

         行并行输出（部分和）
  GPU 0:  P₀[tokens, 4096]
  GPU 1:  P₁[tokens, 4096]   → Reduce-Scatter →  GPU 0: Y[0:tokens/4, 4096]
  GPU 2:  P₂[tokens, 4096]                        GPU 1: Y[tokens/4:tokens/2, 4096]
  GPU 3:  P₃[tokens, 4096]                        GPU 2: Y[tokens/2:3t/4, 4096]
                                                   GPU 3: Y[3t/4:tokens, 4096]
                                                       ↑ 每卡只有 1/4 的序列长度！

  → LayerNorm（在各自的序列片段上独立计算）

  → AllGather → 恢复 [tokens, 4096]（传给下一个 Attention 需要完整序列时）
```

**同步对象对比**：

| 操作 | 同步原语 | 同步前 | 同步后 | 显存变化 |
|------|---------|--------|--------|---------|
| 纯 TP | AllReduce | 部分和 [T,H] | 完整激活 [T,H] | 不变 |
| TP+SP | Reduce-Scatter | 部分和 [T,H] | 序列切片 [T/tp,H] | 减少 tp 倍 |
| TP+SP | AllGather | 序列切片 [T/tp,H] | 完整激活 [T,H] | 增加 tp 倍 |

---

## 二、DP（数据并行）：同步梯度

### 2.1 标准 DP 的梯度同步

#### 同步对象：**权重梯度（Weight Gradients）**

每个 DP rank 处理不同的数据样本，反向传播得到的梯度不同，需要**平均**后再更新参数。

```
DP 前向（各自独立）：
  GPU 0: batch_0 → loss_0 → ∂L₀/∂W（只反映 batch_0 的信息）
  GPU 1: batch_1 → loss_1 → ∂L₁/∂W（只反映 batch_1 的信息）
  GPU 2: batch_2 → loss_2 → ∂L₂/∂W
  GPU 3: batch_3 → loss_3 → ∂L₃/∂W

AllReduce（求平均）：
  ∂L/∂W = (∂L₀/∂W + ∂L₁/∂W + ∂L₂/∂W + ∂L₃/∂W) / 4
                              ↑
              同步的对象：每个参数的梯度（与权重形状相同！）

同步后：每张 GPU 都拿到了全局平均梯度，用它来更新参数
  W ← W - lr × ∂L/∂W   ← 每张卡参数更新一致，保持所有 DP rank 参数相同
```

**梯度同步的时机**：
```
错误方式：等所有层反向传播完才同步（延迟大）
正确方式：梯度一计算好就立刻异步发起 AllReduce（与计算重叠）

PyTorch DDP 的实现：
  每个参数的梯度计算完成时，注册一个 hook：
    grad_hook = lambda grad: allreduce_async(grad)
  → 梯度计算与梯度同步并行进行
  → AllReduce 完成后，梯度已经是全局平均，直接传给 optimizer.step()
```

---

### 2.2 ZeRO（Zero Redundancy Optimizer）的同步

ZeRO 改变了同步对象，用 **Reduce-Scatter** 替代 AllReduce：

```
ZeRO-1（优化器状态分片）：

  反向传播完成后，不再做梯度 AllReduce，而是 Reduce-Scatter：

  GPU 0 梯度：[∂W₀, ∂W₁, ∂W₂, ∂W₃]（完整梯度，4段）
  GPU 1 梯度：[∂W₀, ∂W₁, ∂W₂, ∂W₃]

  Reduce-Scatter 后：
    GPU 0 只收到 ∂W₀ 段的平均梯度  → 只更新 W₀ 段的参数
    GPU 1 只收到 ∂W₁ 段的平均梯度  → 只更新 W₁ 段的参数
    GPU 2 只收到 ∂W₂ 段的平均梯度  → 只更新 W₂ 段的参数
    GPU 3 只收到 ∂W₃ 段的平均梯度  → 只更新 W₃ 段的参数

  更新完成后 AllGather 让所有 GPU 获取完整更新后的参数：
    AllGather：[W₀_new, W₁_new, W₂_new, W₃_new] → 每卡都有完整参数
```

| ZeRO 阶段 | 同步对象 | 分片内容 | 每卡显存节省 |
|-----------|---------|---------|------------|
| ZeRO-0（DDP）| 梯度（AllReduce）| 无分片 | 0 |
| ZeRO-1 | 梯度（ReduceScatter）| 优化器状态 | 4× |
| ZeRO-2 | 梯度（ReduceScatter）| 优化器状态 + 梯度 | 8× |
| ZeRO-3 | 激活/梯度/参数（全分片）| 全部 | ~dp× |

**ZeRO-3 的同步时机**：

```
前向传播中：
  每层计算前 AllGather 该层权重（从 dp 个 rank 拼回完整权重）
  该层计算后立刻丢弃权重（释放显存）

反向传播中：
  每层计算前 AllGather 该层权重（再次拼回）
  梯度计算完成后立刻 ReduceScatter（只保留自己负责的梯度分片）

→ 代价：AllGather 频率大幅增加（每层都要 AllGather 权重）
→ 收益：显存节省 dp 倍，可以训练更大模型
```

---

## 三、PP（流水线并行）：同步激活值和梯度

### 3.1 前向传播同步

#### 同步对象：**层间激活值（Inter-Stage Activations）**

PP 把模型层分配到不同 GPU（Stage）上，Stage i 的输出就是 Stage i+1 的输入：

```
模型切分（PP=4，32层模型，每 Stage 8层）：

Stage 0 (GPU 0): Layers 0~7
Stage 1 (GPU 1): Layers 8~15
Stage 2 (GPU 2): Layers 16~23
Stage 3 (GPU 3): Layers 24~31 → lm_head → loss

数据流（单个 microbatch）：

  GPU 0: x → L0...L7 → h₇[B,S,H]  ──P2P Send──→
  GPU 1:                             P2P Recv→ h₇ → L8...L15 → h₁₅  ──P2P Send──→
  GPU 2:                                                         P2P Recv→ h₁₅ → L16...L23 → h₂₃  ──→
  GPU 3:                                                                                       Recv→ h₂₃ → L24...L31 → loss
```

**同步原语**：点对点（P2P）Send/Recv，不是集合通信

**P2P 的特点**：
```
AllReduce：所有 GPU 都参与，必须等所有人就绪
P2P Send/Recv：只有相邻两张卡通信，其他卡可以继续计算

→ PP 的 P2P 通信本质上是"流水线的接力棒"
→ 不需要等待整个 DP/TP 组就绪，只需要上下游 Stage 就绪
```

**同步后得到什么**：Stage i+1 收到 `h[B,S,H]`（激活值张量），直接作为本 Stage 第一层的输入。

---

### 3.2 反向传播同步

#### 同步对象：**层间梯度（Inter-Stage Gradients）**

反向方向与前向相反：

```
反向流（1F1B 调度）：

GPU 3: loss → ∂L/∂h₂₃  ──P2P Send──→（往 GPU 2 发梯度）
GPU 2: ← P2P Recv ← ∂L/∂h₂₃ → 反向 L16...L23 → ∂L/∂h₁₅  ──Send──→ GPU 1
GPU 1: ← Recv ← ∂L/∂h₁₅ → 反向 L8...L15 → ∂L/∂h₇  ──Send──→ GPU 0
GPU 0: ← Recv ← ∂L/∂h₇  → 反向 L0...L7 → (各层权重梯度)
```

**同步后得到什么**：
- 收到梯度后，当前 Stage 做本地反向传播，计算本 Stage 所有层的权重梯度
- 然后把输入梯度 `∂L/∂h_in` 继续发给上游 Stage

---

### 3.3 Pipeline Bubble：同步的代价

```
1F1B 调度（PP=4，4个 microbatch）：

时间 →  t1   t2   t3   t4   t5   t6   t7   t8   t9   t10  t11  t12
GPU 0:  F0   F1   F2   F3   B0   B1   B2   B3
GPU 1:  [W]  F0   F1   F2   F3   B0   B1   B2   B3
GPU 2:  [W]  [W]  F0   F1   F2   F3   B0   B1   B2   B3
GPU 3:  [W]  [W]  [W]  F0   F1   F2   F3   B0   B1   B2   B3

[W] = Wait，气泡（Bubble）

气泡比例 = (pp-1)/(num_microbatches) = 3/4 = 75%！
（若 num_microbatches=16：气泡比例 = 3/16 ≈ 19%，可接受）
```

**气泡的本质**：PP 的同步等待代价——Stage 0 要等 Stage 3 完成前向才能开始反向，这段等待时间就是"气泡"。

---

## 四、EP（专家并行）：同步 Token 隐状态

### 4.1 Dispatch 阶段（前向第一个 AllToAll）

#### 同步对象：**Token 的隐状态向量（Token Hidden States）**

```
场景（EP=4，4张卡各有 64 个专家，token 选 top-2）：

路由计算完成后（每张 GPU 知道本地 token 要去哪个专家）：
  GPU 0 上有 tokens: [t₀, t₁, t₂, t₃]
    t₀ → Expert 7 (在 GPU 0)、Expert 130 (在 GPU 2)
    t₁ → Expert 55 (在 GPU 0)、Expert 200 (在 GPU 3)
    t₂ → Expert 90 (在 GPU 1)、Expert 15 (在 GPU 0)
    t₃ → Expert 110 (在 GPU 1)、Expert 250 (在 GPU 3)

AllToAll Dispatch：
  GPU 0 发送 t₂, t₃ 给 GPU 1（这些 token 需要 GPU 1 上的专家）
  GPU 0 发送 t₀    给 GPU 2（t₀ 需要 GPU 2 上的专家）
  GPU 0 发送 t₁, t₃ 给 GPU 3
  GPU 0 保留 t₀, t₁, t₂（它们在 GPU 0 上有专家）

每张 GPU 在 AllToAll 后收到：
  "所有路由到本 GPU 专家的 token"
```

**同步后得到什么**：每张 GPU 收到了**应由自己专家处理的 token 集合**，接下来在本地做 FFN 计算（完全并行，无通信）。

---

### 4.2 Combine 阶段（前向第二个 AllToAll）

#### 同步对象：**专家输出（Expert Outputs）**

```
各 GPU 专家计算完成后：
  GPU 0 计算出：token t₀ 经过 Expert 7 的输出 e₀₇、t₂ 经过 Expert 15 的输出 e₂₁₅ ...
  GPU 1 计算出：t₂ 经过 Expert 90 的输出 e₂₉₀、t₃ 经过 Expert 110 的输出 e₃₁₁₀ ...
  GPU 2 计算出：t₀ 经过 Expert 130 的输出 e₀₁₃₀ ...
  GPU 3 计算出：t₁ 经过 Expert 200 的输出 e₁₂₀₀、t₃ 经过 Expert 250 的输出 e₃₂₅₀ ...

AllToAll Combine（发回 token 原始所在的 GPU）：
  GPU 2 把 e₀₁₃₀ 发还给 GPU 0
  GPU 3 把 e₁₂₀₀, e₃₂₅₀ 发还给 GPU 0（t₁, t₃ 原本在 GPU 0）
  ...
```

**同步后得到什么**：每张 GPU 收回了本地所有 token 的**所有专家输出**，做加权求和：

```python
# 同步后的处理
output = sum(router_prob[token_i][expert_j] * expert_output[token_i][expert_j]
             for expert_j in top_k_experts[token_i])
# → 每个 token 得到 top-K 个专家输出的加权和 [hidden_size]
# → 继续传入下一层 Attention
```

---

### 4.3 DeepSeek 的 AllToAll 优化：限制节点路由

```
朴素 EP 的 AllToAll：
  每个 token 可以路由到任意节点上的专家
  → AllToAll 的通信矩阵是"稠密的"，每张卡都要和其他所有卡通信
  → 大量跨节点 InfiniBand 通信（慢）

DeepSeek-V3 限制"每 token 最多路由到 M=4 个节点"：
  Token 的 top-8 专家分布在最多 4 个节点上
  → AllToAll 的通信矩阵是"稀疏的"，大量通信在 NVLink 内完成
  → 跨节点 IB 通信量减少约 2 倍

DeepSeek-V3 Decode 的极限情况（EP320，每卡 1 个专家）：
  AllToAll → P2P（因为每个专家恰好在某张固定 GPU 上）
  Token 直接发到对应 GPU，不需要集合通信
  → 通信延迟从 AllToAll 的 ~100μs 降到 P2P 的 ~10μs
```

---

## 五、各并行方式汇总对比

### 5.1 同步对象、时机、原语对比表

| 并行类型 | 同步阶段 | 同步对象 | 形状 | 通信原语 | 同步时机 | 同步后做什么 |
|---------|---------|---------|------|---------|---------|-------------|
| **TP 前向** | 行并行结束 | o_proj/down_proj 的部分和 | [T, H] | AllReduce | 每层（×2次）| 加残差，传下层 |
| **TP+SP 前向** | 行并行结束 | 部分和 | [T, H] | ReduceScatter | 每层 | LayerNorm（切片上）|
| **TP+SP 前向** | Attention 前 | 序列切片 | [T/tp, H] | AllGather | 每层 | 完整序列做 Attention |
| **TP 反向** | 列并行梯度 | 输入梯度部分和 | [T, H] | AllReduce | 每层（×2次）| 传给上层反向 |
| **DP 反向** | 整个反向结束 | 权重梯度 | 与权重同形状 | AllReduce | 每个 step | optimizer.step() 更新参数 |
| **ZeRO-1/2 反向** | 整个反向结束 | 梯度分片 | 权重的 1/dp | ReduceScatter | 每个 step | 各卡更新自己负责的参数分片 |
| **ZeRO-1/2 更新后** | 参数更新后 | 完整参数 | 权重形状 | AllGather | 每个 step | 前向传播使用完整参数 |
| **ZeRO-3 前向** | 每层前 | 该层完整权重 | 权重形状 | AllGather | 每层 | 本层前向计算后立即释放 |
| **PP 前向** | Stage 边界 | 层间激活值 | [B, S, H] | P2P Send/Recv | 每个 stage 结束 | 下游 Stage 直接用作输入 |
| **PP 反向** | Stage 边界 | 层间梯度 | [B, S, H] | P2P Send/Recv | 每个 stage 结束 | 上游 Stage 继续反向传播 |
| **EP 前向① ** | MoE 层前 | Token 隐状态 | [T_local, H] | AllToAll | 每个 MoE 层 | 各 GPU 计算本地专家 FFN |
| **EP 前向② ** | 专家计算后 | 专家输出 | [T_local, H] | AllToAll | 每个 MoE 层 | 加权求和，传下层 |

---

### 5.2 关键区别：同步对象的本质不同

```
TP：同步的是"同一批 tokens 的计算结果"（激活值的部分和）
  → 多张卡共同完成一个大矩阵乘法，结果要加起来

DP：同步的是"不同数据样本的梯度"（统计平均）
  → 多张卡各自看了一批数据，梯度要平均以代表全局数据分布

PP：同步的是"不同层之间的数据"（流水线的传递）
  → 上游 Stage 计算好的激活值/梯度，传给下游 Stage 继续计算

EP：同步的是"被路由到错误 GPU 上的 token"（数据重分布）
  → Token 在路由器决定后需要物理移动到专家所在的 GPU
```

---

### 5.3 同步与否对结果的影响（不同步会怎样？）

```
TP 不同步（跳过 AllReduce）：
  每张卡拿到的是 Y_partial（不完整的激活值）
  → LayerNorm 归一化的是"1/4 的特征"而非全部特征
  → 数值完全错误，loss 变成 NaN

DP 不同步（跳过梯度 AllReduce）：
  每张卡用的是本地梯度（只反映本地 batch）
  → 不同卡的参数更新不一致
  → 几步之后不同 DP rank 的模型参数完全不同（分叉）
  → 相当于每张卡在训练独立的模型，DP 失去意义

PP 不同步（Stage 之间不传数据）：
  下游 Stage 收不到输入
  → 程序卡死（等待超时/死锁）

EP 不同步（AllToAll 跳过）：
  Token 没有被发到正确的专家 GPU
  → 每个专家计算了错误的 token（或没有 token 可以计算）
  → FFN 输出毫无意义
```

---

## 六、通信与计算重叠：同步的时机优化

不是所有同步都必须"停下来等"——关键是**让通信和计算时间上重叠**。

### 6.1 DP 梯度通信与反向计算重叠

```
朴素方式（先算完所有梯度，再统一 AllReduce）：
  时间线：[反向传播 100ms] → [梯度 AllReduce 30ms] → 总 130ms

优化方式（梯度一出来就立即异步 AllReduce）：
  时间线：[反向传播 100ms]
                    ↑ 每层反向完成，立即发起 AllReduce（异步）
          [反向传播 30ms 时，前 70ms 的 AllReduce 已经完成]
          总 = max(100, 100+最后几层的 AllReduce) ≈ 103ms

代价：梯度 AllReduce 分散发起，而非一次大的 AllReduce
     （NCCL bucket 机制：积累到 25MB 才发起，平衡延迟和带宽）
```

### 6.2 PP 通信与计算重叠（1F1B）

```
1F1B 中的隐式重叠：
  当 GPU 1 在做前向（计算 Layer 8~15）时
  GPU 0 可以同时在做下一个 microbatch 的前向（Layer 0~7）

  → 通信（P2P Send/Recv）和计算（下一个 microbatch）天然重叠

DualPipe 的显式重叠（DeepSeek-V3）：
  将一个 chunk 拆成 4 部分：
    [Attention 计算] [AllToAll Dispatch] [FFN/Expert 计算] [AllToAll Combine]

  双向调度：
    ← 反向 chunk 在做 Attention 时（无通信）
    → 正向 chunk 同时在做 AllToAll（纯通信）
  → 计算与通信 100% 重叠！
```

### 6.3 EP AllToAll 与 Expert 计算重叠

```
朴素：发送 token → 等待所有 token 到达 → 开始 Expert 计算
重叠：
  本地 token（不需要发送的）可以立即开始 Expert 计算
  远程 token 在传输途中，本地 token 已经在计算
  → Token 计算与 AllToAll 重叠（需要硬件支持 RDMA + CUDA Stream 并行）
```

---

## 七、从代码层面理解同步

### 7.1 TP AllReduce（vLLM 代码）

```python
# vllm/model_executor/layers/linear.py（简化）

class RowParallelLinear(nn.Module):
    """行并行线性层，输出需要 AllReduce"""

    def forward(self, input_):
        # 每张 GPU 只有部分输入（input_dim 的 1/tp）
        output_parallel = F.linear(input_, self.weight)  # 部分和 [T, H]

        # 关键：AllReduce 同步部分和
        if self.tp_size > 1:
            # 同步对象：output_parallel（每张 GPU 的部分和）
            output = tensor_model_parallel_all_reduce(output_parallel)
            # 同步后：output 是完整的激活值 [T, H]
        else:
            output = output_parallel

        return output

# tensor_model_parallel_all_reduce 内部：
def tensor_model_parallel_all_reduce(input_):
    group = get_tp_group()  # TP 组内的所有 rank
    torch.distributed.all_reduce(input_, group=group)  # In-place AllReduce
    return input_
```

### 7.2 DP 梯度 AllReduce（PyTorch DDP）

```python
# PyTorch DDP 的梯度同步（简化）

class DDP(nn.Module):
    def __init__(self, module):
        # 注册梯度 hook
        for param in module.parameters():
            param.register_post_accumulate_grad_hook(
                self._gradient_ready_hook(param)
            )

    def _gradient_ready_hook(self, param):
        def hook(grad):
            # 同步对象：param.grad（权重的梯度，形状与权重相同）
            dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=self.dp_group)
            grad.div_(self.world_size)  # 求平均
            # 同步后：grad 是全局平均梯度，optimizer.step() 会用它更新参数
        return hook
```

### 7.3 PP P2P 通信（简化）

```python
# Pipeline Stage 之间的激活值传递

# Stage i（发送方）：
def send_forward(output_tensor):
    # 同步对象：output_tensor（层间激活值 [B, S, H]）
    dist.send(output_tensor,
              dst=next_stage_rank,   # 下游 Stage 的 rank
              group=pp_group)
    # 发送后：本 Stage 继续处理下一个 microbatch（流水线并行的核心）

# Stage i+1（接收方）：
def recv_forward():
    input_tensor = torch.empty([B, S, H], dtype=torch.bfloat16, device='cuda')
    dist.recv(input_tensor,
              src=prev_stage_rank,   # 上游 Stage 的 rank
              group=pp_group)
    # 同步后：input_tensor 就是上游 Stage 的输出，直接作为本 Stage 第一层的输入
    return input_tensor
```

---

## 八、完整前向传播的同步时间线

以 LLaMA-3-8B，TP=4，PP=2，Decode（batch=16，seq=1）为例：

```
时间线（每个格子 ≈ 0.1ms）：

       t1      t2      t3      t4      t5      t6      t7
GPU 0: [L0 Attn][AR①][L0 FFN][AR②][L1 Attn][AR③][L1 FFN][AR④]...Layer 0~15
GPU 1: ← P2P Recv ←    同上 Layer 0~15    → [L16 Attn][AR][L16 FFN]...Layer 16~31
GPU 2: （与 GPU 0 并行，TP 组内同步）
GPU 3: （与 GPU 1 并行，TP 组内同步）

AR = AllReduce（TP 同步），每次 ≈ 0.05ms（NVLink）
P2P = Stage 间传递，≈ 0.1ms（IB，16 × 4096 × 2 bytes = 128KB）

关键观察：
  AR 必须在 TP 组内同步完成后才能继续（阻塞点）
  P2P 完成后 Stage 1 立即开始计算（流水线效率的关键）
```

---

## 九、总结：同步对象的本质分类

```
按同步的"数据语义"分类：

1. 分散计算结果的聚合（加法）：
   → TP AllReduce：部分矩阵乘积求和 → 完整激活值
   → DP AllReduce：各 batch 梯度求和 → 全局平均梯度

2. 数据的重新分布（搬运）：
   → PP P2P：激活值从上游 Stage 搬到下游 Stage
   → EP AllToAll：Token 从原始 GPU 搬到 Expert 所在 GPU，再搬回来

3. 分散数据的选择性汇聚（部分加法）：
   → EP Combine AllToAll：Expert 输出搬回原始 GPU，加权求和

同步后必做的事：
   TP → 继续下一层（激活值已完整）
   DP → optimizer.step() 更新参数（梯度已平均）
   PP → 下游 Stage 开始本地计算（输入已到达）
   EP Dispatch → 各 GPU 做 Expert FFN（token 已到位）
   EP Combine → 加权求和（expert 输出已汇聚）
```

---

*参考资料：*
- *[Megatron-LM 行/列并行：Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053)*
- *[ZeRO 优化器：Rajbhandari et al. 2020](https://arxiv.org/abs/1910.02054)*
- *[Sequence Parallelism：Korthikanti et al. 2022](https://arxiv.org/abs/2205.05198)*
- *[1F1B 流水线调度：Narayanan et al. 2021](https://arxiv.org/abs/2104.04473)*
- *[DualPipe：DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)*
- *[vLLM RowParallelLinear 源码](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/linear.py)*
*更新：2026-03*
