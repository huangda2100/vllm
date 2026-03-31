# vLLM 中的门控网络与 MoE（Mixture of Experts）

## 目录
1. [MoE 的历史与演进](#1-moe-的历史与演进)
2. [MoE 基础原理详解](#2-moe-基础原理详解)
3. [门控网络数学推导](#3-门控网络数学推导)
4. [vLLM MoE 架构概览](#4-vllm-moe-架构概览)
5. [标准 Softmax 门控（Mixtral 风格）](#5-标准-softmax-门控mixtral-风格)
6. [分组 TopK 门控（DeepSeek 风格）](#6-分组-topk-门控deepseek-风格)
7. [带纠正偏置的门控（noaux_tc）](#7-带纠正偏置的门控noaux_tc)
8. [共享专家（Shared Experts）](#8-共享专家shared-experts)
9. [FusedMoE 前向传播流程](#9-fusedmoe-前向传播流程)
10. [CUDA/Triton 内核实现](#10-cudatriton-内核实现)
11. [专家并行（Expert Parallelism）](#11-专家并行expert-parallelism)
12. [专家并行负载均衡（EPLB）](#12-专家并行负载均衡eplb)
13. [量化支持](#13-量化支持)
14. [MoE vs Dense 对比](#14-moe-vs-dense-对比)

---

## 1. MoE 的历史与演进

### 1.1 根源问题：深度学习的 Scaling 困境

在讲 MoE 之前，先理解它要解决的根本问题。

深度学习有一条朴素的经验法则：**模型越大，性能越好**。GPT-3 比 GPT-2 好，是因为参数从 1.5B 增长到了 175B。但"模型更大"有一个致命代价：

```
参数量 × 2  →  计算量（FLOPs） × 2  →  训练/推理成本 × 2
```

这是 Dense（稠密）模型的宿命——每个输入 token 都要经过模型的全部参数。参数量和计算量是**线性绑定**的，想要更强的模型，就必须支付更多的算力。

那么有没有一种方法，可以让模型拥有庞大的参数量（高容量），但每次推理只用其中一小部分（低计算量）？这就是 MoE 的核心思想：**参数量和计算量的解耦**。

```
Dense 模型:   参数量 = 计算量（绑定）
MoE 模型:    参数量 >> 计算量（解耦）

具体来说:
  Dense 8B:   8B 参数, 每个 token 激活 8B 参数
  MoE 47B:   47B 参数, 每个 token 只激活 ~13B 参数
```

### 1.2 MoE 的起源（1991年）：分而治之的直觉

MoE 的原始论文来自 1991 年 Jacobs、Jordan、Nowlan 和 **Hinton** 的经典论文：
> *"Adaptive Mixtures of Local Experts"* (Neural Computation, 1991)

他们的出发点很朴素——一个来自日常生活的类比：

> **一个人不可能精通所有事情，但一个团队可以。** 医生看病、律师打官司、厨师做饭——每个"专家"只擅长自己的领域。当面对一个具体问题时，我们不需要所有人都参与，只需要**找到最合适的专家**来处理。

将这个直觉转化为数学模型：

```
传统网络（一个"全才"）:
  输入 x  ──→  [ 单个大网络 f(x) ]  ──→  输出 y

  问题: 网络必须用同一组参数处理所有类型的输入
        → 参数相互"争抢"，难以同时处理差异巨大的模式

MoE 网络（一个"专家团队"）:
  输入 x  ──→  [ 门控网络 g(x) ]  ──→  选择哪些专家
                    ↓
              ┌─────┼─────┐
              ▼     ▼     ▼
          [Expert₁][Expert₂][Expert₃]   ← 每个专家独立学习
              ↓     ↓     ↓
              └─────┼─────┘
                    ▼
          加权组合:  y = Σ g_i(x) × Expert_i(x)
```

关键组件有两个：
- **专家网络（Experts）**：多个独立的子网络，每个可以专门学习输入空间的某个区域
- **门控网络（Gating Network）**：一个"调度员"，根据当前输入决定该问哪个专家

1991 年的原始 MoE 有几个重要特征：
- 所有专家都参与计算（**Dense MoE**），只是权重不同——热门专家权重高，冷门专家权重低
- 门控输出是 Softmax 概率分布，所有专家权重加起来为 1
- 专家都很小（几十个神经元），整个模型也很小

这时的 MoE 更像一种"soft 分工"，还没有实现真正的计算节省。

### 1.3 沉寂与复兴（1991-2017）

1991 年之后，MoE 的思想在学术界沉寂了很长时间。原因是：

1. **当时的模型本身就小**（几千到几万参数），MoE 的"扩大参数量"优势体现不出来
2. **计算硬件有限**，无法支撑多个专家的并行计算
3. **训练不稳定**：门控网络倾向于只选择少数几个专家（"赢者通吃"），其他专家永远得不到训练信号——这就是后来被称为 **"专家崩溃"（Expert Collapse）** 的问题

期间也有一些理论进展，如 Jordan & Jacobs (1994) 提出了**层级 MoE（Hierarchical MoE）**，将门控网络组织成树状结构，但实用性有限。

### 1.4 关键转折：Shazeer et al. 2017 —— Sparsely-Gated MoE

2017 年，Google 的 Noam Shazeer（Transformer 论文的共同作者之一）发表了一篇里程碑论文：

> *"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer"*

这篇论文做了一个关键创新：**稀疏门控（Sparse Gating）**。

回忆 1991 年的原始 MoE：所有专家都参与计算，只是权重不同。Shazeer 说：**为什么不只激活少数几个专家？** 大部分专家的权重本来就接近 0，直接跳过它们，把计算量省下来。

```
1991 Dense MoE（所有专家参与）:
  y = 0.6×Expert₁(x) + 0.3×Expert₂(x) + 0.05×Expert₃(x) + 0.03×Expert₄(x) + 0.02×Expert₅(x)
  FLOPs = 5 × 单专家FLOPs（全部算一遍）

2017 Sparse MoE（只激活 Top-2）:
  y = 0.67×Expert₁(x) + 0.33×Expert₂(x)    ← 只选分数最高的2个
  FLOPs = 2 × 单专家FLOPs（省了60%计算）
```

具体做法是在门控网络后面加一个 **Top-K + 噪声** 机制：

```
步骤1: 计算每个专家的原始得分
  scores = x @ W_gate            (hidden_state × gate_weight)

步骤2: 加入噪声（鼓励探索不同专家）
  noisy_scores = scores + noise  (noise ~ N(0, softplus(x @ W_noise)))

步骤3: 只保留 Top-K 个最高分
  sparse_scores = KeepTopK(softmax(noisy_scores), k=2)

步骤4: 只激活被选中的专家
  y = Σ sparse_scores_i × Expert_i(x)   (非Top-K的权重为0,不计算)
```

这篇论文的成果震撼了当时的 NLP 界：
- 构建了一个 **137B 参数**的 MoE 模型（当时最大）
- 每次推理只激活约 **2B 参数**
- 在机器翻译任务上用 **1/10 的计算量** 达到了 Dense 模型的性能

但也暴露了 MoE 的核心挑战：

**挑战1 — 专家崩溃（Expert Collapse）**：门控网络发现某几个专家效果好后，就一直选它们，其他专家完全不被训练。

**挑战2 — 负载不均衡（Load Imbalance）**：热门专家处理太多 token，变成计算瓶颈；冷门专家空闲浪费。

Shazeer 的解决方案是引入**辅助损失（Auxiliary Loss）**：

```
L_total = L_task + α × L_balance

L_balance = Σᵢ (fᵢ × pᵢ)

  fᵢ = 被路由到专家 i 的 token 占比（实际负载）
  pᵢ = 门控网络分配给专家 i 的平均概率（倾向性）
```

当某个专家被过度使用时（fᵢ 大），L_balance 增大，梯度推动门控网络减少选择该专家。这是一种"税收"——热门专家要交税，以鼓励均衡分配。

### 1.5 Switch Transformer（2021）：Top-1 简化

Google Brain 的 Fedus 等人提出了 **Switch Transformer**：

> *"Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"*

核心改进：**把 Top-K 简化为 Top-1**——每个 token 只送给一个专家。

```
Shazeer 2017:  Top-2, 每个 token 激活 2 个专家
Switch 2021:   Top-1, 每个 token 只激活 1 个专家

好处:
  1. 计算量减半（只算 1 个专家 vs 2 个）
  2. 通信量减半（专家并行时，token 只需发给 1 个 GPU）
  3. 路由逻辑更简单（不需要多专家权重归一化）

代价:
  每个 token 只有一个"视角"，多样性略降
```

Switch Transformer 还引入了 **Expert Capacity** 概念：每个专家每个 batch 最多处理 C 个 token，超过就丢弃。这是一种粗暴但有效的负载均衡方式。

```
capacity = (tokens_in_batch / num_experts) × capacity_factor

  capacity_factor = 1.0: 每个专家平均分担
  capacity_factor = 1.25: 允许 25% 的不均衡冗余

  如果某专家已经满了，新 token 直接跳过 MoE 层（走 residual）
```

论文展示了 **1.6 万亿参数**的模型训练，但因为 Top-1 路由，实际计算量只相当于 Dense T5-Base。

### 1.6 GShard（2021）与 ST-MoE（2022）：工程化与稳定性

**GShard**（Google, 2021）解决了 MoE 模型的**工程部署问题**——如何把数千个专家分布到数千个 TPU 上。核心贡献是提出了 Expert Parallelism（专家并行）的系统化方案，以及 All-to-All 通信原语的高效实现。

**ST-MoE**（Google, 2022）则解决了 MoE 的**训练稳定性问题**。MoE 训练时经常出现 loss spike（损失突然飙升），ST-MoE 发现罪魁祸首是 router 的梯度不稳定，提出了 router z-loss：

```
L_z = (1/B) × Σ (log Σ exp(router_logits))²

直觉：惩罚 router logits 的绝对值过大（logits 太大 → softmax 趋向 one-hot → 梯度消失/爆炸）
```

### 1.7 开源 MoE 时代（2024-至今）

2024 年是 MoE 的"开花"之年，多个高质量开源 MoE 模型密集发布：

**Mixtral-8x7B（Mistral AI, 2024.01）**
- 第一个让开源社区真正见识到 MoE 威力的模型
- 8 个专家, Top-2, 总参数 47B, 激活 13B
- 性能匹敌 LLaMA-2-70B（Dense），但推理速度快 6 倍
- 设计简洁：标准 Softmax + TopK，无花哨技巧
- 对社区的意义：证明 MoE 不再是 Google 专属的实验品，任何人都能用

**DeepSeek-V2（深度求索, 2024.06）**
- 236B 参数, 激活 21B, 160 个路由专家 + 2 个共享专家
- 三大创新：
  - **分组 TopK 门控**：专家分组 → 先选组 → 再在组内选专家（避免所有 token 都涌向同一组专家）
  - **共享专家（Shared Experts）**：部分专家对所有 token 始终激活（保证基础能力）
  - **noaux_tc（无辅助损失负载均衡）**：用可学习偏置替代辅助损失，训练更稳定

**DeepSeek-V3（深度求索, 2024.12）**
- 671B 参数, 激活 37B, 256 个路由专家 + 1 个共享专家
- 训练仅花费 557 万美元（对比 LLaMA-3-405B 据估计花费数千万美元）
- 性能匹敌 GPT-4o，成本低一个数量级
- 是"MoE 降本增效"理念的最佳证明

**Qwen2-MoE（阿里巴巴, 2024）**
- 57B 参数, 激活 14B, 64 个路由专家 + 8 个共享专家
- 引入 **Sigmoid 共享门控**：对共享专家的输出再乘一个 sigmoid 门（控制共享专家的贡献度）

### 1.8 MoE 演进的核心脉络

把这 30 多年的历史画成一条线：

```
1991 Jacobs et al.        所有专家都参与，只是权重不同（Dense MoE）
  │                         → 计算量没有省下来
  ▼
2017 Shazeer et al.       Top-K 稀疏门控 + 噪声探索 + 辅助损失
  │                         → 参数和计算量解耦，MoE 真正可用
  ▼
2021 Switch Transformer   Top-1 极致稀疏 + Expert Capacity
  │                         → 路由更简单，扩展到万亿参数
  ▼
2021 GShard               Expert Parallelism + All-to-All
  │                         → 工程化：数千专家分布到数千加速器
  ▼
2022 ST-MoE               Router z-loss + 训练稳定性
  │                         → 解决 loss spike，MoE 训练变可靠
  ▼
2024 Mixtral-8x7B         开源标杆：简洁的 Top-2 Softmax
  │                         → 证明 MoE 不是大厂专利
  ▼
2024 DeepSeek-V2/V3       分组 TopK + 共享专家 + noaux_tc
  │                         → 门控机制精细化，去掉辅助损失
  ▼
2024 Qwen2-MoE            Sigmoid 共享门控
                            → 对共享专家的贡献做动态控制
```

每一步演进都在回答同一个问题的不同层面：
- **1991**："能不能让不同的子网络处理不同类型的输入？" → **可以，但没省计算**
- **2017**："能不能只激活部分专家？" → **可以，Top-K 稀疏门控**
- **2021**："能不能更稀疏？" → **可以，Top-1 就够了**
- **2022**："训练能不能不崩？" → **可以，z-loss 稳定 router**
- **2024**："门控能不能更聪明？" → **可以，分组选择 + 去辅助损失 + 共享专家**

---

## 2. MoE 基础原理详解

### 2.1 直觉：为什么 MoE 有效？

用一个类比来理解。假设你经营一家医院：

```
Dense 模型（全科医生模式）：
  你只有 1 个医生，他必须同时精通内科、外科、眼科、牙科...
  → 什么都会一点，但每个领域都不够深入
  → 想让他更强？增加他的"脑容量"（参数量），但每次问诊他都要动用全部知识（计算量线性增长）

MoE 模型（专科医院模式）：
  你有 8 个医生，各自专攻不同方向
  每个病人来了，前台（门控网络）先评估病情，然后分配给 2 个最相关的医生
  → 医院总知识量 = 8个医生的知识之和（参数量大）
  → 每次问诊只需 2 个医生的时间（计算量小）
  → 每个医生可以在自己的领域更深入（专门化）
```

数学上的好处：

```
Dense FFN:  output = FFN(x)          参数量 = H × I × 2,  每token计算量 = H × I × 2
MoE FFN:    output = Σ gate_i × FFN_i(x)   参数量 = E × H × I × 2,  每token计算量 = K × H × I × 2

参数膨胀比 = E（专家数量）
计算膨胀比 = K/1 = K（只激活K个）

当 E=8, K=2 时：
  参数量是 Dense 的 8 倍
  计算量是 Dense 的 2 倍（实际开销比 Dense 多一点，但远小于 8 倍）
```

### 2.2 为什么用"门控网络"而不是随机分配？

一个自然的问题：既然只需要选 K 个专家，为什么不随机选？或者轮流选？

```
随机分配:  每个 token 随机选 2 个专家
  → 所有专家被迫学习所有类型的输入（没有专门化）
  → 等价于一个小了 E/K 倍的 Dense 模型（更差）

轮流分配:  token 0 选专家 {0,1}, token 1 选专家 {2,3}, ...
  → 同样无法专门化
  → 且不同类型的 token 可能得到差异很大的处理质量

门控网络分配:  根据输入 x 的内容动态选择
  → 语法类 token 倾向于选同一批专家 → 这些专家逐渐成为"语法专家"
  → 事实类 token 倾向于选另一批专家 → 这些专家逐渐成为"知识专家"
  → 自发形成分工（虽然分工方式可能不像人类直觉那样清晰）
```

门控网络的本质是一个**可学习的路由策略**。它通过训练学会"什么样的输入应该交给哪个专家处理效果最好"。

### 2.3 实际效果对比

| 模型 | 参数量 | 激活参数 | 专家数 | Top-K | 性能级别 |
|------|--------|---------|-------|-------|---------|
| LLaMA-3-8B（Dense） | 8B | 8B | - | - | 基准 |
| Mixtral-8x7B | ~47B | ~13B | 8 | 2 | ≈ LLaMA-2-70B |
| DeepSeek-V2 | ~236B | ~21B | 160 | 6 | 优于 LLaMA-3-70B |
| DeepSeek-V3 | ~671B | ~37B | 256 | 8 | ≈ GPT-4o |
| Qwen2-57B-A14B | 57B | 14B | 64 | 8 | ≈ LLaMA-3-70B |

核心观察：MoE 用**远少于 Dense** 的激活参数，达到了**需要 Dense 多倍参数**才能达到的性能。

### 2.4 MoE 层在 Transformer 中的位置

MoE **不是** 替换整个 Transformer block，而是只替换其中的 FFN 层：

```
标准 Transformer Block:          MoE Transformer Block:
  ┌──────────────┐                 ┌──────────────┐
  │  Attention   │                 │  Attention   │    ← 完全不变
  └──────┬───────┘                 └──────┬───────┘
         │                                │
  ┌──────▼───────┐                 ┌──────▼───────┐
  │   Dense FFN  │     替换为→     │   MoE Layer  │    ← 这里变了
  │  (1个大FFN)  │                 │ (E个小FFN     │
  └──────┬───────┘                 │  + 门控网络)  │
         │                         └──────┬───────┘
         ▼                                ▼
     下一层                            下一层
```

为什么只替换 FFN？
- Attention 本身已经是一种"稀疏"机制（通过注意力权重动态聚焦）
- FFN 是 Transformer 中参数最密集的部分（通常占总参数的 2/3）
- FFN 是逐 token 独立计算的，天然适合按 token 路由

有些模型（如 Mixtral）的每一层都用 MoE，有些则交替使用 Dense FFN 和 MoE（如 DeepSeek-V2 只在部分层用 MoE）。

### 2.5 MoE 层内部结构

```
输入: x ∈ R^{B×H}   (B=批次token数, H=hidden_size)
         │
         ▼
┌─────────────────┐
│  Gate Network   │  Linear(H, E) → router_logits ∈ R^{B×E}
│  (门控网络)      │  （只有一个矩阵乘法，非常轻量）
└────────┬────────┘
         │ Softmax → Top-K 选择
         ▼
    topk_ids    ∈ Z^{B×K}   (每个token选中的K个专家ID)
    topk_weights ∈ R^{B×K}  (对应的门控权重)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Experts (E个FFN，每个结构相同但参数独立)                  │
│                                                         │
│  Expert_i: Gate(x) → SiLU → Up(x) → Down(x)            │
│  （SwiGLU结构，和 Dense 的 FFN 一样，只是每个更小或独立）   │
└────────┬────────────────────────────────────────────────┘
         │ 加权求和
         ▼
输出: y = Σ_{i∈TopK(x)} w_i × Expert_i(x)   ∈ R^{B×H}
```

门控网络本��极其简单——就是一个没有 bias 的线性层 `nn.Linear(hidden_size, num_experts, bias=False)`。对于 hidden_size=4096, num_experts=8 的情况，门控网络只有 4096×8 = 32768 个参数，相比整个 MoE 层的数十亿参数微不足道。但正是这个小小的线性层，决定了每个 token 的路由走向。

---

## 3. 门控网络数学推导

### 3.1 标准 Softmax TopK 门控

```
设: x ∈ R^H          (单个 token 的 hidden state)
    W_g ∈ R^{H×E}    (门控网络权重，gate linear)

步骤1 - 计算 router logits:
    l = x W_g  ∈ R^E     (每个专家的"得分")

步骤2 - Softmax:
    p_i = exp(l_i) / Σ_j exp(l_j)   (概率分布)

步骤3 - Top-K 选择:
    K = {k | p_k ∈ top-k(p)}         (选K个最高概率的专家)

步骤4 - 重归一化（renormalize）:
    w_k = p_k / Σ_{j∈K} p_j          (使选中的K个权重和为1)

步骤5 - 专家计算:
    y = Σ_{k∈K} w_k × FFN_k(x)
```

**为何 renormalize？**
- 不 renormalize：top-k 权重之和 < 1，输出量级随 K 变化
- renormalize：无论 K 多大，输出量级稳定，利于训练
- Mixtral-8x7B 使用 renormalize=True

### 3.2 分组 TopK 门控（DeepSeek-V2/V3）

DeepSeek-V2 有 160 个路由专家，分成 4 组（每组 40 个）。门控采用**两级选择**：

```
步骤1 - 评分:
    scores = softmax(l)   ∈ R^{160}

步骤2 - 组间打分:
    group_scores[g] = sum(top2(scores[g×40 : (g+1)×40]))  ∈ R^4
    （每组取top-2分数之和作为该组的代表分数）

步骤3 - 选组:
    selected_groups = argtop(group_scores, k=topk_group)  ∈ Z^{topk_group}
    （从4个组里选topk_group个）

步骤4 - 组内选专家:
    # 将未选中组的分数设为 -∞
    masked_scores = scores.masked_fill(not_in_selected_groups, -∞)
    topk_ids = argtop(masked_scores, k=top_k)  ∈ Z^6

步骤5 - 权重:
    topk_weights = original_scores[topk_ids]   # 用原始分数（不含偏置）
    topk_weights = topk_weights / sum(topk_weights)   # renormalize
```

**设计意图**：
- 避免所有专家集中在同一个组（强制跨组分散）
- 减少专家崩溃（expert collapse）问题
- 不同组可能特化为不同语言/领域

### 3.3 noaux_tc 纠正偏置门控

传统负载均衡用**辅助损失（auxiliary loss）**：
```
L_aux = α × Σ_i f_i × p_i
（f_i = token选择第i个专家的频率，p_i = 专家的平均路由概率）
```
但 aux_loss 会污染主任务梯度，影响模型质量。

DeepSeek-V2 的 `noaux_tc` 方法用**可学习偏置**替代 aux_loss：

```
scores_for_routing = softmax(l) + e_score_correction_bias
topk_ids = argtop(scores_for_routing, k=top_k)   # 偏置影响选择

topk_weights = softmax(l)[topk_ids]    # 权重用原始分数（不含偏置！）
```

**关键区别**：偏置只影响专家**选择**，不影响专家**权重**。偏置通过梯度学习自动调整，使各专家负载均衡，无需手动设置 aux_loss 系数。

---

## 4. vLLM MoE 架构概览

```
vllm/model_executor/layers/fused_moe/
├── layer.py              # FusedMoE 主类（~2400行）
├── fused_moe.py          # 路由函数 + 专家计算（~2100行）
├── shared_fused_moe.py   # SharedFusedMoE（含共享专家）
├── config.py             # FusedMoEQuantConfig
└── topk_softmax_triton.py  # Triton TopK Softmax 内核

csrc/moe/
├── topk_softmax_kernels.cu   # CUDA TopK Softmax 内核
├── grouped_topk_kernels.cu   # CUDA 分组 TopK 内核
└── moe_align_block_size_kernels.cu   # token-expert 分配对齐

vllm/model_executor/models/
├── mixtral.py            # Mixtral-8x7B（标准MoE）
├── deepseek_v2.py        # DeepSeek-V2（分组TopK + 共享专家）
├── qwen2_moe.py          # Qwen2-MoE（共享专家 + sigmoid门控）
└── ...
```

### 核心类层次

```
FusedMoE (layer.py)
  ├── select_experts()     ← 门控路由
  ├── forward_cuda()       ← GPU前向
  └── fused_experts()      ← 专家计算内核分发

SharedFusedMoE (shared_fused_moe.py)
  └── FusedMoE
      └── _shared_experts  ← 共享专家 (DeepSeek/Qwen2)
```

---

## 5. 标准 Softmax 门控（Mixtral 风格）

### 5.1 Mixtral 模型结构

```python
# vllm/model_executor/models/mixtral.py

class MixtralMoE(nn.Module):
    def __init__(self, num_experts=8, top_k=2, hidden_size=4096,
                 intermediate_size=14336, ...):

        # 门控网络：轻量 Linear，不带 bias
        self.gate = ReplicatedLinear(hidden_size, num_experts, bias=False)

        # 专家组：FusedMoE 统一管理所有专家权重
        self.experts = FusedMoE(
            num_experts=num_experts,      # 8
            top_k=top_k,                  # 2
            hidden_size=hidden_size,      # 4096
            intermediate_size=intermediate_size,  # 14336
            reduce_results=True,          # 最后做 all_reduce
            renormalize=True,             # 重归一化 top-k 权重
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (num_tokens, 4096)
        router_logits, _ = self.gate(hidden_states)  # → (num_tokens, 8)
        return self.experts(hidden_states, router_logits)
```

### 5.2 专家权重存储格式

```python
# FusedMoE 中，所有专家权重合并为大矩阵（便于批量GEMM）：

layer.w13_weight:  # shape: (num_local_experts, hidden_size, 2 × intermediate // tp)
                   # w1（gate proj）和 w3（up proj）打包在一起
layer.w2_weight:   # shape: (num_local_experts, intermediate // tp, hidden_size)
                   # w2（down proj）

# 实际数值示例（Mixtral-8x7B，TP=1）：
w13_weight: (8, 4096, 28672)   # 8个专家，每个 4096→14336×2
w2_weight:  (8, 14336, 4096)   # 8个专家，每个 14336→4096
```

### 5.3 单个专家的 FFN 结构（SwiGLU）

```
x ∈ R^H
  │
  ├── Gate proj: x @ W1  → gate ∈ R^I
  └── Up proj:   x @ W3  → up   ∈ R^I
        │
        ▼
  SiLU(gate) × up        → hidden ∈ R^I   （SwiGLU 激活）
        │
        ▼
  Down proj: hidden @ W2  → output ∈ R^H
```

---

## 6. 分组 TopK 门控（DeepSeek 风格）

### 6.1 DeepSeek-V2 MoE 结构

```python
# vllm/model_executor/models/deepseek_v2.py

class DeepseekV2MoE(nn.Module):
    def __init__(self, config, ...):
        # 门控网络（路由到 160 个路由专家）
        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.n_routed_experts,   # 160
            bias=False,
        )

        # noaux_tc 纠正偏置（可选）
        if config.topk_method == "noaux_tc":
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(config.n_routed_experts)  # (160,)
            )

        # 共享专家（所有 token 都通过）
        self.shared_experts = DeepseekV2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size * config.n_shared_experts,
        )

        # 路由专家（通过 SharedFusedMoE 包装）
        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            num_experts=config.n_routed_experts,          # 160
            top_k=config.num_experts_per_tok,              # 6
            use_grouped_topk=True,                         # ← 启用分组TopK
            num_expert_group=config.n_group,               # 4
            topk_group=config.topk_group,                  # 2
            scoring_func=config.scoring_func,              # "softmax"
            renormalize=config.norm_topk_prob,             # True
            e_score_correction_bias=self.gate.e_score_correction_bias,
        )

    def forward(self, hidden_states):
        router_logits, _ = self.gate(hidden_states)   # (N, 160)
        shared_out, routed_out = self.experts(hidden_states, router_logits)

        # 合并共享专家 + 路由专家输出
        final = routed_out
        if shared_out is not None:
            final = final + shared_out

        # TP all_reduce
        if self.tp_size > 1:
            final = tensor_model_parallel_all_reduce(final)
        return final
```

### 6.2 grouped_topk() 实现

```python
# vllm/model_executor/layers/fused_moe/fused_moe.py

def grouped_topk(
    hidden_states, gating_output,
    topk: int,                          # 最终选 6 个专家
    renormalize: bool,
    num_expert_group: int = 4,          # 4 个专家组
    topk_group: int = 2,                # 选 2 个最好的组
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias = None,
):
    num_token = gating_output.shape[0]

    # 步骤1: 计算专家得分
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output, dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.sigmoid()

    # 步骤2: 处理纠正偏置（noaux_tc）
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0)
        # 组内取 top2 之和作为组得分
        group_scores = (
            scores.view(num_token, num_expert_group, -1)
            .topk(2, dim=-1)[0].sum(dim=-1)        # (N, 4)
        )
    else:
        group_scores = (
            scores.view(num_token, num_expert_group, -1)
            .max(dim=-1).values                     # (N, 4)
        )

    # 步骤3: 选 topk_group 个最好的组
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1)[1]  # (N, 2)

    # 步骤4: 创建组掩码，把未选中组的分数置为 -inf
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)            # (N, 4), 选中组为1
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand_as(scores.view(num_token, num_expert_group, -1))
        .reshape(num_token, -1)                     # (N, 160), 选中专家为1
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

    # 步骤5: 在选中的组内选 topk 个专家
    topk_ids = torch.topk(tmp_scores, k=topk, dim=-1)[1]   # (N, 6)

    # 步骤6: 权重使用原始分数（不含偏置）
    if e_score_correction_bias is not None:
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights = scores.gather(1, topk_ids)

    # 步骤7: renormalize + scale
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)
```

---

## 7. 带纠正偏置的门控（noaux_tc）

### 7.1 fused_topk_bias()

```python
# vllm/model_executor/layers/fused_moe/fused_moe.py

def fused_topk_bias(
    hidden_states,
    gating_output,                              # router logits
    e_score_correction_bias: torch.Tensor,      # (num_experts,)，可学习
    topk: int,
    renormalize: bool,
):
    n_routed_experts = gating_output.shape[-1]

    # 得分
    scores = gating_output.softmax(dim=-1)                    # (N, E)

    # 用偏置后的得分选择专家
    scores_for_choice = (
        scores.view(-1, n_routed_experts) + e_score_correction_bias.unsqueeze(0)
    )
    topk_indices = torch.topk(scores_for_choice, k=topk, sorted=True)[1]

    # 权重用原始 scores（不含偏置）
    topk_weights = scores.gather(1, topk_indices)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_indices.to(torch.int32)
```

### 7.2 偏置学习机制

`e_score_correction_bias` 在训练时通过梯度下降学习：
- 若专家 i 被过度选择（负载高），其偏置逐渐变小（降低被选概率）
- 若专家 i 很少被选择（负载低），其偏置逐渐变大（提高被选概率）
- 偏置与主任务梯度**分离**，不影响权重计算，无需手动调 aux_loss 系数

---

## 8. 共享专家（Shared Experts）

### 8.1 概念

DeepSeek-V2/V3 和 Qwen2-MoE 引入**共享专家**：
- 所有 token 都通过共享专家（类似 Dense FFN）
- 路由专家只处理部分 token（稀疏）
- 最终输出 = 共享专家输出 + 加权路由专家输出

```
         hidden_states
         /            \
        /              \
   shared_expert     gate → topk → routed_experts
       ↓                              ↓
  shared_output              routed_output
        \                           /
         \                         /
          └───── + ────────────────┘
                ↓
           final_output
```

### 8.2 SharedFusedMoE 实现

```python
# vllm/model_executor/layers/fused_moe/shared_fused_moe.py

class SharedFusedMoE(FusedMoE):
    def forward(self, hidden_states, router_logits):
        # 共享专家（所有 token 都算）
        if self._shared_experts is not None:
            shared_out = self._shared_experts(hidden_states)
        else:
            shared_out = None

        # 路由专家（稀疏，每 token 只算 top-k 个）
        routed_out = super().forward(hidden_states, router_logits)

        return shared_out, routed_out  # 调用方负责相加
```

### 8.3 Qwen2-MoE 的 sigmoid 共享门控

Qwen2-MoE 对共享专家额外加一个 sigmoid 门：

```python
# vllm/model_executor/models/qwen2_moe.py

# sigmoid gate 控制共享专家的贡献
shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
gate_value = torch.sigmoid(shared_expert_gate(hidden_states))  # (N, 1)

shared_output = shared_expert(hidden_states) * gate_value   # 按 sigmoid 缩放
```

---

## 9. FusedMoE 前向传播流程

### 9.1 完整调用链

```
FusedMoE.forward()
    │
    ▼
FusedMoE.forward_cuda()                    ← GPU 路径
    │
    ├── [步骤1] select_experts()
    │       │
    │       ├── grouped_topk()             ← DeepSeek/Qwen2
    │       ├── fused_topk_bias()          ← noaux_tc
    │       ├── fused_topk()              ← Mixtral（标准）
    │       └── custom_routing_function() ← 自定义
    │
    ├── [EPLB] eplb_map_to_physical_and_record()  ← 逻辑→物理专家ID映射
    │
    └── [步骤2] ��核选择 & 专家计算
            │
            ├── rocm_aiter_fused_experts()      ← ROCm/AMD
            ├── flashinfer_cutlass_moe()         ← FlashInfer Cutlass
            ├── deep_gemm_moe_fp8()              ← FP8 DeepGemm（最高性能）
            ├── cutlass_block_scaled_grouped_gemm ← Cutlass FP8 块量化
            └── dispatch_fused_experts_func()    ← 通用 Triton fallback
```

### 9.2 fused_experts() 内核核心逻辑

```
输入:
  hidden_states: (num_tokens, H)
  w1: (num_local_experts, H, 2I)    # gate + up 融合
  w2: (num_local_experts, I, H)
  topk_weights: (num_tokens, K)
  topk_ids: (num_tokens, K)

步骤1 - Token 到专家的分配（moe_align_block_size）:
  对每个专家 e，收集所有被路由到 e 的 token
  → expert_tokens[e]: 该专家需处理的 token 列表

步骤2 - 批量 GEMM（w1）:
  每个专家 e 对其 token 做: h1 = tokens @ w1[e]
  → h1: (tokens_for_expert, 2I)

步骤3 - SwiGLU 激活:
  gate, up = h1.chunk(2, dim=-1)   # 各 (tokens_for_expert, I)
  h2 = silu(gate) * up              # (tokens_for_expert, I)

步骤4 - 批量 GEMM（w2）:
  output = h2 @ w2[e]               # (tokens_for_expert, H)

步骤5 - 加权聚合（scatter + weighted sum）:
  对每个 token t，将其 top-k 个专家输出加权求和：
  y[t] = Σ_{k=1}^{K} topk_weights[t,k] × expert_output[t, topk_ids[t,k]]
```

### 9.3 权重应用时机控制

```python
# apply_router_weight_on_input=True（某些模型优化）
# 将权重乘在输入而非输出，可减少一次 GEMM 的数据量

if apply_router_weight_on_input:
    # 在 w1 GEMM 前乘权重
    tokens = tokens * topk_weights.unsqueeze(-1)
    output = tokens @ w1 → silu → @ w2
else:
    # 在输出后乘权重（默认）
    output = (tokens @ w1 → silu → @ w2) * topk_weights.unsqueeze(-1)
```

---

## 10. CUDA/Triton 内核实现

### 10.1 TopK Softmax CUDA 内核

**文件**：`csrc/moe/topk_softmax_kernels.cu`

```
每个 CUDA block 处理一行（一个 token），template 参数编译期固化专家数：

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA>
__global__ void topkGatingSoftmax(router_logits, output, topk_indices, k, ...):

  [Phase 1] BlockReduce 找最大值 max_val
  [Phase 2] 计算 exp(val - max_val)
  [Phase 3] BlockReduce 求和 sum_val
  [Phase 4] softmax = exp_val / sum_val，写到 output
  [Phase 5] Top-K: 循环 k 次，每次找剩余最大值

  [Optional] Renormalize: top-k 权重之和 → 1
```

**关键优化**：
- 专家数编译期固化（template 参数），向量化读取（VPT=向量宽度）
- 单 kernel 完成 softmax + top-k，避免额外 HBM 写入
- Warp-level reduction（`__shfl_xor_sync`）

### 10.2 分组 TopK CUDA 内核

**文件**：`csrc/moe/grouped_topk_kernels.cu`

```
使用 Bitonic Sort 和 WarpSelect：

1. 将 scores reshape 为 (num_tokens, num_groups, experts_per_group)
2. 每个 warp 在一个 group 内做 top-2 selection
3. 对 group scores 做 Bitonic Sort 选出 topk_group 个组
4. 在选中的组内做最终 TopK
```

### 10.3 Token-Expert 对齐（moe_align_block_size）

**问题**：不同专家分到的 token 数不均匀，CUDA 需要对齐到 block_size 边界。

```
输入: topk_ids (num_tokens, K)
输出: sorted_token_ids   ← 按专家 ID 排序的 token 索引
      expert_ids         ← 对应的专家 ID（对齐到 block_size）
      num_tokens_post_pad ← 总处理量（含 padding）

示例（8个专家，block_size=16，3个token选2个专家）:
  token 0 → expert {2, 5}
  token 1 → expert {2, 7}
  token 2 → expert {5, 3}

  expert 2 处理 token [0, 1]，padding 到 16 个 slot
  expert 3 处理 token [2]，padding 到 16 个 slot
  ...
```

---

## 11. 专家并行（Expert Parallelism）

### 11.1 概念

**Tensor Parallel (TP)**：每个 GPU 持有所有专家的部分权重（沿 intermediate 维度切分）

**Expert Parallel (EP)**：每个 GPU 持有部分专家的完整权重

```
TP（Tensor Parallel）：
  GPU 0: Expert_{0..7} 的 W[:, :I/2]   (全部专家，半个 intermediate)
  GPU 1: Expert_{0..7} 的 W[:, I/2:]   (全部专家，另半个 intermediate)
  通信：每层 all-reduce

EP（Expert Parallel）：
  GPU 0: Expert_{0..3} 的完整权重
  GPU 1: Expert_{4..7} 的完整权重
  通信：all-to-all (token dispatch + result gather)
```

### 11.2 专家映射（determine_expert_map）

```python
# vllm/model_executor/layers/fused_moe/layer.py

def determine_expert_map(
    ep_size: int,        # EP group 大小（如 2）
    ep_rank: int,        # 当前 GPU 的 rank（0 或 1）
    global_num_experts: int,         # 总专家数（如 8）
    expert_placement_strategy: str = "linear",
) -> tuple[int, Tensor, Tensor]:

    local_num_experts = global_num_experts // ep_size   # 4

    if strategy == "linear":
        # Rank 0: [0,1,2,3], Rank 1: [4,5,6,7]
        start = ep_rank * local_num_experts
        local_experts = range(start, start + local_num_experts)

    elif strategy == "round_robin":
        # Rank 0: [0,2,4,6], Rank 1: [1,3,5,7]
        local_experts = range(ep_rank, global_num_experts, ep_size)

    # expert_map[global_id] = local_id（不属于本 rank 的专家为 -1）
    expert_map = torch.full((global_num_experts,), -1)
    for local_id, global_id in enumerate(local_experts):
        expert_map[global_id] = local_id

    return local_num_experts, expert_map, expert_mask
```

### 11.3 All-to-All 通信

EP 的关键挑战：token 需要跨 GPU 发送到其被路由的专家所在的 GPU。

```
[Before All-to-All]
  GPU 0 有 token {A, B, C, D}
  token A 路由到专家 0 (GPU 0) 和专家 5 (GPU 1)
  token B 路由到专家 1 (GPU 0) 和专家 4 (GPU 1)

[All-to-All Dispatch]
  GPU 0 → GPU 1 发送: token {A, B} 的 hidden states（用于专家 4,5）
  GPU 1 → GPU 0 发送: token {C, D} 的 hidden states（若它们路由到专家 0,1,2,3）

[Expert Compute]
  每个 GPU 在本地计算其专家

[All-to-All Gather]
  GPU 0 收回 token {A, B} 在 GPU 1 上的专家输出
  加权合并 → 最终输出
```

**vLLM 支持的 All-to-All 后端**：
- `naive`：broadcast 实现（兼容性最好）
- `pplx`：PPLX 高性能内核
- `deepep_high_throughput`：DeepEP 大 batch 优化
- `deepep_low_latency`：DeepEP 小 batch 低延迟
- `allgather_reducescatter`：基于 allgather + reducescatter

---

## 12. 专家并行负载均衡（EPLB）

### 12.1 问题

MoE 的核心挑战：专家负载不均衡。

```
理想：每个专家处理 N×K/E 个 token（均匀分布）
实际：热门专家可能处理 10× 平均量，冷门专家几乎不被选择

→ 导致：部分 GPU 成为瓶颈，整体吞吐降低
```

### 12.2 三层专家 ID 体系

```
逻辑专家 ID（Logical ID）
  ↓  由训练决定，模型文件中存储 0..E-1
物理专家 ID（Physical ID）
  ↓  包含冗余副本，0..E+R-1（R = 冗余专家数）
GPU 本地专家 ID（Local ID）
     每个 GPU 只持有自己负责的专家
```

### 12.3 冗余专家机制

**热门专家多副本**：
```
逻辑专家 3（被频繁访问）→ 物理专家 3, 17（两个副本，分布在不同 GPU）
逻辑专家 7（被较少访问）→ 物理专家 7（单副本）

eplb_map_to_physical_and_record() 在 routing 时：
  1. 查询 logical_replica_count[logical_id]（该专家有几个副本）
  2. replica_idx = position % replica_count（伪随机选副本，均摊负载）
  3. physical_id = logical_to_physical_map[logical_id, replica_idx]
```

### 12.4 动态重新平衡

```python
# EPLBConfig
window_size = 1000     # 统计最近 1000 步的专家负载
step_interval = 3000   # 每 3000 步重新优化专家放置方案

# 每步更新负载统计
expert_load_view.scatter_add_(0, topk_ids.flatten(), ones)

# 周期性触发：重新计算哪些专家需要冗余副本
# 负载高的逻辑专家 → 增加副本数、分散到更多 GPU
# 负载低的逻辑专家 → 减少副本数、释放 GPU 内存
```

---

## 13. 量化支持

### 13.1 FusedMoEQuantConfig

```python
# vllm/model_executor/layers/fused_moe/config.py

@dataclass
class FusedMoEQuantConfig:
    use_fp8_w8a8: bool       # FP8 权重 + FP8 激活（最高性能，H100）
    use_int8_w8a16: bool     # INT8 权重 + FP16 激活
    use_int4_w4a16: bool     # INT4 权重 + FP16 激活

    # 缩放因子
    w1_scale: Tensor   # (num_experts, 1, I)    专家 w1 权重 scale
    w2_scale: Tensor   # (num_experts, 1, H)    专家 w2 权重 scale
    a1_scale: Tensor   # 激活 scale（静态）
    a2_scale: Tensor

    # 块量化
    block_shape: list[int]   # e.g., [128, 128]，块级 scale
```

### 13.2 内核选择策略

```
FP8 MoE 内核优先级（从高到低）：

1. DeepGemm（FP8 w8a8）
   条件：Hopper（H100）+ batch > 512 + DeepGemm 可用
   特点：最高 FLOPS，使用 E8M0 block scale

2. Cutlass Block Scaled Grouped GEMM
   条件：FP8 w8a8 + Cutlass 支持块量化
   特点：灵活的块量化格式

3. Triton Fused MoE
   条件：通用 fallback
   特点：支持所有量化格式，可配置性最强
```

---

## 14. MoE vs Dense 对比

### 14.1 计算特性对比

| 特性 | Dense FFN | MoE FFN |
|------|-----------|---------|
| 前向计算 | 所有 token × 全部参数 | 所有 token × K/E 参数 |
| 总参数量 | H×I×2 | E×H×I×2 |
| 活跃参数 | H×I×2（100%） | K/E×H×I×2（通常 25%） |
| 内存访问 | 顺序，对 cache 友好 | 稀疏分散，需对齐 |
| 通信量 | All-reduce (TP) | All-to-All (EP) + All-reduce (TP) |
| 负载均衡 | 天然均衡 | 需显式处理（aux_loss 或 EPLB） |

### 14.2 推理挑战

**1. Decode 阶段 batch_size 小 → GEMM 效率低**

```
Dense GEMM（decode）: (1, H) × (H, I)  → GEMM 退化为 GEMV（利用率 <5%）
MoE GEMM（decode）:  (tokens/E, H) × (H, I)  → batch 更小，更低效

解决方案：
- Continuous Batching 汇聚足够多 token
- Speculative Decoding 增加 batch 中的 token 数
```

**2. 专家权重大 → 内存受限**

```
Mixtral-8x7B：8个专家，权重 ~47GB（fp16）
→ 需要多卡 EP 分散存储

DeepSeek-V3：256个专家，权重 ~671GB
→ EP=32 分到 32 张 H100
```

**3. All-to-All 通信开销**

```
EP 下每个 step 需要 2 次 All-to-All（dispatch + gather）
→ 需要高带宽互联（NVLink / InfiniBand）
→ 通信量 ∝ batch_size × top_k × hidden_size × 2
```

### 14.3 vLLM MoE 设计总结

```
灵活的门控机制：
  ├── 标准 Softmax TopK（Mixtral）
  ├── 分组 TopK（DeepSeek）
  ├── noaux_tc 纠正偏置（DeepSeek-V2）
  └── Sigmoid 共享门控（Qwen2）

高性能内核：
  ├── CUDA Fused TopK Softmax（单 kernel 完成）
  ├── Triton Fused MoE GEMM（SwiGLU 融合）
  ├── FP8 DeepGemm（H100 最优路径）
  └── Cutlass Block Scaled GEMM

可扩展并行：
  ├── TP：权重沿 intermediate 切分（all-reduce）
  ├── EP：专家分配到不同 GPU（all-to-all）
  └── TP+EP：混合并行

负载均衡：
  ├── noaux_tc 纠正偏置（训练侧）
  ├── EPLB 冗余专家（推理侧动态重平衡）
  └── round_robin 专家放置策略
```

---

## 总结

**门控网络**是 MoE 的核心组件，本质是一个轻量 Linear 层将 hidden state 映射为 E 维 logits，再通过 Softmax + TopK 选出 K 个专家。vLLM 支持从 Mixtral 的简单 TopK Softmax 到 DeepSeek 的分组 TopK + noaux_tc 纠正偏置等多种门控机制，通过 `FusedMoE` 统一封装，在 CUDA/Triton 层将 routing、GEMM、激活函数、加权聚合融合为少量 kernel 调用，并通过专家并行（EP）+ 负载均衡（EPLB）支持超大规模 MoE 模型（如 DeepSeek-V3 671B）的高效推理。
