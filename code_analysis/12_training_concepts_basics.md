# 训练与推理基础概念：层内、层间、步内、步间、Epoch、Batch Size

> 本文从最基础的概念出发，逐步建立起"一次训练是如何进行的"的完整心智模型。
> 目标读者：理解了 Transformer 架构，但对训练流程的时间维度和空间维度还不够清晰的工程师。

---

## 零、先建立一张"全景图"

理解所有概念之前，先看训练的完整嵌套结构：

```
训练过程（整体）
└── Epoch 1（过完一遍完整数据集）
    └── Step 1（处理一个 batch）
    │   ├── 前向传播（Forward Pass）
    │   │   ├── Layer 0 计算             ← 层内
    │   │   ├── Layer 1 计算             ← 层内
    │   │   │   ↑ 层间（Layer 0 的输出传给 Layer 1）
    │   │   └── ...
    │   ├── 损失计算（Loss）
    │   ├── 反向传播（Backward Pass）
    │   └── 参数更新（Optimizer Step）
    └── Step 2（下一个 batch）
    │   ↑ 步间（Step 1 的参数传给 Step 2）
    └── Step 3 ...
    └── Step N（一个 Epoch 中共 N 步）
└── Epoch 2 ...
└── Epoch K ...
```

**两个维度**：
- **空间维度**：层内 vs 层间（模型结构的层次）
- **时间维度**：步内 vs 步间、Epoch（训练过程的时间粒度）

---

## 一、数据相关概念：Dataset、Epoch、Batch Size、Step

### 1.1 Dataset（数据集）

训练数据的全集，在语言模型中通常是大量的文本：

```
Dataset（以 LLaMA 预训练为例）：
  总量：1.4 万亿 token（1.4T tokens）
  存储：数百 TB 的文本文件

  具体包含：
  - Common Crawl（网页文本，67%）
  - Books（书籍，4.5%）
  - Wikipedia（百科，4.5%）
  - ...

  存储格式：通常是 tokenized 后的 token ID 序列，
           存为二进制文件（.bin / .tfrecord / .parquet）
```

### 1.2 Batch Size（批大小）⭐

**定义**：模型**一次前向传播**中同时处理的**训练样本数量**。

```
为什么不一次处理一个样本（Batch Size = 1）？

  原因 1：GPU 利用率
    矩阵乘法：Y = X · W
    X 形状 [1, 4096]（单样本）：
      → 一个 4096 维向量 × [4096, 4096] 矩阵
      → GPU 上几乎所有计算单元空闲（向量乘矩阵 ≠ 矩阵乘矩阵）
      → Tensor Core 利用率 < 5%

    X 形状 [512, 4096]（Batch=512）：
      → 矩阵乘矩阵，满足 Tensor Core 的并行需求
      → 利用率 > 80%

  原因 2：梯度估计质量
    单样本梯度：噪声极大，可能完全朝错误方向更新
    大 batch 梯度：多个样本的平均，更接近真实梯度方向
    → 更稳定的训练
```

**Batch Size 的多个含义（容易混淆的地方）**：

```
概念层次（从小到大）：

Micro Batch Size（微批大小）：
  = 单张 GPU 在一次前向传播中实际处理的样本数
  例如：micro_batch_size = 4（每张卡处理 4 个 sequence）

Global Batch Size（全局批大小）：
  = 模型参数一次更新用到的总样本数
  = micro_batch_size × data_parallel_size × gradient_accumulation_steps

  例如：
    micro_batch_size = 4
    data_parallel_size = 64（64 张卡做 DP）
    gradient_accumulation_steps = 4（累积 4 步再更新）
    → global_batch_size = 4 × 64 × 4 = 1024

Effective Batch Size（有效批大小）= Global Batch Size
```

**DeepSeek-V3 的 Batch Size 设置**：

```
官方技术报告：
  "We set the batch size to gradually increase from 3072 to 15360 sequences
   during the early training stages, and eventually set it to
   15360 sequences × 4096 tokens = 61,440K tokens per step"

  解读：
    global_batch_size = 15360 个 sequence × 4096 tokens/sequence
                      = 62,914,560 tokens ≈ 63M tokens per step

  每步处理 6300 万个 token，
  用于更新一次 671B 参数！
```

**Batch Size 与训练稳定性的关系**：

```
太小（< 256 tokens）：
  梯度噪声大 → 训练不稳定，loss 曲线抖动剧烈
  GPU 利用率低

刚好（~2K~64K tokens）：
  梯度质量好，训练稳定
  与学习率配合（通常 lr ∝ sqrt(batch_size) 或 lr ∝ batch_size）

太大（> 1M tokens，一般 LLM 训练不会这么大）：
  梯度太"平滑"，模型可能陷入平坦区域，泛化能力下降
  （Large Batch Training 是个专门的研究方向）
```

---

### 1.3 Step（步，也叫 Iteration 迭代）⭐

**定义**：一次完整的**"前向 + 反向 + 参数更新"**过程。每个 step 消耗一个 global batch。

```
Step 的详细时间线：

  t=0ms:  读取 batch（从 CPU 内存异步传输到 GPU）
  t=2ms:  前向传播（Forward）
            → 逐层计算激活值（Layer 0 → Layer 1 → ... → Layer N）
            → 得到 logits（词汇表上的分布）
  t=12ms: 损失计算
            → cross_entropy(logits, labels)  ← 语言模型是预测下一个 token
            → 得到 scalar loss 值
  t=14ms: 反向传播（Backward）
            → 自动微分，计算每个参数的梯度 ∂Loss/∂W
            → 逐层从后往前（Layer N → Layer N-1 → ... → Layer 0）
  t=24ms: 梯度同步（分布式 DP 时）
            → AllReduce，让所有 DP rank 梯度平均
  t=28ms: 参数更新（Optimizer Step）
            → Adam: m = β₁m + (1-β₁)g; v = β₂v + (1-β₂)g²; W -= lr * m/√v
  t=30ms: 进入下一个 Step

  即一个 Step ≈ 30ms（LLaMA-7B 训练，单机 8×A100 的估算）
```

**Step Count（步数）** 是训练进度的核心度量：

```
  训练目标：1.4T tokens（LLaMA-7B）
  global_batch_size = 4M tokens

  总步数 = 1.4T / 4M = 350,000 步（35万步）

  DeepSeek-V3：
    总 tokens = 14.8T
    batch_size = 63M tokens/step
    总步数 ≈ 14.8T / 63M ≈ 235,000 步（约23.5万步）
```

---

### 1.4 Epoch（轮次）

**定义**：训练集中**每个样本被模型看一遍**叫做 1 个 Epoch。

```
Epoch 的直觉：
  想象你在背英语单词（词汇表 = 训练数据集）：
  把所有单词过一遍 = 1 Epoch
  把所有单词过三遍 = 3 Epochs

计算关系：
  1 Epoch 的 Step 数 = 数据集大小 / global_batch_size

  例：MNIST（6万张图）训练分类器：
    global_batch_size = 128
    1 Epoch = 60000 / 128 ≈ 469 步

  例：LLaMA-3-8B（15T tokens）：
    global_batch_size = 4M tokens
    1 Epoch = 15T / 4M = 3,750,000 步（375万步）
    → 实际只训练 1 Epoch（步数已经足够）！
```

**Epoch vs Step：哪个更重要？**

```
图像/NLP 传统模型：
  通常训练 10~100 Epochs（数据集小，需要反复看）
  常见配置："训练 30 epochs，每 10 epochs 学习率衰减 0.1 倍"

大型语言模型（GPT、LLaMA、DeepSeek）：
  数据集太大，通常只训练 1 Epoch，有时甚至不到 1 Epoch
  → Epoch 概念不那么重要，Step Count 更常用

  例：
    LLaMA-2-70B 训练了 2T tokens，而训练集有 2T tokens
    → 恰好 1 Epoch

    Chinchilla（DeepMind）定律：
    最优 token 数 ≈ 20 × 模型参数数量
    70B 参数 → 最优训练 tokens ≈ 1.4T（约 1 Epoch on 1.4T 数据集）
```

---

### 1.5 数据维度的完整关系图

```
数据集（Dataset）
  ├── 1.4 万亿 tokens（LLaMA-2）
  │
  ├── 被切割成 Sequence（序列）
  │   每条 sequence = 2048 tokens（context window）
  │   → 1.4T / 2048 ≈ 6.8 亿条 sequences
  │
  ├── 被分成 Global Batch
  │   每个 global_batch = micro_batch × DP × grad_accum
  │   例：2048条 × 2048 tokens = 4M tokens
  │   → 1.4T / 4M = 350,000 steps
  │
  └── 1 Epoch = 350,000 steps（对于此配置）

  训练通常 1~2 Epochs，总 Step 数为关键进度指标
```

---

## 二、模型结构中的"层"概念

### 2.1 什么是"层"（Layer）

神经网络由多个"层"堆叠而成，每一层是一个**可学习的变换函数**：

```
输入                        输出
 ↓                           ↑
[B, S, H] → LayerN(·) → [B, S, H]

每层包含的子模块（LLaMA Decoder Layer）：
  ┌─────────────────────────────────────┐
  │  input_layernorm                    │
  │  self_attention（q/k/v/o_proj）      │
  │  post_attention_layernorm           │
  │  mlp（gate/up/down_proj）            │
  │  残差连接（residual connections）    │
  └─────────────────────────────────────┘
```

**层的类型**：

```
按位置分：
  Embedding Layer（嵌入层）：token ID → 向量
  Transformer Layers（N 个 Decoder Block）：核心计算
  Final Norm + LM Head：输出到词汇表

按功能分：
  注意力层（Attention）：序列内 token 间的交互
  前馈层（FFN/MLP）：逐 token 的非线性变换
  归一化层（LayerNorm/RMSNorm）：稳定训练
  激活函数（SiLU/GELU）：引入非线性

LLaMA-3-8B 的层结构：
  1 个 Embedding 层
  32 个 Decoder Layers（每层 ~218M 参数）
  1 个 Final Norm + LM Head
  总计：~8B 参数
```

---

### 2.2 层内（Intra-Layer）⭐

**定义**：发生在**单个层内部**的计算和通信。

```
"层内"的边界：从该层输入开始，到该层输出结束

Layer i 的层内过程：
  输入: h[B, S, H]
  ↓
  │ [RMSNorm]：h_norm = h / RMS(h) × w
  │ [Q Proj]：Q = h_norm · W_q  ← 层内计算
  │ [K Proj]：K = h_norm · W_k  ← 层内计算
  │ [V Proj]：V = h_norm · W_v  ← 层内计算
  │ [Flash Attention]：Attn = softmax(QKᵀ/√d) · V
  │ [O Proj]：out = Attn · W_o  ← 层内计算
  │ [残差]：h = h + out
  │ [RMSNorm]：h_norm2 = h / RMS(h) × w2
  │ [Gate/Up Proj]：gate, up = h_norm2 · W_gate, h_norm2 · W_up
  │ [SwiGLU]：ffn_out = SiLU(gate) ⊙ up
  │ [Down Proj]：ffn_out = ffn_out · W_down
  │ [残差]：h = h + ffn_out
  ↓
  输出: h[B, S, H]
```

**层内通信（TP 的核心）**：

TP 并行切分的是**层内的矩阵乘法**，因此 TP 通信发生在**层内**：

```
层内的 TP 通信（一个 Decoder Layer，TP=4）：

GPU 0,1,2,3 同时进入 Layer i：
  ↓
  各自计算 Q₀,K₀,V₀ / Q₁,K₁,V₁ / ...（各自不同的 head，无通信）
  ↓
  各自独立做 Flash Attention（各自的 head）
  ↓
  O Proj（行并行）→ 各自得到部分和
  ↓
  ★ AllReduce ← 层内通信点！（同步 o_proj 的部分和）
  ↓
  各卡都有完整 [B, S, H]
  ↓
  Gate/Up Proj → SwiGLU（各自 3584/4096 的维度）
  ↓
  Down Proj（行并行）→ 各自得到部分和
  ↓
  ★ AllReduce ← 层内通信点！（同步 down_proj 的部分和）
  ↓
  进入 Layer i+1
```

**"层内"这个词在并行系统中的用法**：

```
"层内并行" = TP（Tensor Parallelism）
  → 单层的计算被切分到多卡上
  → 通信（AllReduce）必须在本层结束前完成
  → 属于"层内通信"

对比：
"层间并行" = PP（Pipeline Parallelism）
  → 不同层分配到不同卡
  → 通信（P2P）在层与层之间发生
  → 属于"层间通信"
```

---

### 2.3 层间（Inter-Layer）⭐

**定义**：发生在**相邻层之间**的数据流动（一层的输出成为下一层的输入）。

```
层间的数据流：

Layer i   → 输出 h_i: [B, S, H]
                ↓
              "层间"边界
                ↓
Layer i+1 → 接收 h_i 作为输入
```

**层间通信（PP 的核心）**：

PP 把不同层分到不同 GPU，层与层之间的激活值传输就是**跨 GPU 的层间通信**：

```
PP=4，32层模型，每 Stage 8层：

Stage 0（GPU 0）: Layer 0~7
  Layer 7 的输出 h₇: [B, S, H]
        ↓
    ★ P2P Send ← 层间通信！
        ↓
Stage 1（GPU 1）: Layer 8~15
  接收 h₇ 作为 Layer 8 的输入
  ...
```

**层间的依赖关系**：

```
关键约束：Layer i+1 必须等 Layer i 计算完成后才能开始！

这造成了 PP 的 Pipeline Bubble：
  GPU 0 完成前 8 层 → 传给 GPU 1
  GPU 1 开始计算时，GPU 0 只能等待（idle）
  → 等下一个 microbatch 来才能继续工作

1F1B 调度的解决思路：
  GPU 0 在等 GPU 1 时，立刻开始下一个 microbatch 的前 8 层
  → 层间通信与下一个 batch 的层内计算重叠
```

---

## 三、训练时间维度：步内与步间

### 3.1 步内（Intra-Step）⭐

**定义**：一次 Step（前向 + 反向 + 更新）**内部**发生的事情。

```
步内的时间线（完整展开）：

Step N 的步内：
─────────────────────────────────────────────────────
  Phase 1：数据准备（异步，与上一步计算重叠）
    → 从 CPU 内存 DMA 传输 batch 到 GPU 显存
    → Tokenization 已预处理好，直接读取 token ID

  Phase 2：前向传播（Forward）
    → Layer 0: embedding lookup
    → Layer 1..32: Attention + FFN（各层顺序执行）
      ├─ 步内的层内通信：TP AllReduce
      └─ 步内的层间通信：PP P2P
    → Layer 33: lm_head → logits

  Phase 3：损失计算（Loss Computation）
    → loss = cross_entropy(logits, labels)
    → 在 GPU 上计算（标量值）

  Phase 4：反向传播（Backward）
    → 自动微分引擎（Autograd）
    → 从 loss 开始，反向穿过每一层
    → 计算每个参数的梯度：∂loss/∂W
    → 步内的梯度通信：DP AllReduce（异步，与反向计算重叠）

  Phase 5：参数更新（Optimizer Step）
    → Adam 更新：利用梯度更新所有参数
    → 涉及：一阶矩(m)、二阶矩(v)、参数(W)三份存储

  Phase 6：日志/Checkpoint（可选）
    → 记录 loss、learning_rate、grad_norm
    → 每 N 步保存一次 checkpoint
─────────────────────────────────────────────────────
```

**步内的并行通信可见性**：

```
TP 通信：发生在步内的前向（每层 ×2次）和反向（每层 ×2次）
EP 通信：发生在步内的前向（每 MoE 层 ×2次）
PP 通信：发生在步内的前向（Stage 间）和反向（Stage 间）
DP 通信：发生在步内的反向结束后（与最后几层的反向重叠）

→ 所有通信都在步内完成！
→ 步内通信是性能优化的主战场（如何让这些通信与计算重叠）
```

---

### 3.2 步间（Inter-Step）⭐

**定义**：**相邻两次 Step 之间**的关系（一步的结束状态是下一步的起始状态）。

**步间传递的核心内容：参数（模型权重）**

```
Step N 结束后，模型参数被更新为 W_new
Step N+1 开始时，使用更新后的 W_new 做前向传播

这是步间"通信"的本质：
  不是网络传输，而是 GPU 显存中参数的版本更新

步间状态的完整传递：
  训练状态 = {
    W（模型参数）     ← 每步更新
    m（Adam 一阶矩）   ← 每步更新
    v（Adam 二阶矩）   ← 每步更新
    step（步数计数器）  ← 每步 +1
    lr（当前学习率）    ← 根据调度器更新
    rng_state（随机数种子状态）← 用于 Dropout/数据采样
  }

步间不需要网络通信（所有状态都在 GPU 本地），
但需要从前一步的更新结果"接续"。
```

**步间的依赖关系**：

```
Step N 的参数更新 → Step N+1 的前向传播
（强依赖：不能并行，必须串行）

梯度累积（Gradient Accumulation）打破了这种严格依赖：
  正常：每步都更新参数
  累积：每 K 步才更新一次参数

  步 1、步 2、...步 K：只做前向 + 反向，累积梯度（不更新参数）
  步 K 结束：梯度求和 → 参数更新
  步 K+1、...步 2K：继续累积

  效果：等价于把 global_batch_size 扩大 K 倍（无需额外显存）
  代价：步间独立性（步 1~K 都用相同的参数，不是最优的）
```

**步间的学习率调度（LR Schedule）**：

```
步间的最重要变化之一：学习率随步数变化

常见调度策略（以 LLaMA 为例）：
  Warmup 阶段（步 0 到 步 2000）：
    lr 从 0 线性增加到 max_lr = 3e-4
    → 防止初始大梯度破坏权重（权重随机初始化，初始梯度不可靠）

  Cosine Decay 阶段（步 2000 到训练结束）：
    lr = max_lr × 0.5 × (1 + cos(π × step/total_steps))
    → 从 3e-4 缓慢降到 3e-5（衰减 10 倍）

  步数与学习率的对应：
  Step:  0    2k    50k   100k  200k  350k
  LR:    0   3e-4  2.5e-4 2e-4  1e-4  3e-5
                                       ↑ 收敛
```

---

### 3.3 步内 vs 步间：什么操作属于哪个维度

```
步内（同一个 batch 内部发生的）：
  ✓ 前向传播（所有层的计算）
  ✓ 损失计算
  ✓ 反向传播（所有梯度计算）
  ✓ TP/PP/EP 通信（都在同一步内完成）
  ✓ 梯度裁剪（clip_grad_norm）
  ✓ 参数更新（optimizer.step）
  ✓ 学习率更新（scheduler.step）

步间（跨步骤存在的状态）：
  ✓ 模型参数 W（核心！每步更新）
  ✓ 优化器状态（m、v、step_count）
  ✓ KV Cache（推理时）
  ✓ 随机数状态（确保可复现）
  ✓ 数据集迭代器位置（下一步从哪读数据）

Epoch 结束时（步间的特殊边界）：
  ✓ 数据集随机 Shuffle（重新打乱顺序）
  ✓ 验证集评估（Validation）
  ✓ Checkpoint 保存
```

---

## 四、推理时的概念对应

训练中的"步"在推理中对应不同的概念：

### 4.1 推理的"请求"与"步"

```
推理时没有"训练 Step"，而是"Decode Step"：

Decode Step（解码步）：
  = 生成一个新 token 的完整过程

  t=0: Prefill（预填充）
       输入 prompt（例如"帮我写一首诗："）
       → 一次前向传播，处理整个 prompt
       → 生成第一个 token（"春"）
       → KV Cache 填满 prompt 的 K/V 值

  t=1: Decode Step 1
       输入：["春"]（上一步生成的）
       → 前向传播（只处理 1 个新 token，但用到完整 KV Cache）
       → 生成第二个 token（"风"）

  t=2: Decode Step 2
       输入：["风"]
       → 前向传播 → 生成"送"
  ...

  直到生成 <EOS>（结束符）

→ 推理的"步间"：每一个 Decode Step 的 KV Cache 是上一步的扩展
  KV Cache 步间增长：每步追加 1 行（[1, num_heads, head_dim]）
```

### 4.2 Prefill vs Decode 的层内/层间差异

```
Prefill（输入长序列）：
  批形状：[1, S_prompt, H]，例如 S_prompt=1024

  层内：
    Q/K/V: [1, 1024, H] → Flash Attention（计算密集，Compute Bound）
    FFN: [1, 1024, H] → 矩阵乘法

  层间：激活 [1, 1024, H] 在层间流动
  → Prefill 是 Compute Bound（计算是瓶颈）

Decode（每步只有 1 个新 token）：
  批形状：[B, 1, H]，例如 B=32（32个并发请求）

  层内：
    Q: [32, 1, H]，K/V 需要 Attend 到 KV Cache [32, S_cache, H]
    Flash Attention 变成 Flash Decoding（Memory Bound）
    FFN: [32, 1, H] × W → 向量乘矩阵（低 GPU 利用率）

  层间：激活 [32, 1, H] 在层间流动
  → Decode 是 Memory Bound（显存带宽是瓶颈）
```

---

## 五、综合案例：追踪一个 Training Step 的完整数据流

以 LLaMA-3-8B，TP=1，PP=1，DP=8，global_batch_size=512（序列长度=2048）为例：

### Step 0 开始

```
数据准备：
  512 条 sequences（每条 2048 tokens），分给 8 张卡
  每张卡：512/8 = 64 条 sequences（micro_batch=64）
  数据形状（每卡）：[64, 2048]（token IDs，整数）
```

### 步内 - 前向传播

```
层间传递：
  embed_tokens: [64, 2048] → [64, 2048, 4096]  ← token IDs → 向量

  Layer 0（层内）：
    输入：h = [64, 2048, 4096]
    RMSNorm → [64, 2048, 4096]
    q_proj  → Q: [64, 2048, 4096]（32头×128维）
    k_proj  → K: [64, 2048, 1024]（8头×128维，GQA）
    v_proj  → V: [64, 2048, 1024]
    FlashAttn → [64, 2048, 4096]
    o_proj  → [64, 2048, 4096]
    + 残差  → h: [64, 2048, 4096]
    RMSNorm → [64, 2048, 4096]
    gate_up_proj → [64, 2048, 28672]（合并 gate+up）
    SwiGLU  → [64, 2048, 14336]
    down_proj → [64, 2048, 4096]
    + 残差  → h: [64, 2048, 4096]   ← Layer 0 输出
    （层间）↓
  Layer 1（层内）：...重复上述过程
  ...
  Layer 31（层内）：输出 h: [64, 2048, 4096]

  model.norm → [64, 2048, 4096]
  lm_head   → logits: [64, 2048, 32000]
```

### 步内 - 损失计算

```
loss = cross_entropy(logits[:, :-1, :], tokens[:, 1:])
     = 预测每个位置的下一个 token

形状：
  logits[:, :-1, :] = [64, 2047, 32000]（每个位置的预测）
  tokens[:, 1:]     = [64, 2047]（真实的下一个 token）

  loss = -log(softmax(logits)[真实 token 位置])
       → 每个位置一个 loss 值，平均 → scalar loss
```

### 步内 - 反向传播

```
从 loss 开始，自动微分向后传播：

  ∂loss/∂logits → ∂loss/∂lm_head.weight
  → ∂loss/∂h_31 （Layer 31 的输出）
  → 通过 Layer 31 的反向：
      ∂loss/∂down_proj.weight
      ∂loss/∂gate_up_proj.weight
      ∂loss/∂o_proj.weight
      ∂loss/∂qkv_proj.weight
      ∂loss/∂input_layernorm.weight
      ∂loss/∂h_30 （传给 Layer 30）
  → 依次 Layer 30, 29, ..., 0

同时（与反向计算重叠）：
  Layer 31 的梯度算完 → 立刻发起 DP AllReduce（步内通信）
  Layer 30 的梯度算完 → 立刻发起 DP AllReduce
  ...（流水线式梯度同步）
```

### 步内 - 参数更新

```
所有梯度 AllReduce 完成后：

Adam 更新每个参数（示例：Layer 0 的 q_proj.weight）：
  g = ∂loss/∂q_proj.weight（经过 AllReduce 的全局平均梯度）
  m = 0.9 × m + 0.1 × g    （一阶矩，方向）
  v = 0.999 × v + 0.001 × g²（二阶矩，幅度）
  W = W - lr × m / (√v + ε)  （参数更新）

全部 ~8B 参数都要做上述计算：
  参数本身：8B × 2 bytes = 16 GB（BF16）
  一阶矩 m：8B × 4 bytes = 32 GB（FP32，Adam 需要高精度）
  二阶矩 v：8B × 4 bytes = 32 GB（FP32）
  总：~80 GB！→ 单卡 80GB 恰好能放下 7/8B 模型的训练状态
```

### 步间 - 进入 Step 1

```
Step 0 结束，模型参数 W 已更新为 W_new
Step 1 开始：读取下一个 batch，用 W_new 进行前向传播

步间的关键不变量：
  ✓ W_new 在所有 8 张 DP 卡上完全相同（因为 AllReduce 保证了梯度相同）
  ✓ Adam 状态（m, v）同步更新
  ✓ step_count = 1（用于学习率计算）
  ✓ lr = warmup_schedule(1)（步 1 的学习率）
```

---

## 六、概念速查表

| 概念 | 简明定义 | 维度 | 分布式对应 |
|------|---------|------|-----------|
| **Dataset** | 全部训练数据 | — | — |
| **Epoch** | 完整过一遍数据集 | 时间（最大粒度）| — |
| **Step/Iteration** | 一次前向+反向+更新 | 时间（基本粒度）| 步间参数传递 |
| **Global Batch Size** | 一步处理的总样本数 | — | = micro_batch × DP × grad_accum |
| **Micro Batch Size** | 单卡单步样本数 | — | TP/PP 的基本计算单元 |
| **层（Layer）** | 一个可学习的变换模块 | 空间 | PP 的切分单元 |
| **层内（Intra-Layer）** | 单层内部的计算 | 空间（细粒度）| **TP 通信发生在层内** |
| **层间（Inter-Layer）** | 相邻层之间的传递 | 空间（粗粒度）| **PP 通信发生在层间** |
| **步内（Intra-Step）** | 单步内部的所有操作 | 时间（细粒度）| TP/PP/EP 通信都在步内 |
| **步间（Inter-Step）** | 相邻步之间的状态传递 | 时间（粗粒度）| 参数更新（无网络通信）|
| **Decode Step** | 推理时生成一个 token | 时间（推理粒度）| KV Cache 步间增长 |

---

## 七、一句话概括各概念的本质

```
Epoch  ：我把所有数据都看了一遍
Step   ：我看了一批数据并更新了参数
Batch  ：这一批数据（数量是 batch_size）

层内   ：单层的矩阵乘法、注意力计算（TP 在这里切分）
层间   ：激活值从一层"流"到下一层（PP 在这里切分）
步内   ：一步内前向→反向→更新的全过程（TP/PP/EP 通信都在步内）
步间   ：上一步的参数"继承"给下一步（参数更新是步间的本质）
```

---

*参考资料：*
- *[LLaMA 技术报告：Touvron et al. 2023](https://arxiv.org/abs/2302.13971)*
- *[Chinchilla Scaling Laws：Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)*
- *[DeepSeek-V3 Technical Report：训练配置细节](https://arxiv.org/abs/2412.19437)*
- *[Megatron-LM 3D 并行论文：Narayanan et al. 2021](https://arxiv.org/abs/2104.04473)*
- *[Large Batch Training：Goyal et al. 2017](https://arxiv.org/abs/1706.02677)*
*更新：2026-03*
