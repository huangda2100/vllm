# 层归一化（Layer Normalization）

## 目录
1. [归一化解决什么问题](#1-归一化解决什么问题)
2. [归一化的基本思想](#2-归一化的基本思想)
3. [Batch Norm vs Layer Norm](#3-batch-norm-vs-layer-norm)
4. [LayerNorm 完整推导](#4-layernorm-完整推导)
5. [RMSNorm：现代 LLM 的选择](#5-rmsnorm现代-llm-的选择)
6. [归一化的位置：Pre-Norm vs Post-Norm vs Sandwich-Norm](#6-归一化的位置pre-norm-vs-post-norm-vs-sandwich-norm)
7. [不同归一化范围的对比](#7-不同归一化范围的对比)
8. [数值例子：从头算一遍](#8-数值例子从头算一遍)
9. [vLLM 中的实现](#9-vllm-中的实现)

---

## 1. 归一化解决什么问题

### 1.1 没有归一化时的灾难

假设每一层的输出**平均**放大 1.1 倍：

```
第 1 层输出量级:  1.0
第 10 层输出量级: 1.1^10  ≈ 2.6
第 32 层输出量级: 1.1^32  ≈ 21.1
第 100 层输出量级: 1.1^100 ≈ 13,781

即使每层只放大 10%，100 层后数值已经爆炸到 13000+
```

反过来，如果每层平均缩小 0.9 倍：

```
第 32 层: 0.9^32  ≈ 0.034    信号几乎消失
第 100 层: 0.9^100 ≈ 0.000027  完全消失
```

**问题不止是数值大小**，还有分布漂移：

```
训练初期第 5 层的输入：均值 ≈ 0，标准差 ≈ 1
训练 1000 步后第 5 层的输入：均值 ≈ 2.3，标准差 ≈ 5.7

第 5 层的权重是基于"均值0，方差1"的输入学习的
输入分布变了 → 该层学到的特征突然"失效"
→ 第 5 层被迫重新适应 → 导致后续所有层的分布也变
→ 整个网络在"追着自己的尾巴跑"
```

这就是 **Internal Covariate Shift**（内部协变量偏移）。

### 1.2 归一化的目标

**让每一层的输入保持在稳定的分布**（约"均值 0，方差 1"）：

```
无归一化: 每层输入分布 → 不确定，随训练和输入不断漂移
有归一化: 每层输入分布 → 强制拉回到"标准化"状态

类比：
  无归一化 = 在摇晃的船上射击（目标一直在动）
  有归一化 = 在固定的靶场射击（目标稳定）
  → 参数学习更容易收敛
```

---

## 2. 归一化的基本思想

### 2.1 核心公式

```
给定一组数字 x₁, x₂, ..., xₙ

步骤 1：计算均值
  μ = (x₁ + x₂ + ... + xₙ) / n

步骤 2：计算方差
  σ² = [(x₁-μ)² + (x₂-μ)² + ... + (xₙ-μ)²] / n

步骤 3：归一化
  x̂ᵢ = (xᵢ - μ) / √(σ² + ε)

步骤 4：仿射变换（可学习参数 γ, β）
  yᵢ = γ × x̂ᵢ + β
```

### 2.2 为什么要步骤 4？

如果只做步骤 1-3，所有层的输出都被**强制**压成均值 0、方差 1。这过于死板——有时网络**需要**输出均值不为 0 或方差不为 1 的分布。

可学习参数 γ（缩放）和 β（平移）让网络可以"选择性地撤销归一化"：

```
如果网络学到 γ = σ 且 β = μ → 完全撤销归一化（y = x）
如果网络学到 γ = 1 且 β = 0 → 保持归一化
通常学到的是介于两者之间的值

→ 网络自己决定"归一化多少"，而非人为强制
```

---

## 3. Batch Norm vs Layer Norm

归一化有多种方式，关键区别在于**"沿哪个维度求均值和方差"**。

### 3.1 图解

```
假设输入张量形状: (B, N, H) = (batch_size, seq_len, hidden_size)

                  ┌─────── H (hidden_size) ──────┐
                  │                               │
              ┌───┼───┐                           │
         B    │   │   │                           │
       (batch)│ token₁│ [0.2, -0.5, 1.3, ..., 0.8]  │
              │ token₂│ [0.1,  0.3, 0.7, ..., 0.4]  │
              │ token₃│ [...]                     │
              └───┼───┘                           │
              ┌───┼───┐                           │
              │ token₁│ [...]                     │
              │ token₂│ [...]                     │
              └───┼───┘                           │
                  │                               │
                  └───────────────────────────────┘

Batch Norm（沿 B 维度归一化）：
  对同一个特征维度（同一列），跨所有样本和所有 token 求均值/方差
  → 统计量来自一个 batch 的所有样本

Layer Norm（沿 H 维度归一化）：
  对同一个 token（同一行），跨所有特征维度求均值/方差
  → 统计量只来自该 token 自己的 H 个数值
```

### 3.2 为什么 LLM 用 Layer Norm 而非 Batch Norm

| 特性 | Batch Norm | Layer Norm |
|------|-----------|-----------|
| 统计范围 | 跨样本（整个 batch） | 单个样本内（一行/一个 token） |
| batch_size=1 时 | 统计不稳定（只有一个样本） | 完全不受影响 |
| 变长序列 | 需要 padding，短序列引入噪声 | 每个 token 独立归一化 |
| 推理时 | 需要维护 running_mean/var | 无需额外状态 |
| 训练/推理差异 | 有（训练用 batch 统计，推理用 running 统计） | 无差异 |

**LLM 推理的典型场景**：
- Decode 阶段 batch_size=1（逐 token 生成）→ Batch Norm 完全失效
- 序列长度动态变化 → Batch Norm 需要 padding，引入误差
- Layer Norm 对每个 token 独立操作，天然适合自回归生成

---

## 4. LayerNorm 完整推导

### 4.1 定义

对 hidden_size 维度的向量 $\mathbf{x} = [x_1, x_2, \dots, x_H]$ 做归一化：

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

其中：
- $\mu = \frac{1}{H} \sum_{i=1}^{H} x_i$（均值）
- $\sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2$（方差）
- $\gamma \in \mathbb{R}^H$（可学习的缩放参数，初始化为全 1）
- $\beta \in \mathbb{R}^H$（可学习的平移参数，初始化为全 0）
- $\epsilon$（极小常数，如 $10^{-6}$，防止除零）
- $\odot$ 表示逐元素乘法

### 4.2 反向传播的梯度

LayerNorm 的梯度推导（理解为什么它有效）：

```
∂L/∂xᵢ = (γᵢ / σ) × [∂L/∂x̂ᵢ - mean(∂L/∂x̂) - x̂ᵢ × mean(x̂ × ∂L/∂x̂)]

关键性质:
  ① 梯度被 1/σ 缩放 → 如果某层输出方差很大，梯度被自动缩小（防爆炸）
  ② 梯度减去均值 → 梯度也被"中心化"，方向更稳定
  ③ 不依赖 batch 中其他样本 → 各样本梯度独立
```

---

## 5. RMSNorm：现代 LLM 的选择

### 5.1 动机

LayerNorm 需要计算均值 $\mu$ 和方差 $\sigma^2$，涉及两次归约（reduction）操作。

Zhang & Sennrich (2019) 发现：**去掉"减均值"这一步，效果几乎不变，但省了一次归约**。

### 5.2 定义

$$\text{RMSNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})}$$

其中：

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{H}\sum_{i=1}^{H} x_i^2 + \epsilon}$$

与 LayerNorm 的区别：
```
LayerNorm:  (x - 均值) / 标准差 × γ + β     两个可学习参数 γ, β
RMSNorm:    x / 均方根 × γ                   一个可学习参数 γ（无 β）

少了什么：
  ✗ 不减均值（不做中心化）
  ✗ 无 β 偏移参数

为什么可以省？
  实验表明：
    1. 减均值对 Transformer 的贡献不大
       （残差连接已经让输出均值接近 0）
    2. β 偏移可以被后续线性层的 bias 吸收
```

### 5.3 RMSNorm 对比 LayerNorm

| 特性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 中心化（减均值） | 是 | 否 |
| 归一化 | 除以标准差 σ | 除以均方根 RMS |
| 可学习参数 | γ（缩放）+ β（偏移） | 仅 γ（缩放） |
| 参数量 | 2H | H |
| 计算量 | 2 次归约（mean + var） | 1 次归约（mean of squares） |
| 效果 | 基准 | 接近，训练可能更稳定 |
| 使用模型 | BERT, GPT-2 | LLaMA, DeepSeek, Qwen, Mistral |

---

## 6. 归一化的位置：Pre-Norm vs Post-Norm vs Sandwich-Norm

"归一化放在哪里"对模型训练的稳定性和最终性能有显著影响。

### 6.1 Post-Norm（原始 Transformer, 2017）

```
                  x
                  │
                  ▼
              SubLayer          ← Attention 或 FFN
                  │
                  ▼
              x + SubLayer(x)   ← 先做残差加法
                  │
                  ▼
              LayerNorm(↑)      ← 再归一化
                  │
                  ▼
                output

代码：
  output = LayerNorm(x + Attention(x))
  output = LayerNorm(output + FFN(output))
```

**梯度分析**：

```
反向传播时，梯度必须穿过 LayerNorm 才能到达残差分支

∂L/∂x = ∂L/∂output × ∂LayerNorm/∂input × (1 + ∂SubLayer/∂x)
                       ↑
                 LayerNorm 的梯度可能不稳定

底层（靠近输入的层）累积了多个 LayerNorm 的梯度 → 不稳定
→ 需要 Learning Rate Warmup（训练前期用很小的学习率逐步升温）
→ 深层网络（>100层）训练困难
```

**优点**：训练收敛后，最终性能可能略好（归一化在残差路径上，约束更强）。

**缺点**：训练不稳定，深层网络需要精心调参。

### 6.2 Pre-Norm（现代 LLM 标准）

```
                  x ──────────────────┐
                  │                    │ skip connection
                  ▼                    │
              LayerNorm(x)             │  ← 先归一化
                  │                    │
                  ▼                    │
              SubLayer                 │  ← Attention 或 FFN
                  │                    │
                  └────── + ───────────┘  ← 残差加法
                          │
                          ▼
                        output

代码：
  output = x + Attention(LayerNorm(x))
  output = output + FFN(LayerNorm(output))
```

**梯度分析**：

```
反向传播时，梯度可以直接通过残差分支回传，不经过 LayerNorm

∂L/∂x = ∂L/∂output × 1                   ← 残差直通路
       + ∂L/∂output × ∂SubLayer/∂(LN(x)) × ∂LN/∂x   ← 子层路径

关键：常数 1 保证梯度畅通，无论 LayerNorm 的梯度如何
→ 训练非常稳定
→ 无需 warmup
→ 支持 1000+ 层
```

**优点**：训练稳定，容易扩展到大模型。

**缺点**：最终性能可能比 Post-Norm 略低（但差距很小，且稳定性带来的实际好处远大于这点差距）。

### 6.3 Sandwich-Norm（CogView, 2021）

在 Sub-Layer 的输入**和输出**都做归一化：

```
              x ──────────────────┐
              │                    │
              ▼                    │
          LayerNorm(x)    ← 入口归一化    │
              │                    │
              ▼                    │
          SubLayer                 │
              │                    │
              ▼                    │
          LayerNorm(↑)    ← 出口归一化    │
              │                    │
              └────── + ───────────┘
                      │
                      ▼
                    output

代码：
  output = x + LayerNorm( Attention( LayerNorm(x) ) )
```

**动机**：某些大模型（特别是多模态模型）训练时仍会出现 loss spike（突然的损失跳升）。出口归一化可以抑制 SubLayer 输出中偶尔出现的异常大值。

**缺点**：多一次 LayerNorm，计算开销略增。

### 6.4 DeepNorm（深层 Transformer, 2022）

微软提出的方案，用于训练超深网络（1000 层）：

```
output = LayerNorm(α × x + SubLayer(x))

α > 1（如 α = (2N)^{1/4}，N 为层数）

通过放大残差的比例，让 SubLayer 的贡献在总量中占比更小
→ 每层的"扰动"更小 → 训练更稳定
```

### 6.5 各方案对比

| 方案 | Norm 位置 | 训练稳定性 | 最终性能 | 代表模型 |
|------|-----------|-----------|---------|---------|
| Post-Norm | 子层之后 | 差（需 warmup） | 略好 | 原始 Transformer, BERT |
| **Pre-Norm** | **子层之前** | **好** | **好** | **LLaMA, DeepSeek, Qwen** |
| Sandwich | 子层前后 | 更好 | 好 | CogView |
| DeepNorm | 子层之后 + α缩放 | 极好 | 好 | 1000层模型 |

---

## 7. 不同归一化范围的对比

除了位置（前/后），归一化的"范围"（沿哪些维度计算统计量）也有多种：

### 7.1 全景对比

```
输入张量: (B, N, H) = (batch_size, seq_len, hidden_size)

┌─────────────────────────────────────────────────┐
│              Batch Norm                          │
│  沿 B 维度归一化（同一特征维度，跨所有样本/token） │
│  统计量: 每个特征维度一个 μ 和 σ                 │
│  参数数: H 个 γ + H 个 β                        │
│  适合: CNN（batch 大，空间位置无关）               │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              Layer Norm                          │
│  沿 H 维度归一化（同一 token，跨所有特征维度）    │
│  统计量: 每个 token 一个 μ 和 σ                  │
│  参数数: H 个 γ + H 个 β                        │
│  适合: Transformer / LLM（变长序列，batch 可能为1）│
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              RMS Norm                            │
│  沿 H 维度归一化（同 Layer Norm，但不减均值）      │
│  统计量: 每个 token 一个 RMS                     │
│  参数数: H 个 γ                                  │
│  适合: 现代 LLM（更快，效果相当）                  │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              Instance Norm                       │
│  沿 H 的空间维度归一化（同一样本的同一通道内）     │
│  适合: 风格迁移                                   │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│              Group Norm                          │
│  将 H 分成 G 组，每组内做 Layer Norm              │
│  适合: 小 batch CNN                               │
└─────────────────────────────────────────────────┘
```

### 7.2 用一个具体例子理解"范围"

```
数据：2 个样本，每个样本 3 个 token，hidden_size = 4

样本 0:
  token 0: [1.0,  2.0,  3.0,  4.0]
  token 1: [5.0,  6.0,  7.0,  8.0]
  token 2: [9.0, 10.0, 11.0, 12.0]

样本 1:
  token 0: [2.0,  3.0,  1.0,  5.0]
  token 1: [4.0,  1.0,  6.0,  3.0]
  token 2: [8.0,  7.0,  2.0,  9.0]


Batch Norm（沿第 0 维，跨所有样本的同一位置、同一特征）:
  特征维度 0 的所有值: [1, 5, 9, 2, 4, 8]
  μ₀ = (1+5+9+2+4+8)/6 = 4.83
  → 统计量来自不同样本的混合，batch_size=1 时退化

Layer Norm（沿最后一维，单个 token 内）:
  样本0 token0: [1, 2, 3, 4]
  μ = (1+2+3+4)/4 = 2.5
  σ² = [(1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²] / 4 = 1.25
  → 统计量完全来自这一个 token 的 4 个值
  → 与 batch 无关，与其他 token 无关

RMS Norm（沿最后一维，单个 token 内，不减均值）:
  样本0 token0: [1, 2, 3, 4]
  RMS = √((1²+2²+3²+4²)/4) = √(30/4) = √7.5 ≈ 2.739
  → 只需计算平方的均值，再开方
```

---

## 8. 数值例子：从头算一遍

### 8.1 LayerNorm 完整计算

```
输入: x = [1.0, -2.0, 3.0, 0.0, -1.0]   H=5
γ = [1.0, 1.0, 1.0, 1.0, 1.0]           （初始化全1）
β = [0.0, 0.0, 0.0, 0.0, 0.0]           （初始化全0）
ε = 1e-5

步骤 1: 均值
  μ = (1 + (-2) + 3 + 0 + (-1)) / 5 = 1/5 = 0.2

步骤 2: 方差
  σ² = [(1-0.2)² + (-2-0.2)² + (3-0.2)² + (0-0.2)² + (-1-0.2)²] / 5
     = [0.64 + 4.84 + 7.84 + 0.04 + 1.44] / 5
     = 14.8 / 5 = 2.96

步骤 3: 归一化
  √(σ² + ε) = √2.96001 ≈ 1.7205

  x̂ = (x - μ) / √(σ² + ε)
  x̂₀ = (1.0 - 0.2) / 1.7205 = 0.8 / 1.7205 ≈ 0.465
  x̂₁ = (-2.0 - 0.2) / 1.7205 = -2.2 / 1.7205 ≈ -1.279
  x̂₂ = (3.0 - 0.2) / 1.7205 = 2.8 / 1.7205 ≈ 1.627
  x̂₃ = (0.0 - 0.2) / 1.7205 = -0.2 / 1.7205 ≈ -0.116
  x̂₄ = (-1.0 - 0.2) / 1.7205 = -1.2 / 1.7205 ≈ -0.697

  x̂ ≈ [0.465, -1.279, 1.627, -0.116, -0.697]

步骤 4: 仿射变换
  y = γ ⊙ x̂ + β = 1.0 × x̂ + 0.0 = x̂   （初始化时不变）

验证:
  mean(x̂) ≈ 0.465 - 1.279 + 1.627 - 0.116 - 0.697 = 0.000 ✓
  var(x̂) ≈ (0.465² + 1.279² + 1.627² + 0.116² + 0.697²)/5 ≈ 1.000 ✓
```

### 8.2 RMSNorm 完整计算

```
输入: x = [1.0, -2.0, 3.0, 0.0, -1.0]   H=5
w = [1.0, 1.0, 1.0, 1.0, 1.0]           （可学习权重，初始化全1）
ε = 1e-5

步骤 1: 计算平方均值
  mean(x²) = (1² + (-2)² + 3² + 0² + (-1)²) / 5
           = (1 + 4 + 9 + 0 + 1) / 5 = 15/5 = 3.0

步骤 2: 计算 RMS
  RMS = √(3.0 + 1e-5) ≈ 1.7321

步骤 3: 归一化
  x̂ = x / RMS
  x̂₀ = 1.0 / 1.7321 ≈ 0.577
  x̂₁ = -2.0 / 1.7321 ≈ -1.155
  x̂₂ = 3.0 / 1.7321 ≈ 1.732
  x̂₃ = 0.0 / 1.7321 ≈ 0.000
  x̂₄ = -1.0 / 1.7321 ≈ -0.577

  x̂ ≈ [0.577, -1.155, 1.732, 0.000, -0.577]

步骤 4: 乘以可学习权重
  y = w ⊙ x̂ = x̂   （初始化时不变）

注意：
  mean(x̂) = (0.577 - 1.155 + 1.732 + 0 - 0.577) / 5 = 0.115 ≠ 0
  → RMSNorm 不保证均值为 0（与 LayerNorm 的差异）
  → 但 RMS(x̂) = √(mean(x̂²)) ≈ 1.0 ✓（均方根被归一化了）
```

### 8.3 两者输出对比

```
输入:       [1.0,   -2.0,    3.0,    0.0,   -1.0]
LayerNorm:  [0.465, -1.279,  1.627, -0.116, -0.697]   均值=0, 方差=1
RMSNorm:    [0.577, -1.155,  1.732,  0.000, -0.577]   RMS=1

差异不大，但 RMSNorm 少了一次归约计算（省 mean），
在 4096 维向量上这个差距被放大，RMSNorm 更快。
```

---

## 9. vLLM 中的实现

### 9.1 Python 层面

```python
# vllm/model_executor/layers/layernorm.py

class RMSNorm(CustomOp):
    """Root mean square normalization.
    Computes x -> w * x / sqrt(E[x^2] + eps)
    """

    def __init__(self, hidden_size, eps=1e-6):
        self.weight = nn.Parameter(torch.ones(hidden_size))  # γ，初始化全1
        self.variance_epsilon = eps

    def forward_cuda(self, x, residual=None):
        if residual is not None:
            # 融合操作: residual_add + norm 一步完成
            return fused_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
        else:
            return rms_norm(x, self.weight, self.variance_epsilon)
```

### 9.2 Fused Add + RMSNorm（融合操作）

```python
# 非融合（朴素实现，两步）：
residual = x + residual           # 写入 HBM
output = rms_norm(residual)       # 读 HBM，写 HBM
# → 3 次 HBM 访问

# 融合（vLLM 实现，一步）：
output, residual = fused_add_rms_norm(x, residual, weight, eps)
# → 2 次 HBM 访问（节省 ~33% 内存带宽）

# CUDA 内核中：
# 1. 从 HBM 读入 x 和 residual
# 2. 在寄存器/SRAM 中: residual_out = x + residual
# 3. 计算 RMS = √(mean(residual_out²) + ε)
# 4. norm_out = residual_out / RMS × weight
# 5. 写出 norm_out 和 residual_out
```

### 9.3 在 Transformer 层中的位置

```python
# vllm/model_executor/models/llama.py（Pre-Norm 结构）

class LlamaDecoderLayer:
    def forward(self, positions, hidden_states, residual):

        # ── Attention 子块 ──
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
            # 第一层：归一化后送入 Attention
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            # 非第一层：融合 (residual + hidden_states) 后归一化
            # residual 被更新为 FFN_prev_output + prev_residual
            # hidden_states 被更新为 RMSNorm(新 residual)

        hidden_states = self.self_attn(positions, hidden_states)
        # Attention 的输入是归一化后的 hidden_states

        # ── FFN 子块 ──
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        # 再次融合: (Attention输出 + residual) → RMSNorm

        hidden_states = self.mlp(hidden_states)
        # FFN 的输入也是归一化后的

        return hidden_states, residual
```

### 9.4 归一化出现的所有位置

```
一个 Transformer 层中，RMSNorm 出现 2 次:

    x（上层输出）
    │
    ▼
  ┌──────────────────┐
  │ ① input_layernorm │  ← Attention 之前的归一化
  └────────┬─────────┘
           ▼
      Attention
           │
           ▼
  ┌────────────────────────┐
  │ ② post_attention_layernorm │  ← FFN 之前的归一化
  └────────┬───────────────┘
           ▼
         FFN
           │
           ▼
        输出

整个模型末尾还有 1 次:
  ③ final_norm   ← 最后一层输出到 LM Head 之前的归一化

总计: 32 层 × 2 + 1 = 65 次 RMSNorm（LLaMA-3-8B）
```

---

## 总结

**层归一化**解决的核心问题是：**让深层网络每一层的输入保持在稳定的数值分布**，防止数值爆炸/消失和分布漂移。

```
归一化做了什么:
  输入: [1.0, -2.0, 3.0, 0.0, -1.0]   范围 [-2, 3]，分布不确定
  输出: [0.47, -1.28, 1.63, -0.12, -0.70]  范围 ≈[-1.3, 1.6]，均值≈0，方差≈1

为什么有效:
  1. 稳定数值范围 → 不会爆炸/消失
  2. 稳定输入分布 → 权重学习更稳定
  3. 平滑损失曲面 → 优化更容易收敛
```

**归一化的位置决定了训练稳定性**：
- **Pre-Norm**（现代 LLM 标准）：归一化在子层之前，梯度直通残差分支不经过 Norm → 稳定
- Post-Norm（原始 Transformer）：归一化在残差加法之后，梯度必须穿过 Norm → 较不稳定

**RMSNorm 是当前主流**：去掉减均值步骤，省一次归约计算，效果几乎不变。vLLM 将 residual add + RMSNorm 融合为单个 CUDA kernel，节省 33% 的 HBM 访问，是关键的性能优化之一。
