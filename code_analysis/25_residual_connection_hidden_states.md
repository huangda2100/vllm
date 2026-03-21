# 残差连接与隐态层

## 目录
1. [深度网络面临的问题](#1-深度网络面临的问题)
2. [残差连接原理](#2-残差连接原理)
3. [为什么残差连接有效](#3-为什么残差连接有效)
4. [LayerNorm 与残差的配合](#4-layernorm-与残差的配合)
5. [Pre-Norm vs Post-Norm](#5-pre-norm-vs-post-norm)
6. [隐态层（Hidden States）](#6-隐态层hidden-states)
7. [vLLM 中的具体实现](#7-vllm-中的具体实现)
8. [Fused Residual Add + Norm CUDA 内核](#8-fused-residual-add--norm-cuda-内核)
9. [完整数据流总结](#9-完整数据流总结)

---

## 1. 深度网络面临的问题

### 1.1 梯度消失（Gradient Vanishing）

在反向传播中，每层的梯度通过链式法则逐层传递：

```
∂L/∂x₁ = ∂L/∂xₙ × ∂xₙ/∂xₙ₋₁ × ... × ∂x₂/∂x₁

如果每层梯度 |∂xᵢ₊₁/∂xᵢ| < 1（比如 sigmoid 导数最大 0.25）：
  50层后：梯度 ≈ 0.25⁵⁰ ≈ 10⁻³⁰  → 梯度完全消失，底层参数无法更新
```

对应地，**梯度爆炸**（每层梯度 > 1）会导致参数震荡无法收敛。

### 1.2 退化问题（Degradation）

2015 年前，研究者发现了一个反直觉的现象：

```
20层网络 的测试误差  <  56层网络 的测试误差（CIFAR-10 上）
```

这不是过拟合——**训练误差**也是 56 层更高。更深的网络理论上至少能学到"恒等映射"（identity mapping），结果却比浅网络更差。说明**深层网络很难学习恒等映射**，优化困难是根本原因。

### 1.3 根本问题

深层网络要学的函数 $H(x)$：
- 若理想映射接近恒等（即"基本不变"），网络要把所有层的参数都调整到使输出≈输入，这在高维空间里极难
- 越深，信息流（前向）和梯度流（反向）越容易中断

---

## 2. 残差连接原理

### 2.1 核心思想（He et al., 2015 ResNet）

**不学习目标映射，改为学习残差（residual）**：

```
普通网络：  y = F(x)         F 学习完整映射
残差网络：  y = F(x) + x     F 只需学习残差 ΔF = H(x) - x
```

图示：

```
         x
         │
         ├────────────────────────────┐
         │                            │  skip connection（跳跃连接）
         ▼                            │
    ┌─────────┐                       │
    │ Weight  │                       │
    │ Layer 1 │                       │
    └────┬────┘                       │
         │                            │
    ┌────▼────┐                       │
    │  ReLU   │                       │
    └────┬────┘                       │
         │                            │
    ┌────▼────┐                       │
    │ Weight  │                       │
    │ Layer 2 │  F(x)                 │  x
    └────┬────┘                       │
         │                            │
         └──────────── + ─────────────┘
                       │
                  y = F(x) + x
```

### 2.2 数学表达

设第 $l$ 层的输入为 $x_l$，该"残差块"包含若干变换 $\mathcal{F}$：

$$x_{l+1} = x_l + \mathcal{F}(x_l, \{W_i\})$$

- $x_l$：**残差（residual）**，通过 skip connection 直接传递
- $\mathcal{F}(x_l)$：子网络学习的**增量/残差函数**，理想情况下接近零
- $x_{l+1}$：下一层的输入

若理想映射 $H(x) \approx x$（恒等映射），则：
$$\mathcal{F}(x) = H(x) - x \approx 0$$

**让网络学习"接近零的修正量"比学习"接近恒等的完整映射"容易得多**。

---

## 3. 为什么残差连接有效

### 3.1 梯度高速公路

残差连接在反向传播中创造了一条**梯度直达通道**：

$$\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} \cdot \underbrace{\frac{\partial x_L}{\partial x_l}}_{\text{展开}}$$

对于残差网络，从第 $l$ 层到第 $L$ 层（$L > l$）：

$$x_L = x_l + \sum_{i=l}^{L-1} \mathcal{F}(x_i, W_i)$$

因此：

$$\frac{\partial x_L}{\partial x_l} = 1 + \frac{\partial}{\partial x_l}\sum_{i=l}^{L-1}\mathcal{F}(x_i, W_i)$$

关键：上式中有一个**常数 1**！

```
梯度 = 1（直接通路）+ 各层残差函数的梯度之和

即使各层残差函数梯度很小（接近0），常数项 1 保证了梯度不会完全消失
→ 底层参数始终能收到有效梯度信号
→ 100层、1000层的网络也能正常训练
```

### 3.2 集成学习视角

残差网络可以看作**指数多个浅层网络的集成**。

展开 $n$ 层残差网络，等价于 $2^n$ 条路径：

```
3个残差块（每块可走或不走 skip）：
  路径1: x → F₁ → F₂ → F₃ → y
  路径2: x → skip → F₂ → F₃ → y
  路径3: x → F₁ → skip → F₃ → y
  路径4: x → F₁ → F₂ → skip → y
  路径5: x → skip → skip → F₃ → y
  ...共 2³ = 8 条路径

实验验证：随机移除残差块（像 Dropout 一样），性能下降平滑
→ 不同路径提供冗余，任何单一路径失效不会导致完全失败
```

### 3.3 恒等初始化优势

训练开始时，$\mathcal{F}(x) \approx 0$（权重接近零初始化时）：

$$x_{l+1} = x_l + \underbrace{\mathcal{F}(x_l)}_{\approx 0} \approx x_l$$

整个深层网络初始近似为**恒等映射**。训练从"什么都不做"开始逐渐学习有用的变换，而不是从随机噪声开始，稳定性大幅提升。

---

## 4. LayerNorm 与残差的配合

### 4.1 为何需要 LayerNorm

残差连接解决了梯度消失，但随着网络加深，激活值分布会**漂移**（Internal Covariate Shift）：

```
第 1 层输出：均值≈0，方差≈1
第 10 层输出：均值≈2.3，方差≈15.6   ← 分布已偏移
第 50 层输出：均值≈-8.1，方差≈312   ← 严重偏移，激活函数饱和 （当输入值过大或过小时，激活函数的输出几乎不再变化）
```

**LayerNorm** 在每一层对 hidden state 做归一化，将其拉回标准分布：

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

其中 $\mu = \text{mean}(x)$，$\sigma^2 = \text{var}(x)$，$\gamma, \beta$ 是可学习参数。

### 4.2 RMSNorm（Transformer 中的 LayerNorm 变体）

现代 LLM（LLaMA, DeepSeek, Qwen 等）通常使用 **RMSNorm**，去掉了均值中心化：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{H}\sum_{i=1}^H x_i^2 + \epsilon}} \cdot w$$

- 只用均方根（RMS），无需计算均值
- 计算量更小（省去均值计算），效果相当
- $w$（weight/scale）是可学习的逐维缩放参数

---

## 5. Pre-Norm vs Post-Norm

Transformer 中，LayerNorm 和残差连接有两种组合方式：

### 5.1 Post-Norm（原始 Transformer，2017）

```
x_out = LayerNorm( x + SubLayer(x) )

数据流：
  x ─────────────────────────┐
  │                           │
  ▼                           │
SubLayer(Attn 或 FFN)         │ (skip)
  │                           │
  └────── + ─────────────────┘
           │
           ▼
        LayerNorm
           │
           ▼
         x_out
```

**问题**：
- 残差加法后才 Norm，底层梯度仍可能不稳定
- 训练需要**学习率 warmup**，否则易发散。 warmup = 在训练刚开始时，用“很小的学习率”，然后逐步增大到目标学习率
- 更深的网络（>100层）难以训练

### 5.2 Pre-Norm（现代 LLM 标准，GPT-2 / LLaMA / DeepSeek）

```
x_out = x + SubLayer( LayerNorm(x) )

数据流：
  x ─────────────────────────┐
  │                           │
  ▼                           │
LayerNorm                     │ (skip)
  │                           │
  ▼                           │
SubLayer(Attn 或 FFN)         │
  │                           │
  └────── + ─────────────────┘
           │
           ▼
         x_out
```

**优势**：
- 归一化发生在子层**之前**，子层接收到稳定的输入
- 梯度通过 skip connection 直接回传到底层，**不经过** LayerNorm 的反向传播
- 训练更稳定，无需严格 warmup，支持更深网络
- GPT-2 之后几乎所有 LLM 都采用 Pre-Norm

### 5.3 对比总结

| 特性 | Post-Norm | Pre-Norm |
|------|-----------|----------|
| 归一化位置 | 残差加法**之后** | 子层输入**之前** |
| 训练稳定性 | 较差，需 warmup | 更好，开箱即用 |
| 典型模型 | 原始 BERT、Transformer | GPT-2/3/4、LLaMA、DeepSeek |
| 极深网络 | 困难 | 支持良好 |
| 最终性能 | 略高（收敛后） | 略低但差距小 |

---

## 6. 隐态层（Hidden States）

### 6.1 概念

**Hidden State（隐态层/隐藏状态）** 是神经网络中**中间层的输出向量**，是网络对输入的内部表示。

```
输入 token "猫"
     │
     ▼ Embedding
[0.2, -0.5, 1.3, ..., 0.8]   ← 词嵌入（维度 H=4096）
     │
     ▼ Transformer Layer 1
[0.4, -0.1, 0.9, ..., 1.2]   ← 第1层的 hidden state
     │
     ▼ Transformer Layer 2
[0.7,  0.3, 0.6, ..., 0.5]   ← 第2层的 hidden state
     │                            ...（每层都是一个 hidden state）
     ▼ Transformer Layer 32
[1.1, -0.8, 0.2, ..., 0.9]   ← 第32层的 hidden state
     │
     ▼ LM Head（Linear）
logits: [概率分布 over 词表]
```

### 6.2 Hidden State 的语义

Hidden state 并不是人类可直接理解的信息，而是模型学到的**分布式表示**（distributed representation）：

- **底层（Layer 1-4）**：倾向于编码词法/句法信息（词性、短语结构）
- **中间层（Layer 10-20）**：倾向于编码语义信息（实体、关系、上下文）
- **顶层（Layer 28-32）**：倾向于编码任务相关的高层语义（推理、语用）

实验（来自 Anthropic 等研究）：可以在不同层的 hidden state 上训练探针（probe），预测：
- 第 2 层：词性标注准确率 ~95%
- 第 8 层：命名实体识别准确率 ~90%
- 第 16 层：情感分类准确率 ~92%

### 6.3 Hidden State 的维度

```
shape: [num_tokens, hidden_size]

以 LLaMA-3-8B 为例：
  num_tokens = 当前批次的总 token 数（如 128）
  hidden_size = 4096

每个 token 在每一层都有一个 4096 维的向量，
这个向量就是该 token 在该层的 hidden state。

完整前向传播中的 hidden state 变化：
  输入 token IDs:   [128]     (整数)
  Embedding 输出:   [128, 4096]   ← hidden state（第0层）
  Layer 1 输出:     [128, 4096]   ← hidden state（第1层）
  ...
  Layer 32 输出:    [128, 4096]   ← hidden state（最后层）
  LM Head 输出:     [128, 32000]  ← logits（词表概率）
```

### 6.4 Hidden State 与残差的关系

在 Pre-Norm Transformer 中，每层的 hidden state 更新是**增量式**的：

```python
# 伪代码：每层对 hidden state 的处理

hidden_states = embedding(input_ids)   # 初始化，形状 [N, H]

for layer in transformer_layers:
    # 残差保存当前状态
    residual = hidden_states

    # Attention 子层（只在归一化后的输入上操作）
    hidden_states = layer_norm_1(hidden_states)
    hidden_states = attention(hidden_states)

    # 第一次残差加法：将 attention 的输出叠加到原始状态
    hidden_states = hidden_states + residual   ← 残差连接

    # FFN 子层
    residual = hidden_states
    hidden_states = layer_norm_2(hidden_states)
    hidden_states = feed_forward(hidden_states)

    # 第二次残差加法
    hidden_states = hidden_states + residual   ← 残差连接

# hidden_states 现在是最后一层的表示
logits = lm_head(hidden_states)
```

**核心理解**：每一层的 hidden state 是**前一层 hidden state + 本层学到的增量**，残差连接使信息得以**无损传递**，网络只需要专注于学习"在现有表示上做什么修改"。

---

## 7. vLLM 中的具体实现

### 7.1 LlamaDecoderLayer 的残差模式

```python
# vllm/model_executor/models/llama.py（第 258~347 行）

class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, ...):
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = LlamaAttention(...)
        self.mlp = LlamaMLP(...)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,      # ← 残差从上一层传入
    ) -> tuple[torch.Tensor, torch.Tensor]:

        # ── Attention 子块 ──
        if residual is None:
            # 第一层：没有上层传来的 residual，自己作为 residual
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            # 非第一层：融合执行 (residual + hidden_states)，再 norm
            # 等价于：new_residual = residual + hidden_states
            #          hidden_states = RMSNorm(new_residual)
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
            #                         ↑ 这里 residual 被就地更新为 attn_input

        hidden_states = self.self_attn(
            positions=positions, hidden_states=hidden_states
        )
        # 此时 hidden_states = attn_output（未加残差）
        # residual = attn_input（即本层的输入）

        # ── FFN 子块 ──
        # 再次融合：new_residual = residual + hidden_states
        #           hidden_states = RMSNorm(new_residual)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        hidden_states = self.mlp(hidden_states)

        # 返回 FFN 输出 + 更新后的 residual（供下一层使用）
        return hidden_states, residual
        #                      ↑ 下一层的 residual 是 attn_output + attn_input（即 ffn_input）
```

**关键设计**：`residual` 以参数形式跨层传递，避免在每层内部显式创建中间张量。

### 7.2 LlamaModel 的多层循环

```python
# vllm/model_executor/models/llama.py（第 412~443 行）

def forward(self, input_ids, positions, ...):
    hidden_states = self.get_input_embeddings(input_ids)  # [N, H]
    residual = None   # 第一层的 residual 为 None

    for layer in self.layers:
        hidden_states, residual = layer(positions, hidden_states, residual)
        #                          ↑             ↑
        #                       FFN输出      更新后的残差（=FFN输入）

    # 最后一层：融合残差 + 最终 norm
    hidden_states, _ = self.norm(hidden_states, residual)
    #                    等价于：final = RMSNorm(hidden_states + residual)

    return hidden_states
```

**每层的残差状态变化**（以第 $l$ 层为例）：

```
层输入：
  hidden_states = FFN_{l-1} 的输出   (= x_{l-1,ffn})
  residual      = Attn_{l-1} 输入    (= x_{l-1,attn_in})

执行 input_layernorm(hidden_states, residual)：
  new_residual  = x_{l-1,attn_in} + x_{l-1,ffn}   ← 残差加法
  hidden_states = RMSNorm(new_residual)             ← 归一化
  → 这就是第 l 层 Attention 的输入

执行 Attention：
  hidden_states = Attn_l(RMSNorm(x_l))

执行 post_attention_layernorm(hidden_states, residual)：
  new_residual  = x_l + Attn_l(RMSNorm(x_l))      ← 残差加法
  hidden_states = RMSNorm(new_residual)             ← 归一化
  → 这就是第 l 层 FFN 的输入

执行 FFN：
  hidden_states = FFN_l(RMSNorm(x_l + Attn_l(...)))

返回：
  hidden_states = FFN_l 的输出（未加残差）
  residual      = FFN_l 的输入（已加上一个残差）
```

---

## 8. Fused Residual Add + Norm CUDA 内核

### 8.1 为什么要融合

不融合的朴素实现需要**两次** HBM 读写：

```
Step 1: residual_out = hidden + residual    # 写入 HBM
Step 2: norm_out = RMSNorm(residual_out)    # 读 HBM，写 HBM
```

融合后**一次** HBM 读写：

```
fused_add_rms_norm(hidden, residual):
  1. 从 HBM 读入 hidden 和 residual（各一次）
  2. 在 GPU 寄存器/SRAM 中：
     - residual_out = hidden + residual    （就地更新 residual）
     - 计算 RMS = sqrt(mean(residual_out²) + ε)
     - norm_out = residual_out / RMS × weight  （就地更新 hidden）
  3. 写出 norm_out（hidden）和 residual_out（residual）

内存访问量：减少约 33%（从 3次读+2次写 → 2次读+2次写）
```

### 8.2 Python 接口

```python
# vllm/model_executor/layers/layernorm.py（第 40~58 行）

def fused_add_rms_norm(
    x: torch.Tensor,          # hidden_states（输入/输出）
    residual: torch.Tensor,   # 残差（输入/输出）
    weight: torch.Tensor,     # RMSNorm 的 gamma 权重
    variance_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    就地操作（in-place）：
      x        → 更新为 RMSNorm(x + residual)    即 norm_output
      residual → 更新为 x + residual              即 下一层的 residual
    """
    ops.fused_add_rms_norm(x, residual, weight, variance_epsilon)
    return x, residual   # x 和 residual 的内容已被修改
```

### 8.3 CUDA 内核实现

```cpp
// csrc/layernorm_kernels.cu（第 59~111 行，向量化版本）

template <typename scalar_t, int width>    // width=8：每线程处理8个FP16
__global__ void fused_add_rms_norm_kernel(
    scalar_t* input,      // hidden_states，就地改为 norm_output
    scalar_t* residual,   // 残差，就地改为 hidden + residual
    const scalar_t* weight,
    const float epsilon, const int hidden_size)
{
    __shared__ float s_variance;

    // ── 第一阶段：残差加法 + 计算方差 ──
    float variance = 0.0f;
    for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
        // 读入 hidden 和 residual（向量化，一次读 8 个 FP16）
        _f16Vec<scalar_t, 8> temp = input_v[strided_id];
        temp += residual_v[id];          // 残差加法（向量化）
        variance += temp.sum_squares();  // 累加平方和用于方差
        residual_v[id] = temp;           // 就地写回 residual_out = hidden + residual
    }

    // 块内规约求总平方和（CUB BlockReduce）
    variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);
    if (threadIdx.x == 0) {
        s_variance = rsqrtf(variance / hidden_size + epsilon);  // 1/RMS
    }
    __syncthreads();   // 确保所有线程都能读到 s_variance

    // ── 第二阶段：归一化 ──
    for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
        _f16Vec<scalar_t, 8> temp = residual_v[id];  // 读 residual_out
        temp *= s_variance;    // 除以 RMS
        temp *= weight_v[idx]; // 乘以 gamma 权重
        input_v[strided_id] = temp;  // 就地写回 norm_output
    }
}
```

**关键优化点**：
- `width=8`：将 8 个 FP16 打包为一个 128-bit 向量加载（`LDG.128`），提高内存带宽利用率
- 两阶段设计：第一阶段计算 residual_out 并积累方差，第二阶段用共享内存中的方差做 norm
- 自适应 block size：token 数 < 256 时用 block=1024，否则 block=256

---

## 9. 完整数据流总结

### 9.1 一个 Transformer Block 的完整数据流（Pre-Norm + Fused）

```
┌─────────────────────────────────────────────────────────────────────────┐
│  输入: hidden_states [N, H], residual [N, H]                            │
│                                                                         │
│  ① fused_add_rms_norm(hidden_states, residual)                         │
│     ┌────────────────────────────────────────────┐                      │
│     │ residual  = hidden_states + residual       │  ← 残差加法          │
│     │ hidden_states = RMSNorm(residual)          │  ← 归一化            │
│     └────────────────────────────────────────────┘                      │
│                                                                         │
│  ② Attention(hidden_states)                                             │
│     hidden_states = flash_attn(Q, K, V)          ← 注意力计算           │
│                                                                         │
│  ③ fused_add_rms_norm(hidden_states, residual)                         │
│     ┌────────────────────────────────────────────┐                      │
│     │ residual  = hidden_states + residual       │  ← 残差加法          │
│     │ hidden_states = RMSNorm(residual)          │  ← 归一化            │
│     └────────────────────────────────────────────┘                      │
│                                                                         │
│  ④ FFN(hidden_states)                                                   │
│     hidden_states = down(silu(gate) × up)        ← FFN 计算            │
│                                                                         │
│  输出: hidden_states [N, H], residual [N, H]                            │
│  （hidden_states = FFN输出，residual = FFN输入 = Attn输出 + Attn输入）  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 残差流（Residual Stream）的直觉理解

残差连接构成了一条贯穿全网络的**主干流**（residual stream）：

```
词嵌入
  │
  ▼  ─── Layer 1: 读取并叠加语法信息 ────► residual stream +Δ₁
  │
  ▼  ─── Layer 2: 读取并叠加词汇语义 ────► residual stream +Δ₂
  │
  ▼  ─── Layer 3: 读取并叠加上下文信息 ──► residual stream +Δ₃
  │
  ...
  │
  ▼  ─── Layer N: 读取并叠加任务信息 ────► residual stream +ΔN
  │
  ▼
final hidden state = 词嵌入 + Δ₁ + Δ₂ + ... + ΔN
```

每层从 residual stream 读取信息，通过 Attention 和 FFN 计算出"增量"，将增量叠加回 residual stream。**Attention 是层间通信机制，FFN 是知识存储机制，残差是信息传递的主干道**。

### 9.3 各模型残差实现方式对比

| 模型 | 文件 | 残差方式 | 特点 |
|------|------|---------|------|
| LLaMA-3 | `llama.py` | Pre-Norm + Fused | 残差跨层传递，CUDA 融合 |
| Mixtral | `mixtral.py` | Pre-Norm + Fused | 同 LLaMA，FFN 替换为 MoE |
| DeepSeek-V2 | `deepseek_v2.py` | Pre-Norm + Fused | 同上 + 专家并行 |
| Qwen2 | `qwen2.py` | Pre-Norm + Fused | 同 LLaMA |
| OLMo | `olmo.py` | Pre-Norm + 显式 `+` | 无融合优化 |
| BERT | (旧) | Post-Norm | Norm 在加法后 |

---

## 总结

**残差连接**（Skip Connection）是深度学习的基础技术，解决了两个关键问题：
1. **梯度消失**：通过 $y = F(x) + x$ 引入常数梯度通路 $\frac{\partial y}{\partial x} = 1 + ...$，梯度可以无衰减回传
2. **退化问题**：网络只需学习残差 $F(x) = H(x) - x$（接近零），比学习完整映射容易

**隐态层**（Hidden States）是网络对 token 的内部表示，形状为 `[num_tokens, hidden_size]`，是每一层 Transformer 处理后的中间结果。残差连接使每层的 hidden state = 前一层 hidden state + 本层增量，形成累积叠加的**残差流（Residual Stream）**。

vLLM 将残差加法与 RMSNorm 融合为单个 CUDA kernel（`fused_add_rms_norm`），减少约 33% 的 HBM 访问，并通过将 `residual` 张量跨层传递（而非每层重新创建）进一步降低显存分配开销。
