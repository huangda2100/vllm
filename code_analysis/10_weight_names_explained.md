# Transformer 权重名称全解：o_proj、down_proj 等背后的原理

> 本文以 LLaMA-3-8B 为基准（hidden=4096，heads=32，kv_heads=8，FFN=14336），
> 逐一解释每个权重矩阵的命名含义、数学本质、形状推导和设计原因。

---

## 零、先有一张全图

```
输入 tokens: [B, S]（整数 token ID）
     ↓ embed_tokens（查表）
[B, S, 4096]
     ↓ ×32 个 Decoder Layer
     │
     ├─ input_layernorm.weight       [4096]         Pre-Attention 归一化
     │
     ├─ Attention Block
     │   ├─ q_proj.weight            [4096, 4096]   Query 线性变换
     │   ├─ k_proj.weight            [1024, 4096]   Key 线性变换（GQA）
     │   ├─ v_proj.weight            [1024, 4096]   Value 线性变换（GQA）
     │   └─ o_proj.weight            [4096, 4096]   Output 投影（合并多头）
     │
     ├─ post_attention_layernorm.weight [4096]      Pre-FFN 归一化
     │
     └─ FFN Block（SwiGLU）
         ├─ gate_proj.weight         [14336, 4096]  门控路径
         ├─ up_proj.weight           [14336, 4096]  放大路径
         └─ down_proj.weight         [4096, 14336]  压缩回 hidden
     ↓
model.norm.weight                    [4096]         最终归一化
     ↓
lm_head.weight                       [32000, 4096]  词汇表投影（与 embed 共享）
```

---

## 一、Attention 模块的 4 个投影

### 1.1 `q_proj`（Query Projection，查询投影）

**数学表示**：$Q = X W_Q^T$

**形状**：`[num_heads × head_dim, hidden_size]` = `[4096, 4096]`（LLaMA-3-8B）

**命名来源**："Query"——查询。在注意力机制中，Q 是"提问者"，用来和所有 Key 比较相似度，问的是：

> "在当前 token 的视角下，序列里哪些位置最相关？"

**内部结构**（TP 切分视角）：
```
q_proj.weight: [4096, 4096]
实际上是 32 个头的投影堆叠：
  Head 0：weight[0:128,   :]  → 生成 head_0 的 Q 向量
  Head 1：weight[128:256, :]  → 生成 head_1 的 Q 向量
  ...
  Head 31：weight[3968:4096, :] → 生成 head_31 的 Q 向量

vLLM 的 TP 切分（TP=4）：
  GPU 0：q_proj[0:1024,    :]  → 负责 head 0~7
  GPU 1：q_proj[1024:2048, :]  → 负责 head 8~15
  GPU 2：q_proj[2048:3072, :]  → 负责 head 16~23
  GPU 3：q_proj[3072:4096, :]  → 负责 head 24~31
```

**为什么叫"proj"（projection，投影）？**

在线性代数中，矩阵乘法 $Y = XW$ 是把 X 投影到 W 的列空间。q_proj 的意思是：把 hidden state 投影到"查询空间"，这个空间是模型学习到的、适合做 attention score 计算的表示空间。

---

### 1.2 `k_proj`（Key Projection，键投影）

**数学表示**：$K = X W_K^T$

**形状（GQA）**：`[num_kv_heads × head_dim, hidden_size]` = `[1024, 4096]`

注意 LLaMA-3 使用了 **GQA（Grouped Query Attention）**：
- Q 头数 = 32
- KV 头数 = 8（每组 4 个 Q 头共享 1 个 KV 头）
- 因此 k_proj 只有 8 × 128 = 1024 行，比 q_proj 小 4 倍

**命名来源**："Key"——键。K 是"被检索的索引"，存储在 KV Cache 中，供后续 token 的 Q 来查询相似度：

> "我（这个 token）能提供什么样的信息？"

**GQA 的形状意义**：
```
MHA（标准多头）：
  Q: [B, S, 32*128] = [B, S, 4096]
  K: [B, S, 32*128] = [B, S, 4096]  ← 32组，KV Cache 很大

GQA（LLaMA-3）：
  Q: [B, S, 32*128] = [B, S, 4096]
  K: [B, S, 8*128]  = [B, S, 1024]  ← 只有 8 组！KV Cache 缩小 4 倍

每个 Q 头只与对应组的 K 头做 attention：
  Q_head_0  →  K_head_0 (同一组)
  Q_head_1  →  K_head_0 (同一组)
  Q_head_2  →  K_head_0 (同一组)
  Q_head_3  →  K_head_0 (同一组)
  Q_head_4  →  K_head_1 (下一组)
  ...
```

---

### 1.3 `v_proj`（Value Projection，值投影）

**数学表示**：$V = X W_V^T$

**形状（GQA）**：`[num_kv_heads × head_dim, hidden_size]` = `[1024, 4096]`（与 k_proj 相同）

**命名来源**："Value"——值。V 是"实际内容"，attention score 决定从哪里取，V 决定取出什么：

> "如果你关注我，你能从我这里获得什么信息？"

**Q/K/V 三者关系类比**：

```
图书馆检索系统：
  你的检索词 → Q（你想找什么？）
  书的索引标签 → K（这本书是关于什么的？）
  书的正文内容 → V（这本书实际包含什么？）

  注意力分数 = softmax(Q × K^T / √d)  ← 检索词与索引的匹配程度
  输出 = 注意力分数 × V               ← 按匹配程度加权获取内容
```

**为什么 K 和 V 都是从同一个 X 生成？**

因为注意力是自注意力（Self-Attention）：每个位置既是"提问者"（Q），也是"被提问者"（K/V）。K 和 V 都源自同一个 hidden state X，只是通过不同的投影矩阵变换到不同的表示空间。

---

### 1.4 `o_proj`（Output Projection，输出投影）⭐

**数学表示**：$\text{Output} = \text{concat}(\text{head}_0, ..., \text{head}_{n-1}) W_O^T$

**形状**：`[hidden_size, num_heads × head_dim]` = `[4096, 4096]`

这是最容易被忽视但至关重要的一个矩阵，理解它需要先理解多头注意力的数据流：

```
输入 X: [B, S, 4096]
  ↓ q/k/v_proj
Q: [B, S, 32, 128]  K: [B, S, 8, 128]  V: [B, S, 8, 128]
  ↓ 注意力计算（每个头独立）
head_0_out:  [B, S, 128]
head_1_out:  [B, S, 128]
...
head_31_out: [B, S, 128]
  ↓ concat（拼接所有头）
concat: [B, S, 32×128] = [B, S, 4096]
  ↓ o_proj（这就是 o_proj 的输入！）
output: [B, S, 4096]
```

**o_proj 做什么？**

每个注意力头在独立的"子空间"里学习不同的模式：
- Head 0：可能专注于语法依赖（主谓关系）
- Head 7：可能专注于局部相邻词
- Head 15：可能专注于远程指代关系

这 32 个头的 128 维输出**直接拼接后的 4096 维向量**并不适合直接作为下一层的输入——不同子空间的特征需要"融合"和"重新组合"。`o_proj` 就是这个**融合器**：

```
o_proj.weight: [4096, 4096]

可以理解为 32 块列分区：
  o_proj[:, 0:128]    ← 对 head_0 输出的线性变换
  o_proj[:, 128:256]  ← 对 head_1 输出的线性变换
  ...
  o_proj[:, 3968:4096] ← 对 head_31 输出的线性变换

最终输出 = Σ (head_i_out × o_proj[:, head_i_slice])
        = 所有头的输出在统一空间的加权融合
```

**关键洞察**：没有 o_proj，多头注意力退化为"多个独立的单头注意力拼接"，头与头之间的交互信息无法传递。o_proj 是多头注意力中头间信息交互的唯一通道。

**TP 的切分方式（行并行）**：

```
o_proj 是"行并行"（Row Parallel）：
  GPU 0 持有：o_proj[:, 0:1024]     ← 负责接收 head 0~7 的输出
  GPU 1 持有：o_proj[:, 1024:2048]  ← 负责接收 head 8~15 的输出
  GPU 2 持有：o_proj[:, 2048:3072]  ← 负责接收 head 16~23 的输出
  GPU 3 持有：o_proj[:, 3072:4096]  ← 负责接收 head 24~31 的输出

每张 GPU 得到部分输出（partial sum），最后 AllReduce 求和
→ 这与 q/k/v_proj 的"列并行"配合，形成一个完整的 TP 无需额外通信
```

---

## 二、FFN 模块的 3 个投影（SwiGLU）

LLaMA 使用 **SwiGLU** 激活函数，比 ReLU 或 GELU 效果更好。理解 gate/up/down 需要先理解它的设计。

### 2.1 经典 FFN vs SwiGLU FFN

```
经典 FFN（BERT/GPT-2，使用 GELU）：
  h = GELU(X × W₁)    [B,S,H] → [B,S,4H]
  Y = h × W₂          [B,S,4H] → [B,S,H]

SwiGLU FFN（LLaMA，Mistral，Gemma）：
  gate = SiLU(X × W_gate)    [B,S,H] → [B,S,4H]    ← gate_proj
  up   = X × W_up            [B,S,H] → [B,S,4H]    ← up_proj
  h    = gate ⊙ up           [B,S,4H]，逐元素乘法（Hadamard）
  Y    = h × W_down          [B,S,4H] → [B,S,H]    ← down_proj
```

**SwiGLU 比经典 FFN 多一个矩阵（gate_proj），但中间维度缩小到 8/3×H ≈ 2.67H（而非 4H），总参数量大致相当。**

---

### 2.2 `gate_proj`（门控投影）

**数学表示**：$g = \text{SiLU}(X W_{\text{gate}}^T)$

**形状**：`[ffn_size, hidden_size]` = `[14336, 4096]`（LLaMA-3-8B）

**命名来源**："Gate"——门控。受到 LSTM/GRU 门控机制的启发，gate_proj 产生一个**控制信号**，决定 up_proj 的每个维度"开放多少"：

```
SiLU 函数（Sigmoid Linear Unit）：
  SiLU(x) = x × σ(x) = x × 1/(1 + e^{-x})

特性：
  x 很正 → SiLU(x) ≈ x（完全开放）
  x ≈ 0  → SiLU(x) ≈ 0（关闭）
  x 很负 → SiLU(x) ≈ 0（但不完全，有小负值）

→ 与 sigmoid gate 不同，SiLU 可以传递负梯度，训练更稳定
```

**列并行（Column Parallel）**——TP 切分方式：

```
gate_proj: [14336, 4096]  → TP=4 时每卡持有 [3584, 4096]
（输出维度切分：每张 GPU 负责 FFN 中间维度的 1/4）
```

---

### 2.3 `up_proj`（上升投影，也叫 Value 投影）

**数学表示**：$u = X W_{\text{up}}^T$

**形状**：`[ffn_size, hidden_size]` = `[14336, 4096]`（与 gate_proj 完全相同）

**命名来源**："Up"——向上升维。把 hidden_size（4096）投影到更大的 FFN 维度（14336），在高维空间进行特征变换。

**与 gate_proj 的关系**：

```
gate 和 up 是 FFN 中的两条"平行路径"：
  gate 路径：学习"哪些维度是重要的"（产生软掩码）
  up   路径：学习"这些重要维度上是什么内容"（产生内容向量）

Hadamard 乘法（⊙）：
  h = SiLU(gate) ⊙ up

直觉：up 就像一个"内容库"，gate 决定从内容库里取哪些内容、取多少
     → 这就是 Gated Linear Unit（GLU）的核心思想
```

**为什么两个矩阵形状完全相同？**

这是 GLU（Gated Linear Unit，Guo et al. 2019）的要求：门控信号和内容信号必须形状一致才能做逐元素乘法。

在实际 vLLM 实现中，为了 TP 效率，gate_proj 和 up_proj 被**合并成一个矩阵**：

```python
# vllm/model_executor/models/llama.py
# 不是两个分开的矩阵，而是一个合并的 gate_up_proj：
self.gate_up_proj = MergedColumnParallelLinear(
    hidden_size,
    [intermediate_size] * 2,  # 两份 14336
    ...
)
# 形状：[2 × 14336, 4096] = [28672, 4096]
# 好处：一次 GEMM 完成两个投影，比两次分开调用更高效
```

---

### 2.4 `down_proj`（下降投影）⭐

**数学表示**：$Y = h W_{\text{down}}^T = (\text{SiLU}(XW_{\text{gate}}) \odot XW_{\text{up}}) W_{\text{down}}^T$

**形状**：`[hidden_size, ffn_size]` = `[4096, 14336]`（注意是 gate/up_proj 的转置形状）

**命名来源**："Down"——向下降维。把 FFN 内部的高维表示（14336 维）压缩回 hidden_size（4096 维），才能与残差连接相加。

**它做什么？**

```
FFN 的完整功能示意：

hidden: [B, S, 4096]
  ↓ gate_proj + up_proj（升维，在高维空间提取特征）
intermediate: [B, S, 14336]
  ↓ down_proj（降维，提炼精华返回 residual stream）
output: [B, S, 4096]
  ↓ +残差
new_hidden: [B, S, 4096]
```

**down_proj 的深层理解**（键值记忆观点）：

研究（Geva et al. 2021，"Transformer Feed-Forward Layers Are Key-Value Memories"）指出：

```
up_proj（或 gate_proj）的每一行 = 一个"键"向量
down_proj 的每一列 = 一个"值"向量

前向传播过程：
  1. 输入 X 与所有"键"做点积（gate/up_proj）→ 激活分数
  2. 高激活分数选出对应的"值"（down_proj 对应列）
  3. 加权求和后输出

直觉类比：
  gate_proj / up_proj ≈ 匹配"当前语境触发了哪些知识点"
  down_proj ≈ "这些知识点对应的具体内容是什么"

  例如："法国的首都是___" → 触发某些神经元 → down_proj 输出"巴黎"相关的方向
```

**行并行（Row Parallel）**——TP 切分方式：

```
down_proj: [4096, 14336]

TP=4 时，列维度切分：
  GPU 0：down_proj[:, 0:3584]       ← 接收 intermediate 的 0~3583 维
  GPU 1：down_proj[:, 3584:7168]    ← 接收 intermediate 的 3584~7167 维
  GPU 2：down_proj[:, 7168:10752]   ← 接收 intermediate 的 7168~10751 维
  GPU 3：down_proj[:, 10752:14336]  ← 接收 intermediate 的 10752~14335 维

每张 GPU 得到部分和 [B, S, 4096]，AllReduce 后得到完整输出
→ 与 gate/up_proj 的列并行配合，down_proj 的行并行收尾，一次 AllReduce 完成
```

---

## 三、归一化权重

### 3.1 `input_layernorm.weight`（前注意力归一化）

**形状**：`[hidden_size]` = `[4096]`

**数学表示**：
$$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \times w, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum_{i=1}^{d} x_i^2}$$

**命名来源**："Input Layer Norm"——输入层归一化（施加在 Attention 输入前）。

**为什么没有 bias（偏置）？**

标准 LayerNorm 有 weight（γ）和 bias（β）：
```
LayerNorm: y = (x - mean) / std × γ + β
RMSNorm:   y = x / RMS(x) × w
```

LLaMA 使用 **RMSNorm**（Root Mean Square Norm，Zhang & Sennrich 2019），去掉了均值中心化和 bias：

```
理由：
  1. 计算快：RMS 比 LayerNorm 省一次均值计算，快 7%~64%
  2. 效果相当：实验表明 bias 和均值中心化对语言模型帮助有限
  3. 参数量少：每层少一个 [hidden_size] 的 bias 向量
```

**初始值**：全 1.0（不缩放）

**训练后典型值**：
```
weight 在 [0.8, 1.5] 范围内波动
> 1.0 的维度：模型认为"这个维度很重要，放大它"
< 1.0 的维度：模型认为"这个维度噪声多，抑制它"
```

---

### 3.2 `post_attention_layernorm.weight`（前 FFN 归一化）

**形状**：`[hidden_size]` = `[4096]`

与 input_layernorm 完全相同结构，只是位置在 Attention 之后、FFN 之前。

**LLaMA 使用 Pre-Norm 而非 Post-Norm**：

```
Pre-Norm（LLaMA，GPT-3）：             Post-Norm（原始 Transformer）：
  x → Norm → Attention → + x            x → Attention → + x → Norm
  ↓                                      ↓
  x → Norm → FFN → + x                  x → FFN → + x → Norm

Pre-Norm 的优势：
  1. 梯度直接从残差路径回传（绕过 Norm 层），不会消失
  2. 训练更稳定，可以不需要 warm-up
  3. 深层网络（>100层）几乎必须用 Pre-Norm
```

---

### 3.3 `model.norm.weight`（最终归一化）

**形状**：`[hidden_size]` = `[4096]`

在所有 32 个 Decoder Layer 之后，输出到 lm_head 之前的最后一次归一化。

---

## 四、Embedding 相关权重

### 4.1 `embed_tokens.weight`（词嵌入）

**形状**：`[vocab_size, hidden_size]` = `[32000, 4096]`

**数学**：将 token ID（整数）映射为连续向量：
$$h_0 = E[token\_id, :] \in \mathbb{R}^{4096}$$

这是一个简单的查表操作（`torch.embedding`），没有矩阵乘法。

```python
# 等价实现：
h = embed_tokens.weight[token_ids, :]
# 比 one-hot × weight 快得多：
# one-hot: [B, S, 32000] × [32000, 4096] → 稀疏矩阵乘法，低效
# 查表:    直接索引       → O(1) 内存访问，高效
```

---

### 4.2 `lm_head.weight`（语言模型头）

**形状**：`[vocab_size, hidden_size]` = `[32000, 4096]`

**作用**：将最后一层的 hidden state 投影回词汇表空间，生成 logits：
$$\text{logits} = h_{\text{final}} W_{\text{lm\_head}}^T \in \mathbb{R}^{32000}$$

然后 softmax 得到下一个 token 的概率分布。

**Weight Tying（权重绑定）**：

大多数语言模型（LLaMA 包含）中，`lm_head.weight` 与 `embed_tokens.weight` **共享同一份参数**：

```python
# vllm/model_executor/models/llama.py
if not self.config.tie_word_embeddings:
    self.lm_head = ParallelLMHead(...)
else:
    # 直接引用 embed_tokens 的权重，不额外分配内存
    self.lm_head = self.model.embed_tokens
```

**为什么 weight tying 有效？**

```
直觉：
  embed_tokens: token_id → 语义向量（编码器视角）
  lm_head:      语义向量 → token_id（解码器视角）

  这两个变换的"语义空间"是相同的。
  例如："king"的嵌入向量方向，应该与"什么词汇的方向最接近'king'"是一致的。

数学视角：
  lm_head 的输出 logit_i = h · embed_tokens[i]
  = 当前 hidden state 与 token_i 的嵌入向量的点积（余弦相似度 × 模）

  → 模型学会让 hidden state 的方向"指向"它认为最可能的下一个 token 的嵌入方向

优势：
  1. 节省内存：32000 × 4096 × 2 bytes = 256 MB
  2. 训练信号共享：lm_head 的梯度同时更新 embed_tokens，嵌入质量更好
```

---

## 五、vLLM 中的额外变量名

### 5.1 vLLM 合并后的权重名（节省 GEMM 调用）

vLLM 为了推理效率，将原始 HuggingFace 权重在加载时合并：

```python
# HuggingFace 原始格式：
"model.layers.0.self_attn.q_proj.weight"  # [4096, 4096]
"model.layers.0.self_attn.k_proj.weight"  # [1024, 4096]
"model.layers.0.self_attn.v_proj.weight"  # [1024, 4096]

# vLLM 合并为：
"model.layers.0.self_attn.qkv_proj.weight"  # [4096+1024+1024, 4096] = [6144, 4096]
# 一次 GEMM 同时算出 Q、K、V
```

```python
# HuggingFace：
"model.layers.0.mlp.gate_proj.weight"  # [14336, 4096]
"model.layers.0.mlp.up_proj.weight"    # [14336, 4096]

# vLLM 合并为：
"model.layers.0.mlp.gate_up_proj.weight"  # [28672, 4096]
# 一次 GEMM 同时算出 gate 和 up
```

这对应 `vllm/model_executor/models/llama.py` 中的 `stacked_params_mapping`：

```python
stacked_params_mapping = [
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
]
```

---

### 5.2 RoPE 相关：`inv_freq` 和 `cos_cached`/`sin_cached`

这些不是可训练权重，而是**预计算的位置编码缓冲区**：

```python
# RoPE（Rotary Position Embedding）的频率缓冲区
inv_freq: [head_dim // 2]  # 如 [64]，值为 1/10000^(2i/d)

# 预计算好的 cos/sin 值缓存
cos_cached: [max_seq_len, head_dim]  # 如 [4096, 128]
sin_cached: [max_seq_len, head_dim]

# 初始值：
inv_freq[i] = 1 / (10000 ** (2i / head_dim))
# 例如 head_dim=128：
# inv_freq[0] = 1.0
# inv_freq[1] = 0.9306
# inv_freq[2] = 0.8660
# ...
# inv_freq[63] = 1e-4（极小值，对应最慢的旋转频率）
```

**为什么 RoPE 没有可训练权重？**

RoPE 是数学确定的旋转变换，不需要学习。它通过给 Q 和 K 向量旋转不同角度（角度取决于 token 位置），让点积 $Q \cdot K$ 自然包含相对位置信息。

---

### 5.3 MoE 模型额外权重（DeepSeek-V3 / Mixtral）

DeepSeek-V3 的 MoE 层额外包含：

```
# 路由器
model.layers.N.mlp.gate.weight         [num_experts, hidden_size]
# 例如 DeepSeek-V3：[256, 7168]
# 计算每个 token 路由到哪些专家的 logit

# 每个专家独立的 FFN（共 256 个）
model.layers.N.mlp.experts.0.gate_proj.weight  [expert_ffn_size, hidden_size]
model.layers.N.mlp.experts.0.up_proj.weight    [expert_ffn_size, hidden_size]
model.layers.N.mlp.experts.0.down_proj.weight  [hidden_size, expert_ffn_size]
...
model.layers.N.mlp.experts.255.down_proj.weight

# 共享专家（每个 token 必经）
model.layers.N.mlp.shared_experts.gate_proj.weight
model.layers.N.mlp.shared_experts.up_proj.weight
model.layers.N.mlp.shared_experts.down_proj.weight
```

**MoE 路由过程**：

```python
# 简化示意
def moe_forward(x, gate_weight, experts):
    # Step 1: 路由计算
    router_logits = x @ gate_weight.T         # [B, S, num_experts]
    router_probs = softmax(router_logits)
    top_k_probs, top_k_idx = router_probs.topk(k=8)  # 选 top-8 专家

    # Step 2: Dispatch（分发 token 到对应专家）
    # ...（EP AllToAll 通信）

    # Step 3: 每个专家独立计算 FFN
    for expert_id, expert in enumerate(experts):
        # 只计算路由到自己的 token
        out = expert(tokens_for_this_expert)

    # Step 4: Combine（汇集结果）
    # ...（EP AllToAll 通信）

    # Step 5: 加权求和（按 router_probs）
    output = sum(top_k_probs[i] * expert_outputs[i] for i in range(8))
```

---

## 六、为什么 LLaMA 没有 bias？

绝大多数 LLaMA 权重都**没有 bias 项**（`bias=False`）。

```python
# vllm/model_executor/models/llama.py
self.q_proj = QKVParallelLinear(hidden_size, ..., bias=False)
self.o_proj = RowParallelLinear(hidden_size, ..., bias=False)
self.gate_up_proj = MergedColumnParallelLinear(hidden_size, ..., bias=False)
self.down_proj = RowParallelLinear(hidden_size, ..., bias=False)
```

**原因**：

```
理由 1：Pre-Norm 使 bias 冗余
  RMSNorm 前的 hidden state 已经是零均值（大约），
  Linear 层不需要 bias 来调整均值

理由 2：减少参数量（边际效益低）
  每层的 bias 参数量：
  q_proj bias: [4096]、o_proj bias: [4096]、...
  32层合计 ≈ 32 × 5 × 4096 = 655,360 个 bias 参数
  占总参数 7B 的 0.01%，可忽略不计

理由 3：TP 更简单
  bias 是"每卡都要有完整副本"还是"按 TP 切分"？
  去掉 bias 彻底消除这个麻烦

  实验结果（Chowdhery et al. 2022）：
  无 bias 的大模型 vs 有 bias 的，困惑度几乎没有差异
```

---

## 七、完整形状与参数量速查表（LLaMA-3-8B）

| 权重名 | 形状 | 参数量 | 作用 | TP 切分方式 |
|--------|------|--------|------|------------|
| embed_tokens.weight | [32000, 4096] | 131M | token → 向量 | 词汇维度切分 |
| q_proj.weight | [4096, 4096] | 16.8M | hidden → Q | 列并行（Q头切分）|
| k_proj.weight | [1024, 4096] | 4.2M | hidden → K | 列并行（KV头切分）|
| v_proj.weight | [1024, 4096] | 4.2M | hidden → V | 列并行（KV头切分）|
| o_proj.weight | [4096, 4096] | 16.8M | concat(heads) → hidden | 行并行 |
| gate_proj.weight | [14336, 4096] | 58.7M | hidden → gate（升维）| 列并行 |
| up_proj.weight | [14336, 4096] | 58.7M | hidden → up（升维）| 列并行 |
| down_proj.weight | [4096, 14336] | 58.7M | intermediate → hidden（降维）| 行并行 |
| input_layernorm.weight | [4096] | 4K | Attention 前归一化 | 广播（不切分）|
| post_attn_layernorm.weight | [4096] | 4K | FFN 前归一化 | 广播（不切分）|
| model.norm.weight | [4096] | 4K | 最终归一化 | 广播（不切分）|
| lm_head.weight | [32000, 4096] | 131M | hidden → logits | 词汇维度切分 |

**每层参数量**：
```
Attention：(4096+1024+1024+4096) × 4096 = 42.0M
FFN：      (14336+14336+14336) × 4096   = 176.1M
Norm：     4096 × 2                      ≈ 0
每层合计：≈ 218M
32 层合计：≈ 6.98B
加 Embedding：≈ 8.0B（与名称"8B"对应）
```

---

## 八、一图串联所有权重

```
Token IDs [B, S]
     │ embed_tokens [32000×4096]
     ▼
Hidden [B, S, 4096] ──────────────────────────────── Residual Stream
     │                                                      ↑ +
     │ input_layernorm [4096] (Pre-Attention Norm)          │
     ▼                                                      │
     │ q_proj [4096×4096]   K/V Cache                      │
     │ k_proj [1024×4096] → ↓                              │
     │ v_proj [1024×4096] → KV Store                       │
     │                                                      │
     │ ← Flash Attention（QK^T V）→                        │
     │                                                      │
     │ o_proj [4096×4096]  ─────────────────────────────→ +
     │
Hidden [B, S, 4096] ──────────────────────────────── Residual Stream
     │                                                      ↑ +
     │ post_attention_layernorm [4096] (Pre-FFN Norm)       │
     ▼                                                      │
     │ gate_proj [14336×4096] → SiLU(gate)                 │
     │                              ↘⊙                     │
     │ up_proj  [14336×4096] ──────→ h [14336]             │
     │                                  │                   │
     │ down_proj [4096×14336] ←────────┘                   │
     │                                                      │
     └──────────────────────────────────────────────────→ +
                                                            │
                              ×32 层                       │
                                                            ↓
                              model.norm [4096]
                                    │
                              lm_head [32000×4096]
                                    │
                              Logits [B, S, 32000]
                                    │
                              Softmax → Next Token 概率
```

---

*参考资料：*
- *[LLaMA 原始论文：Touvron et al. 2023](https://arxiv.org/abs/2302.13971)*
- *[SwiGLU 激活函数：Noam Shazeer 2020](https://arxiv.org/abs/2002.05202)*
- *[RMSNorm：Zhang & Sennrich 2019](https://arxiv.org/abs/1910.07467)*
- *[GQA：Ainslie et al. 2023](https://arxiv.org/abs/2305.13245)*
- *[FFN 作为键值记忆：Geva et al. 2021](https://arxiv.org/abs/2012.14913)*
- *[RoPE：Su et al. 2021](https://arxiv.org/abs/2104.09864)*
- *[Megatron-LM 列/行并行：Shoeybi et al. 2019](https://arxiv.org/abs/1909.08053)*
*更新：2026-03*
