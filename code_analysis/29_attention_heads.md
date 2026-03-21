# 彻底理解模型的"头"（Head）

## 目录
1. [一句话解释：头是什么](#1-一句话解释头是什么)
2. [为什么要分多个头](#2-为什么要分多个头)
3. [从矩阵形状看头的本质](#3-从矩阵形状看头的本质)
4. [头与 KV 的关系：MHA / MQA / GQA](#4-头与-kv-的关系mha--mqa--gqa)
5. [矩阵形状由什么决定](#5-矩阵形状由什么决定)
6. [完整数值例子：手算多头 Attention](#6-完整数值例子手算多头-attention)
7. [vLLM 中的代码实现](#7-vllm-中的代码实现)
8. [LM Head：另一种"头"](#8-lm-head另一种头)

---

## 1. 一句话解释：头是什么

**头（Head）= 把一个大的注意力计算拆成多个小的、独立的注意力计算，每个小计算关注"不同方面"的信息。**

```
单头注意力（1 Head）：
  一次性用 4096 维向量计算注意力
  → 只能关注一种"模式"

多头注意力（32 Heads）：
  把 4096 维拆成 32 份，每份 128 维
  → 32 份各自独立计算注意力
  → 最后拼回 4096 维

好处：32 个头可以同时关注不同"方面"：
  头 1：关注语法关系（主语→谓语）
  头 2：关注共指关系（"他"→指代谁）
  头 3：关注邻近词（"New"→"York"）
  头 4：关注标点结构（句号→下一句开头）
  ...
```

---

## 2. 为什么要分多个头

### 2.1 一个类比

想象你在读一篇文章，想理解某句话。你可以从多个角度同时分析：

```
原句："苹果公司的新产品在发布会上获得了巨大成功"

角度 1（实体关系）：苹果公司 ──拥有──→ 新产品
角度 2（事件关系）：发布会 ──场所──→ 获得成功
角度 3（修饰关系）：巨大 ──修饰──→ 成功
角度 4（主题关系）：苹果公司 ──主语──→ 获得成功
```

如果只有一个注意力头，它只能从一个混合角度做加权。多个头让模型可以**同时从多个角度建立 token 之间的连接**。

### 2.2 核心直觉

```
单头：Q、K 都是 4096 维，点积 Q·K 混合了所有维度的信号
      → 某种特定模式的信号可能被其他维度的噪声淹没

多头：每个头只用 128 维做点积
      → 头 1 的 128 维可以专门学习"语法关系"
      → 头 2 的 128 维可以专门学习"位置邻近性"
      → 互不干扰
```

### 2.3 数学上的等价与不等价

如果没有 softmax，多头注意力和单头在数学上等价（矩阵拼接后做线性变换可以合并）。但 **softmax 是非线性操作，它打破了这种等价性**：

```
单头：softmax( [q₁,q₂,...,q₄₀₉₆] · [k₁,k₂,...,k₄₀₉₆]^T / √4096 )
  → 一个统一的注意力分布

多头：
  头1: softmax( [q₁,...,q₁₂₈] · [k₁,...,k₁₂₈]^T / √128 )
  头2: softmax( [q₁₂₉,...,q₂₅₆] · [k₁₂₉,...,k₂₅₆]^T / √128 )
  ...
  → 32 个不同的注意力分布！

每个头做独立的 softmax → 产生完全不同的注意力模式
→ 这正是多头注意力的"超能力"
```

---

## 3. 从矩阵形状看头的本质

### 3.1 关键参数

以 LLaMA-3-8B 为例：

```
hidden_size (H)    = 4096     模型的隐藏维度
num_heads (n_h)    = 32       Q 头的数量
num_kv_heads (n_kv)= 8        K/V 头的数量（GQA）
head_dim (d)       = H / n_h = 4096 / 32 = 128   每个头的维度

关系：
  hidden_size = num_heads × head_dim
  4096       = 32       × 128
```

### 3.2 权重矩阵形状

```
输入:  hidden_states  形状 (N, 4096)    N = token 数

W_Q 权重矩阵：(4096, 4096)
  ● 输入维度 = hidden_size = 4096
  ● 输出维度 = num_heads × head_dim = 32 × 128 = 4096
  ● 等价于 32 个独立的 (4096, 128) 矩阵并排放置

W_K 权重矩阵：(4096, 1024)
  ● 输入维度 = hidden_size = 4096
  ● 输出维度 = num_kv_heads × head_dim = 8 × 128 = 1024
  ● 等价于 8 个独立的 (4096, 128) 矩阵并排放置

W_V 权重矩阵：(4096, 1024)
  ● 同 W_K

W_O 权重矩阵：(4096, 4096)
  ● 输入维度 = num_heads × head_dim = 4096
  ● 输出维度 = hidden_size = 4096
```

### 3.3 核心操作：投影 + Reshape

```
步骤 1：线性投影（一个大矩阵乘法）
  Q_flat = hidden @ W_Q    形状: (N, 4096) × (4096, 4096) → (N, 4096)
  K_flat = hidden @ W_K    形状: (N, 4096) × (4096, 1024) → (N, 1024)
  V_flat = hidden @ W_V    形状: (N, 4096) × (4096, 1024) → (N, 1024)

步骤 2：Reshape 为多头形式（不涉及任何计算，只是"换个视角看数据"）
  Q = Q_flat.reshape(N, 32, 128)    → (N, 32, 128)   "32个头，每头128维"
  K = K_flat.reshape(N, 8,  128)    → (N, 8,  128)   "8个KV头，每头128维"
  V = V_flat.reshape(N, 8,  128)    → (N, 8,  128)

步骤 3：每个头独立做 Attention
  对头 h = 0, 1, ..., 31:
    score_h = Q[:, h, :] @ K[:, h//4, :]^T / √128    (N, 128) × (128, N) → (N, N)
    weights_h = softmax(score_h)                       (N, N)
    out_h = weights_h @ V[:, h//4, :]                  (N, N) × (N, 128) → (N, 128)

步骤 4：拼接 + 输出投影
  attn_out = concat(out_0, out_1, ..., out_31)         → (N, 32×128) = (N, 4096)
  output = attn_out @ W_O                               (N, 4096) × (4096, 4096) → (N, 4096)
```

**"头"就是 reshape 之后的那个维度**——把一个 4096 维向量重新看成 32 个 128 维的子向量，每个子向量做独立的注意力计算。

---

## 4. 头与 KV 的关系：MHA / MQA / GQA

### 4.1 三种注意力头配置

```
MHA（Multi-Head Attention）：Q/K/V 头数相同
  num_heads = 32,  num_kv_heads = 32
  每个 Q 头配一个独立的 K 头和 V 头
  KV Cache 大小: 32 × 128 = 4096 per token

MQA（Multi-Query Attention）：K/V 只有 1 个头
  num_heads = 32,  num_kv_heads = 1
  32 个 Q 头共享 1 个 K 头和 V 头
  KV Cache 大小: 1 × 128 = 128 per token（减少 32×！）

GQA（Grouped-Query Attention）：K/V 的头数介于 1 和 Q 之间
  num_heads = 32,  num_kv_heads = 8
  每 4 个 Q 头共享 1 组 K/V 头（32/8 = 4）
  KV Cache 大小: 8 × 128 = 1024 per token（减少 4×）
```

### 4.2 图解三种配置

```
MHA (Multi-Head Attention) — 原始 Transformer

  Q头:  Q₀  Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇   ...  Q₃₁
         ↕   ↕   ↕   ↕   ↕   ↕   ↕   ↕         ↕
  K头:  K₀  K₁  K₂  K₃  K₄  K₅  K₆  K₇   ...  K₃₁
  V头:  V₀  V₁  V₂  V₃  V₄  V₅  V₆  V₇   ...  V₃₁

  一对一。KV Cache = 32 头 × 128 维 × 2(K+V) = 8192 per token
  代表模型: 原始 GPT-2, BERT


MQA (Multi-Query Attention) — 所有 Q 头共享同一个 KV

  Q头:  Q₀  Q₁  Q₂  Q₃  Q₄  Q₅  Q₆  Q₇   ...  Q₃₁
         ↘   ↓   ↓   ↓   ↓   ↓   ↓   ↙         ↙
  K头:           ────────  K₀  ────────
  V头:           ────────  V₀  ────────

  多对一。KV Cache = 1 头 × 128 维 × 2 = 256 per token
  代表模型: PaLM, Falcon


GQA (Grouped-Query Attention) — 分组共享（折中方案）

  Q头:  Q₀  Q₁  Q₂  Q₃ │ Q₄  Q₅  Q₆  Q₇ │ ...
         ↘   ↓   ↓   ↙  │  ↘   ↓   ↓   ↙  │
  K头:      K₀          │     K₁          │  ...  K₇
  V头:      V₀          │     V₁          │  ...  V₇

  每 4 个 Q 头共享 1 组 KV。KV Cache = 8 头 × 128 维 × 2 = 2048 per token
  代表模型: LLaMA-3, Mistral, DeepSeek, Qwen2
```

### 4.3 为什么 GQA 是主流

```
KV Cache 内存 = num_kv_heads × head_dim × 2 × seq_len × num_layers × bytes_per_element

LLaMA-3-8B (GQA, 8 KV头):
  8 × 128 × 2 × 4096 × 32 × 2 bytes = 512 MB per sequence

如果用 MHA (32 KV头):
  32 × 128 × 2 × 4096 × 32 × 2 bytes = 2 GB per sequence（4 倍！）

节省 KV Cache → 同时服务更多并发请求 → 吞吐量更高
精度损失呢？实验证明 GQA (8头) 的质量非常接近 MHA (32头)

因为：K/V 编码的是"上下文信息"，不同 Q 头查询同一个上下文，
     共享 K/V 几乎不损失信息（不同 Q 头关注的"问题"不同，
     但"参考资料"可以共用）
```

### 4.4 GQA 的共享机制

```python
# 在 Attention 计算前，将 8 个 KV 头"扩展"为 32 个

K: (N, 8, 128)   → 复制每个头 4 次 → (N, 32, 128)
V: (N, 8, 128)   → 复制每个头 4 次 → (N, 32, 128)

具体: K_expanded[:, 0, :] = K[:, 0, :]   ← Q头 0,1,2,3 共用 KV头 0
      K_expanded[:, 1, :] = K[:, 0, :]
      K_expanded[:, 2, :] = K[:, 0, :]
      K_expanded[:, 3, :] = K[:, 0, :]
      K_expanded[:, 4, :] = K[:, 1, :]   ← Q头 4,5,6,7 共用 KV头 1
      ...

实际实现中不会真正复制（浪费内存），而是通过索引映射：
  kv_head_index = q_head_index // (num_heads // num_kv_heads)
  kv_head_index = q_head_index // 4
```

---

## 5. 矩阵形状由什么决定

### 5.1 形状决定链

```
模型配置（config.json）
  │
  ├── hidden_size = 4096              ← 决定所有权重矩阵的输入维度
  ├── num_attention_heads = 32        ← 决定 Q 的输出维度和注意力并行度
  ├── num_key_value_heads = 8         ← 决定 K/V 的输出维度和 KV Cache 大小
  ├── intermediate_size = 14336       ← 决定 FFN 中间层维度
  └── vocab_size = 32000              ← 决定 Embedding 和 LM Head 维度

运行时参数
  │
  ├── num_tokens (N) = 当前批次的 token 数   ← 决定所有张量的第 0 维
  ├── seq_len = 序列长度                     ← 决定 Attention 矩阵大小
  └── tensor_parallel_size (TP) = GPU 数     ← 决定权重如何分片
```

### 5.2 所有矩阵形状一览表

以 LLaMA-3-8B（TP=1）为例，输入 N 个 token：

```
组件                  权重形状              输入形状       输出形状
──────────────────────────────────────────────────────────────────
Embedding            (32000, 4096)         (N,)          (N, 4096)
W_Q (qkv_proj的Q部分) (4096, 4096)         (N, 4096)     (N, 4096)
W_K (qkv_proj的K部分) (4096, 1024)         (N, 4096)     (N, 1024)
W_V (qkv_proj的V部分) (4096, 1024)         (N, 4096)     (N, 1024)
W_O (o_proj)          (4096, 4096)         (N, 4096)     (N, 4096)
W_gate (gate_up的前半) (4096, 14336)        (N, 4096)     (N, 14336)
W_up (gate_up的后半)   (4096, 14336)        (N, 4096)     (N, 14336)
W_down (down_proj)     (14336, 4096)        (N, 14336)    (N, 4096)
LM Head              (4096, 32000)         (N, 4096)     (N, 32000)

Attention 内部（reshape 后，per head）：
Q reshape             无                   (N, 4096)     (N, 32, 128)
K reshape             无                   (N, 1024)     (N, 8, 128)
score = Q @ K^T       无                   (32,N,128)×(32,128,N) = (32, N, N)
softmax               无                   (32, N, N)    (32, N, N)
attn_out = w @ V      无                   (32,N,N)×(32,N,128) = (32, N, 128)
concat                无                   (32, N, 128)  (N, 4096)
```

### 5.3 TP 分片后的形状变化

```
TP=4（4卡张量并行），LLaMA-3-8B：

每张卡的 Q 头数: 32 / 4 = 8
每张卡的 KV 头数: 8 / 4 = 2

权重形状变化（每张卡）：
  W_Q: (4096, 4096) → (4096, 1024)    输出维度切 4 份
  W_K: (4096, 1024) → (4096, 256)     输出维度切 4 份
  W_V: (4096, 1024) → (4096, 256)     输出维度切 4 份
  W_O: (4096, 4096) → (1024, 4096)    输入维度切 4 份

每张卡只计算 8 个 Q 头 + 2 个 KV 头的注意力
最后通过 all-reduce 合并结果
```

### 5.4 形状公式总结

```
Q 输出维度 = num_heads × head_dim
           = num_heads × (hidden_size / num_heads)
           = hidden_size   （如果 head_dim = hidden_size / num_heads）

K 输出维度 = num_kv_heads × head_dim
V 输出维度 = num_kv_heads × head_dim

Attention 分数矩阵 = (seq_len, seq_len)   ← 每对 token 之间的相关性
  → 这就是为什么 Attention 的计算复杂度是 O(N²)

KV Cache 每层每 token 大小 = 2 × num_kv_heads × head_dim × dtype_bytes
  LLaMA-3-8B (FP16): 2 × 8 × 128 × 2 = 4096 bytes = 4KB per token per layer
  32 层总计: 32 × 4KB = 128KB per token
  4096 个 token: 4096 × 128KB = 512MB per sequence
```

---

## 6. 完整数值例子：手算多头 Attention

### 6.1 设定

```
极简配置（便于手算）：
  hidden_size = 4       （实际 4096）
  num_heads = 2         （实际 32）
  num_kv_heads = 2      （MHA，暂不用 GQA）
  head_dim = 4 / 2 = 2  （实际 128）
  seq_len = 3           （3 个 token）
```

### 6.2 输入与权重

```
hidden_states: (3, 4)        3 个 token，每个 4 维

X = [[ 1.0,  0.5, -0.3,  0.8],    ← token 0
     [ 0.2,  1.0,  0.4, -0.5],    ← token 1
     [-0.1,  0.3,  1.0,  0.6]]    ← token 2

W_Q: (4, 4)    投影到 Q 空间
W_Q = [[ 0.5,  0.1,  0.3, -0.2],
       [ 0.2,  0.4, -0.1,  0.3],
       [-0.3,  0.2,  0.5,  0.1],
       [ 0.1, -0.1,  0.2,  0.4]]

W_K: (4, 4)    投影到 K 空间
W_K = [[ 0.3, -0.1,  0.2,  0.4],
       [ 0.1,  0.3, -0.2,  0.1],
       [ 0.4,  0.2,  0.3, -0.3],
       [-0.2,  0.1,  0.1,  0.5]]

W_V: (4, 4)    投影到 V 空间
W_V = [[ 0.2,  0.3, -0.1,  0.4],
       [-0.1,  0.2,  0.4,  0.1],
       [ 0.3, -0.2,  0.1,  0.3],
       [ 0.1,  0.4,  0.2, -0.1]]
```

### 6.3 步骤 1：线性投影

```
Q = X @ W_Q    (3,4) × (4,4) = (3,4)

Q[0] = [1.0×0.5 + 0.5×0.2 + (-0.3)×(-0.3) + 0.8×0.1,   = 0.5+0.1+0.09+0.08 = 0.77
        1.0×0.1 + 0.5×0.4 + (-0.3)×0.2 + 0.8×(-0.1),      = 0.1+0.2-0.06-0.08 = 0.16
        1.0×0.3 + 0.5×(-0.1) + (-0.3)×0.5 + 0.8×0.2,       = 0.3-0.05-0.15+0.16 = 0.26
        1.0×(-0.2) + 0.5×0.3 + (-0.3)×0.1 + 0.8×0.4]       = -0.2+0.15-0.03+0.32 = 0.24

（类似计算 Q[1], Q[2], K, V，此处省略具体乘加步骤）
```

### 6.4 步骤 2：Reshape 为多头

```
Q: (3, 4) → reshape → (3, 2, 2)
  "3 个 token，2 个头，每头 2 维"

  token 0 的 Q 向量 [0.77, 0.16, 0.26, 0.24]
  → 头 0: [0.77, 0.16]    头 1: [0.26, 0.24]

  相当于把 4 维向量的前 2 维分给头 0，后 2 维分给头 1

同样处理 K 和 V。
```

### 6.5 步骤 3：每个头独立计算 Attention

```
── 头 0 ──

Q₀: (3, 2)  每个 token 的查询向量（头0的部分）
K₀: (3, 2)  每个 token 的键向量（头0的部分）
V₀: (3, 2)  每个 token 的值向量（头0的部分）

score₀ = Q₀ @ K₀^T / √2    (3,2) × (2,3) = (3,3)

score₀[i][j] = Q₀[i] · K₀[j] / √2
  → token i 对 token j 的"关注程度"（头0的视角）

因果掩罩: score₀[i][j] = -∞ 如果 j > i（不能看未来）

weights₀ = softmax(score₀)  每行和为 1

out₀ = weights₀ @ V₀        (3,3) × (3,2) = (3,2)
  → 每个 token 的新向量（头0的部分）= V₀ 的加权和


── 头 1 ──

完全独立地重复以上过程（用 Q、K、V 的后 2 维）
→ out₁: (3, 2)
```

### 6.6 步骤 4：拼接 + 输出投影

```
attn_output = concat(out₀, out₁)    (3, 2) cat (3, 2) = (3, 4)
  → 把每个头的 2 维结果拼回 4 维

output = attn_output @ W_O           (3, 4) × (4, 4) = (3, 4)
  → 最终的注意力层输出
```

### 6.7 关键领悟

```
"头" 不是一个独立的子网络，而是同一个权重矩阵的"不同片段"

W_Q = [W_Q_head0 | W_Q_head1]    按列拼接
       (4, 2)     (4, 2)

投影后 reshape = 拆开这些片段
拼接后输出投影 = 合并各片段的结果

整个过程的矩阵乘法总量 = 单头一样大
但 softmax 是独立做的 → 产生了不同的注意力模式
```

---

## 7. vLLM 中的代码实现

### 7.1 LlamaAttention 初始化

```python
# vllm/model_executor/models/llama.py 第 116~158 行

class LlamaAttention(nn.Module):
    def __init__(self, config, hidden_size, num_heads, num_kv_heads, ...):

        tp_size = get_tensor_model_parallel_world_size()

        # Q 头：在 TP 卡间均分
        self.total_num_heads = num_heads               # 全局 Q 头数（32）
        self.num_heads = self.total_num_heads // tp_size  # 每卡的 Q 头数

        # KV 头：也均分（但有最小值 1）
        self.total_num_kv_heads = num_kv_heads         # 全局 KV 头数（8）
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        # head_dim 由 hidden_size / num_heads 决定
        self.head_dim = hidden_size // self.total_num_heads   # 128

        # 每卡的 Q/KV 输出维度
        self.q_size  = self.num_heads    * self.head_dim   # 每卡 Q 总维度
        self.kv_size = self.num_kv_heads * self.head_dim   # 每卡 KV 总维度

        # 缩放因子
        self.scaling = self.head_dim ** -0.5   # 1/√128 ≈ 0.0884
```

### 7.2 QKV 合并投影

```python
# vllm/model_executor/layers/linear.py 第 867~916 行

class QKVParallelLinear(ColumnParallelLinear):
    """
    将 W_Q, W_K, W_V 三个���重矩阵合并为一个大矩阵，一次 GEMM 完成 QKV 投影。

    合并后的权重形状:
      (hidden_size, q_size + kv_size + kv_size)
      = (4096, 4096 + 1024 + 1024)
      = (4096, 6144)

    TP 分片后（TP=4，每卡）:
      (4096, 1024 + 256 + 256) = (4096, 1536)
    """
```

### 7.3 Forward：投影 → 拆分 → Attention

```python
# vllm/model_executor/models/llama.py 第 224~234 行

def forward(self, positions, hidden_states):
    # 1. 一次 GEMM 完成 QKV 投影
    qkv, _ = self.qkv_proj(hidden_states)
    #   hidden_states: (N, 4096)
    #   qkv_proj 权重: (4096, 6144)
    #   qkv: (N, 6144)

    # 2. 沿最后一维拆分为 Q, K, V
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    #   q: (N, 4096)    = (N, 32头 × 128维)
    #   k: (N, 1024)    = (N, 8头 × 128维)
    #   v: (N, 1024)    = (N, 8头 × 128维)

    # 3. 旋转位置编码（RoPE）
    q, k = self.rotary_emb(positions, q, k)
    #   RoPE 在每个头的 128 维内做 2D 旋转

    # 4. 注意力计算（内部处理 reshape、GQA 扩展、softmax、KV Cache）
    attn_output = self.attn(q, k, v)
    #   attn_output: (N, 4096)    = (N, 32头 × 128维)

    # 5. 输出投影
    output, _ = self.o_proj(attn_output)
    #   o_proj 权重: (4096, 4096)
    #   output: (N, 4096)

    return output
```

### 7.4 Attention 类：reshape 与 GQA

```python
# vllm/attention/layer.py 第 122~180 行

class Attention(nn.Module):
    """
    接收 flat 形式的 q, k, v:
      q: (N, num_heads × head_dim)
      k: (N, num_kv_heads × head_dim)
      v: (N, num_kv_heads × head_dim)

    内部处理:
      1. reshape 为多头形式: q → (N, num_heads, head_dim)
      2. GQA 扩展: k/v 从 num_kv_heads 扩展到 num_heads
      3. 写入 KV Cache
      4. 调用 Flash Attention / PagedAttention 内核
      5. 返回 (N, num_heads × head_dim)
    """
    def __init__(self, num_heads, head_size, scale, num_kv_heads=None, ...):
        if num_kv_heads is None:
            num_kv_heads = num_heads    # 默认 MHA
        assert num_heads % num_kv_heads == 0   # Q头数必须是KV头数的整数倍
```

---

## 8. LM Head：另一种"头"

"Head"（头）在深度学习中还有另一个含义——**模型输出头**。

### 8.1 注意力头 vs 模型头

```
注意力头（Attention Head）：
  Attention 层内部的并行子单元
  把 hidden_size 拆成多份，每份独立计算 Attention
  数量：num_heads（如 32）

模型头（Model Head）：
  整个模型最顶层的输出层
  将 hidden_state → 任务特定的输出
  数量：通常 1 个
```

### 8.2 常见模型头类型

```
LM Head（语言模型头）：
  功能：hidden_state → 词表概率分布
  形状：(hidden_size, vocab_size) = (4096, 32000)
  用途：文本生成（GPT、LLaMA、DeepSeek）

  hidden: (N, 4096) × W_lm: (4096, 32000) → logits: (N, 32000)
  → softmax → 每个 token 位置对应一个词表概率分布

分类头（Classification Head）：
  功能：hidden_state → 类别概率
  形状：(hidden_size, num_classes) = (4096, 2)
  用途：情感分类、意图识别

嵌入头（Embedding Head / Pooling Head）：
  功能：hidden_state → 句子向量
  用途：向量检索、语义相似度（BGE、E5 等模型）

多模态头：
  功能：hidden_state → 图像/音频 token
  用途：多模态生成
```

### 8.3 LM Head 与 Embedding 的关系

```
很多模型中，LM Head 与 Embedding 层共享权重（weight tying）：

Embedding: token_id → hidden_state
  E[token_id] = E 矩阵的第 token_id 行    E: (32000, 4096)

LM Head: hidden_state → logits
  logits = hidden @ E^T                    E^T: (4096, 32000)

为什么可以共享？
  Embedding 把词映射到语义空间
  LM Head 在语义空间中找"最像的词"
  两者本质是同一个映射的正反两个方向

好处：省下 32000 × 4096 × 2 bytes = 250MB 参数
```

---

## 总结

**注意力头（Attention Head）** 是将高维注意力计算拆成多个低维独立计算的并行机制。核心关系：

```
hidden_size = num_heads × head_dim

                ┌──── num_heads（Q头数）───── 决定 Q 输出维度、注意力并行度
                │
config.json ────┼──── num_kv_heads（KV头数）── 决定 K/V 输出维度、KV Cache 大小
                │
                ├──── hidden_size ──────────── 决定所有权重矩阵的输入维度
                │
                └──── head_dim（自动算出）───── 决定每个头内部的向量维度
                      = hidden_size / num_heads    和 Attention 的 scale = 1/√head_dim
```

**头与 KV Cache 的关系**：KV Cache 的大小直接正比于 `num_kv_heads`。GQA 通过减少 KV 头数（多个 Q 头共享 KV），实现 KV Cache 4× 压缩（LLaMA-3-8B：32 Q 头共享 8 KV 头），几乎不损失质量。

**矩阵形状由什么决定**：`config.json` 中的 4 个参数（`hidden_size`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`）+ 运行时的 token 数 + TP 分片数 → 完全确定了模型中每一个矩阵的形状。
