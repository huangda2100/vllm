# 注意力计算中的张量形状深度解析

> 涉及文件：`vllm/model_executor/models/llama.py`、`vllm/attention/layer.py`、`vllm/v1/attention/backends/flash_attn.py`、`csrc/attention/attention_kernels.cuh`
> 核心问题：注意力计算中每个张量的形状是什么？为什么是这个形状？vLLM 中是如何体现的？

---

## 一、从数学出发：注意力的本质

标准自注意力公式：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
```

分解理解每个矩阵的含义：

| 符号 | 含义 | 维度 |
|------|------|------|
| **Q** | Query：当前 token "想要什么" | `[seq_len, d_k]` |
| **K** | Key：历史 token "能提供什么标签" | `[seq_len, d_k]` |
| **V** | Value：历史 token "实际提供的内容" | `[seq_len, d_v]` |
| **Q·Kᵀ** | 注意力分数：每个 Q 与所有 K 的相似度 | `[seq_len, seq_len]` |
| **softmax(…)** | 归一化的注意力权重 | `[seq_len, seq_len]` |
| **输出** | 加权后的 Value 聚合 | `[seq_len, d_v]` |

这里 `d_k = d_v = head_dim`（每个注意力头的维度）。

**为什么要除以 `√d_k`？**
当 `d_k` 较大时，Q·Kᵀ 的值方差 ≈ d_k（期望为0，方差随维度增长），导致 softmax 进入饱和区（梯度消失）。除以 `√d_k` 将方差归一化为 1，维持 softmax 在有效梯度区域。

---

## 二、多头注意力：为什么要"多头"

单头注意力只能学习一种相关性模式。多头注意力（MHA）将 hidden_size 切成 `num_heads` 份，每个头独立学习不同的注意力模式：

```
hidden_size = num_heads × head_dim

例：LLaMA-3-8B
  hidden_size = 4096
  num_heads   = 32
  head_dim    = 4096 / 32 = 128
```

每个头独立计算注意力，最后将所有头的输出拼接（concat）还原回 `hidden_size`。

**直觉**：不同的头可以关注不同的语义关系——
- 头1：关注句法结构（主语-谓语）
- 头2：关注语义相似性
- 头3：关注位置关系
- ...

---

## 三、vLLM 的核心表示：Flat Batch（扁平批次）

这是理解所有形状的关键基础，与标准 PyTorch 批处理完全不同。

### 传统批处理（HuggingFace 风格）

```
batch_size=3，填充到最大长度 max_len=5:

[token_A1, token_A2, <pad>, <pad>, <pad>]  ← 序列 A, 长度 2
[token_B1, token_B2, token_B3, <pad>, <pad>]  ← 序列 B, 长度 3
[token_C1, token_C2, token_C3, token_C4, token_C5]  ← 序列 C, 长度 5

张量形状：[batch_size=3, max_len=5, hidden_size]
总计算量：3 × 5 = 15 个 token 的计算（但有 5 个是 pad，浪费）
```

### vLLM 的 Flat Batch（连续批处理的核心）

```
3 个序列所有 token 拼接成一条：

[token_A1, token_A2, token_B1, token_B2, token_B3, token_C1, ..., token_C5]

张量形状：[total_tokens=10, hidden_size]
总计算量：10 个 token（无浪费！）
```

**为什么 vLLM 用 Flat Batch？**
1. **无 padding 浪费**：不同长度的序列不需要补齐
2. **连续批处理支持**：每步可以插入/删除序列，token 数量动态变化
3. **内存高效**：单个大矩阵乘法比多个小矩阵乘法更高效

**代价**：需要额外的元数据（`cu_seqlens_q`、`seq_lens`）来追踪每条序列的边界。

---

## 四、形状变化全流程追踪

以 LLaMA-3-8B 在 TP=1 下、批次包含 2 个序列（长度 3 和 5）为例：

```
参数：
  hidden_size    = 4096
  num_heads      = 32   (Q heads)
  num_kv_heads   = 8    (KV heads，GQA)
  head_dim       = 128
  total_tokens   = 3 + 5 = 8
```

### 步骤 0：输入 hidden_states

```python
hidden_states: [total_tokens, hidden_size]
             = [8, 4096]
```

这是上一层（或 embedding 层）输出的每个 token 的表示向量。

### 步骤 1：QKV 投影

```python
# llama.py - LlamaAttention.forward()
qkv, _ = self.qkv_proj(hidden_states)
```

`qkv_proj` 是一个 `QKVParallelLinear`，权重形状：

```
W_qkv: [hidden_size, (num_q_heads + 2*num_kv_heads) * head_dim]
      = [4096,  (32 + 2×8) × 128]
      = [4096,  4096 + 1024 + 1024]
      = [4096,  6144]
```

矩阵乘法：

```
qkv = hidden_states @ W_qkv
    = [8, 4096] @ [4096, 6144]
    = [8, 6144]

qkv: [total_tokens, (num_q_heads + 2*num_kv_heads) * head_dim]
   = [8, 6144]
```

**为什么 Q、K、V 投影合并成一个矩阵？**
三个分开的矩阵乘法 vs 一个大矩阵乘法：GPU Tensor Core 效率更高，内核启动次数更少。

### 步骤 2：分割 Q、K、V

```python
# llama.py
q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
# q_size  = num_heads    * head_dim = 32 × 128 = 4096
# kv_size = num_kv_heads * head_dim =  8 × 128 = 1024
```

```
q: [total_tokens, num_heads    * head_dim] = [8, 4096]
k: [total_tokens, num_kv_heads * head_dim] = [8, 1024]
v: [total_tokens, num_kv_heads * head_dim] = [8, 1024]
```

**关键**：Q 的维度远大于 K/V（32 heads vs 8 heads），这是 GQA 的标志。

### 步骤 3：RoPE 位置编码

```python
# llama.py
q, k = self.rotary_emb(positions, q, k)
# positions: [total_tokens]，每个 token 在其序列中的绝对位置
# positions = [0, 1, 2, 0, 1, 2, 3, 4]  ← 序列A(3 tokens) + 序列B(5 tokens)
```

RoPE 不改变形状，只是对 q 和 k 的每个向量施加旋转变换：

```
q: [8, 4096] → [8, 4096]  (形状不变，内容变化)
k: [8, 1024] → [8, 1024]  (形状不变，内容变化)
```

在 vLLM 的 flat batch 中，`positions` 是显式传入的，不是 `0,1,2,...,total_tokens-1`，而是各序列内部的相对位置。这样跨序列的 token 不会产生"伪相对位置关系"。

### 步骤 4：注意力计算前的 Reshape

在进入 FlashAttention 之前，q/k/v 需要 reshape 成 `[tokens, heads, head_dim]`：

```python
# 在 attention 内部（隐式 reshape）
q = q.view(total_tokens, num_heads, head_dim)
  = [8, 32, 128]

k = k.view(total_tokens, num_kv_heads, head_dim)
  = [8,  8, 128]

v = v.view(total_tokens, num_kv_heads, head_dim)
  = [8,  8, 128]
```

**为什么要 reshape 成 3D？**
FlashAttention 的内核以 `[tokens, heads, head_dim]` 为输入，可以对每个 head 并行计算注意力。不 reshape 就无法区分"哪些维度属于哪个头"。

### 步骤 5：KV Cache 写入

```python
# flash_attn.py - FlashAttentionImpl.forward()
reshape_and_cache_flash(
    key,          # [total_tokens, num_kv_heads, head_dim]
    value,        # [total_tokens, num_kv_heads, head_dim]
    key_cache,    # [num_blocks, block_size, num_kv_heads, head_dim]（NHD layout）
    value_cache,  # [num_blocks, block_size, num_kv_heads, head_dim]
    slot_mapping, # [total_tokens]，每个 token 写入哪个物理 slot
    ...
)
```

`slot_mapping` 是关键：
```
slot_mapping = [42, 43, 44,   # 序列A的3个token写入物理slot 42,43,44
                17, 18, 19, 20, 21]  # 序列B的5个token写入物理slot 17~21
```

物理 slot 编号 = `physical_block_id * block_size + offset_in_block`。

### 步骤 6：注意力计算（FlashAttention varlen）

```python
flash_attn_varlen_func(
    q=query[:num_actual_tokens],   # [8, 32, 128]
    k=key_cache,                    # [num_blocks, block_size, num_kv_heads, head_dim]
    v=value_cache,                  # [num_blocks, block_size, num_kv_heads, head_dim]
    out=output[:num_actual_tokens], # [8, 32, 128]
    cu_seqlens_q=[0, 3, 8],        # 序列边界：[cumsum of query_lengths]
    max_seqlen_q=5,
    seqused_k=[10, 20],            # 每个序列实际的 KV 长度（含历史）
    max_seqlen_k=20,
    block_table=...,               # [num_seqs, max_num_blocks_per_seq]
)
```

**`cu_seqlens_q`（cumulative sequence lengths）** 是 flat batch 的"目录"：

```
cu_seqlens_q = [0, 3, 8]
表示：
  序列0: q[0:3]  （3个 query tokens）
  序列1: q[3:8]  （5个 query tokens）
```

FlashAttention 根据这个数组，对每条序列独立计算 causal mask，**不同序列之间不会相互看到对方的 token**。

### 步骤 7：输出 reshape

```
output: [total_tokens, num_heads, head_dim]
      = [8, 32, 128]

→ view/reshape → [total_tokens, num_heads * head_dim]
               = [8, 4096]
```

### 步骤 8：输出投影 o_proj

```python
output, _ = self.o_proj(attn_output)
# attn_output: [8, 4096]
# o_proj: RowParallelLinear，权重 [num_heads * head_dim, hidden_size] = [4096, 4096]
# output: [8, 4096]
```

形状不变，但内容经过线性变换，混合了所有头的信息。

---

## 五、KV Cache 的形状设计

KV Cache 形状在不同 backend 中不同，原因是各自的访问模式不同。

### FlashAttention Backend（vLLM V1 默认）

```python
# flash_attn.py - FlashAttentionBackend.get_kv_cache_shape()
return (2, num_blocks, block_size, num_kv_heads, head_size)
# 2        → K 和 V 各一份，index 0=K，1=V
# num_blocks → 物理 block 数（GPU 显存切成多少块）
# block_size → 每块存多少个 token 的 KV（通常 16）
# num_kv_heads → GQA 下 KV heads 数
# head_size  → 每个 head 的维度
```

具体数字：LLaMA-3-8B，80GB GPU，block_size=16：

```
num_kv_heads = 8
head_size    = 128
block_size   = 16

每个 KV Block 的内存：
  2 × 16 × 8 × 128 × 2 bytes (BF16) = 65,536 bytes = 64 KB

80GB GPU，假设 75% 用于 KV Cache：
  可用内存 = 60 GB
  num_blocks = 60 * 1024 * 1024 * 1024 / 65536 ≈ 983,040 blocks
  最大上下文总 tokens = 983,040 × 16 ≈ 15M tokens
```

### Layout 选项：NHD vs HND

```python
# flash_attn.py
if cache_layout == "NHD":
    # [num_blocks, block_size, num_kv_heads, head_size]
    stride_order = (0, 1, 2, 3, 4)   # 自然顺序
elif cache_layout == "HND":
    # [num_blocks, num_kv_heads, block_size, head_size]
    stride_order = (0, 1, 3, 2, 4)   # block_size 和 num_kv_heads 交换
```

- **NHD**（默认）：`[block, token_in_block, head, dim]`，适合按 token 顺序访问
- **HND**：`[block, head, token_in_block, dim]`，适合按 head 并行访问

### PagedAttention Backend（CUDA 内核，旧版）

K 和 V 用不同的 layout（已在 `04_cuda_attention_kernels.md` 详解）：

```cpp
// csrc/attention/attention_kernels.cuh 注释
k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
v_cache: [num_blocks, num_kv_heads, head_size, block_size]
```

K 的 `head_size/x, block_size, x` 布局是为 Q·K 点积的列访问模式优化；
V 的 `head_size, block_size` 布局是为加权求和的行访问模式优化。

---

## 六、GQA 的形状：Q heads ≠ KV heads

### 为什么 GQA 的形状是合理的？

```
标准 MHA（Multi-Head Attention）：
  Q: [tokens, 32, 128]
  K: [tokens, 32, 128]   → KV Cache 极大
  V: [tokens, 32, 128]

GQA（Grouped Query Attention）：
  Q: [tokens, 32, 128]   ← 保持不变，expressiveness 不减少
  K: [tokens,  8, 128]   → KV Cache 减少 4 倍！
  V: [tokens,  8, 128]

每组 4 个 Q heads 共享 1 对 KV heads
```

```python
# llama.py - LlamaAttention
self.total_num_heads    = 32   # Q heads
self.num_kv_heads       = 8    # KV heads (TP=1)
self.head_dim           = 128

self.q_size  = 32 * 128 = 4096  # Q 投影输出
self.kv_size =  8 * 128 = 1024  # K 和 V 各自的投影输出

num_queries_per_kv = 32 / 8 = 4  # 每组共享比例
```

### 注意力计算时的 GQA 处理

FlashAttention 原生支持 GQA，只需传入不同 heads 数量的 Q 和 KV：

```python
flash_attn_varlen_func(
    q=query,   # [tokens, 32, 128]   32个 Q heads
    k=key_cache,   # [blocks, block_size, 8, 128]  8个 KV heads
    v=value_cache, # [blocks, block_size, 8, 128]
    ...
)
# FlashAttention 内部自动 broadcast KV → 每组4个Q都看同一对KV
```

在 PagedAttention 内核里是显式计算：

```cpp
// attention_kernels.cuh
const int num_queries_per_kv = num_heads / num_kv_heads;  // = 4
const int kv_head_idx = head_idx / num_queries_per_kv;    // head 32 → KV head 8
```

---

## 七、Prefill vs Decode 阶段的形状差异

这是 LLM 推理最核心的形状区别。

### Prefill 阶段

```
输入：全部 prompt tokens 一起处理

hidden_states: [total_prompt_tokens, hidden_size]
# 例如：5个请求各自512 tokens → total = 2560 tokens

q: [2560, 32, 128]
k: [2560,  8, 128]
v: [2560,  8, 128]

cu_seqlens_q = [0, 512, 1024, 1536, 2048, 2560]  ← 各序列边界
```

Prefill 是 **Compute-Bound**（计算密集）：每个 token 都要和前面所有 token 计算注意力，计算量 O(n²)。

### Decode 阶段

```
输入：每个序列只有 1 个新生成的 token

hidden_states: [batch_size, hidden_size]
# 例如：5个请求各产生 1 token → total = 5 tokens

q: [5, 32, 128]   ← 极小！只有 batch_size 个查询
k: 来自 KV Cache  ← 数千个历史 KV，从显存读取
v: 来自 KV Cache

cu_seqlens_q = [0, 1, 2, 3, 4, 5]  ← 每个序列恰好 1 个 query token
seqused_k    = [513, 516, 489, ...]  ← 每个序列历史 KV 的长度
```

Decode 是 **Memory-Bound**（内存带宽密集）：
- 计算量：5 × 32 × 128 = 20,480 次乘加（极少）
- 内存访问：每个序列可能有 1000+ tokens 的 KV Cache，5 个序列共需读取数十 MB

**这解释了为什么 vLLM 需要两套不同的注意力内核：**

| 阶段 | 瓶颈 | 推荐内核 |
|------|------|---------|
| Prefill | Compute-Bound | FlashAttention（tiling + SRAM 复用） |
| Decode | Memory-Bound | PagedAttention（优化非连续 KV 读取）或 FlashDecoding |

---

## 八、张量并行下的形状变化

TP=2 时，每个 GPU 只处理一半的 heads。

```
全局参数：
  total_num_heads = 32
  total_num_kv_heads = 8

TP=2, rank=0：
  num_heads    = 32 / 2 = 16
  num_kv_heads =  8 / 2 = 4
  q_size  = 16 × 128 = 2048
  kv_size =  4 × 128 =  512

本 rank 的张量形状：
  hidden_states: [tokens, 4096]  ← 不切分（每个 GPU 都有完整 hidden_states）
  qkv_proj 权重: [4096, 2048 + 512 + 512] = [4096, 3072]  ← 仅本 rank 的部分
  qkv output:   [tokens, 3072]
  q: [tokens, 16, 128]
  k: [tokens,  4, 128]
  v: [tokens,  4, 128]
  attn_output: [tokens, 16, 128] = [tokens, 2048]
  o_proj 输入:  [tokens, 2048]   ← RowParallel，本 rank 的切片
  o_proj 输出:  [tokens, 4096]   ← AllReduce 后恢复完整 hidden_size
```

**关键**：`AllReduce` 在 `o_proj`（RowParallelLinear）后发生，把各 GPU 的部分结果加和，恢复完整的 `[tokens, 4096]`。

---

## 九、辅助元数据张量形状

注意力计算除了 Q/K/V，还需要一系列元数据张量，它们的形状同样有严格含义。

### cu_seqlens_q（cumulative sequence lengths）

```
类型: [num_seqs + 1], int32
内容: 各序列在 flat batch 中的起始位置（前缀和）

例：3个序列，长度 [3, 5, 2]
cu_seqlens_q = [0, 3, 8, 10]
             = cumsum([0, 3, 5, 2])

用途：
  序列 i 的 query tokens = flat_q[cu_seqlens_q[i] : cu_seqlens_q[i+1]]
```

### block_table

```
类型: [num_seqs, max_num_blocks_per_seq], int32
内容: 每个序列的逻辑 block 号 → 物理 block 号的映射

例：max_num_blocks_per_seq = 4
block_table = [
    [42, 43, 44, -1],   # 序列A：3个有效block
    [17, 18, 19, 20],   # 序列B：4个有效block
]

访问序列A第2个block中第3个token的K：
  physical_block_id = block_table[0][1] = 43
  k = key_cache[43][2]  # [block_size=16, num_kv_heads, head_dim] → 第3个
```

### slot_mapping

```
类型: [total_tokens], int64
内容: 每个新 token 应该写入 KV Cache 的物理 slot 编号

物理 slot = physical_block_id × block_size + offset_within_block

例：新 token 要写入 block 42 的第 3 个位置（block_size=16）：
  slot = 42 × 16 + 3 = 675

slot_mapping = [675, 676, 677,   # 序列A的3个新token
                272, 273, 274, 275, 276]  # 序列B的5个新token
```

### seq_lens（seqused_k）

```
类型: [num_seqs], int32
内容: 每个序列当前的总 KV 长度（包含已缓存的历史 + 本步新 token）

seq_lens = [513, 516]
意味着：序列A已有513个KV，序列B已有516个KV
```

---

## 十、完整形状流程图

```
输入（Flat Batch，8个tokens，2个序列）
        ↓
hidden_states: [8, 4096]           ← [total_tokens, hidden_size]
        ↓  qkv_proj（大矩阵乘）
qkv: [8, 6144]                     ← [total_tokens, (32+8+8)*128]
        ↓  split
q:   [8, 4096]  → reshape → [8, 32, 128]   ← [T, num_q_heads,  head_dim]
k:   [8, 1024]  → reshape → [8,  8, 128]   ← [T, num_kv_heads, head_dim]
v:   [8, 1024]  → reshape → [8,  8, 128]   ← [T, num_kv_heads, head_dim]
        ↓  RoPE（形状不变）
q:   [8, 32, 128]
k:   [8,  8, 128]
        ↓  写入 KV Cache
key_cache:   [num_blocks, block_size, 8, 128]  ← [B, S, H_kv, D]
value_cache: [num_blocks, block_size, 8, 128]

        ↓  FlashAttention varlen
        ┌─────────────────────────────┐
        │ cu_seqlens_q = [0, 3, 8]   │  ← 序列边界
        │ block_table  = [B, max_B]  │  ← 物理块映射
        │ seq_lens     = [10, 20]    │  ← KV 长度
        └─────────────────────────────┘
output: [8, 32, 128]               ← [total_tokens, num_heads, head_dim]
        ↓  view/reshape
output: [8, 4096]                  ← [total_tokens, num_heads * head_dim]
        ↓  o_proj（矩阵乘 + AllReduce）
output: [8, 4096]                  ← [total_tokens, hidden_size]
```

---

## 十一、为什么各维度是固定值

### head_dim 为什么通常是 128？

```
head_dim = hidden_size / num_heads

LLaMA-3-8B:  4096 / 32 = 128
LLaMA-3-70B: 8192 / 64 = 128
GPT-4（估计）: 可能 128 或 256
```

**128 是硬件友好的数字**：
- 128 × 2 bytes (BF16) = 256 bytes = 2 个 cache line（128 bytes/cache line）
- 128 是 Tensor Core warp tile 的整数倍（M=16, N=16, K=16 的倍数）
- FlashAttention/PagedAttention 的向量化加载恰好是 `16 bytes / sizeof(BF16) = 8` 个元素的整数倍

### block_size 为什么默认是 16？

```python
# flash_attn.py
if block_size % 16 != 0:
    raise ValueError("Block size must be a multiple of 16.")
```

- **Warp 对齐**：CUDA warp 有 32 个线程，16 是 warp 的一半，便于 warp-level 并行处理 KV block
- **向量化读取**：16 tokens × head_dim 可以恰好用向量化指令（128-bit load = float4）连续读取
- **Flash/PagedAttn tiling**：FlashAttention 的 block GEMM tile 大小通常是 16 的倍数

### hidden_size 为什么是 4096 而非 4000？

- 4096 = 2¹²，是 2 的幂次，对 GPU 内存访问的对齐极为友好
- 4096 / 32 (num_heads) = 128 (head_dim)，整除
- 4096 / 8 (TP size) = 512，整除，适合各种 TP 配置

---

## 十二、形状与性能的关系

| 形状特征 | 性能影响 |
|---------|---------|
| `total_tokens` 小（decode 阶段）| Memory-Bound，需优化 KV 读取 |
| `total_tokens` 大（prefill 阶段）| Compute-Bound，需优化矩阵乘 |
| `head_dim = 128`（2的幂）| 向量化加载对齐，Tensor Core 友好 |
| `block_size = 16`（16倍数）| CUDA warp 对齐，减少内存事务 |
| flat batch（无 padding）| 无浪费计算，硬件利用率最大化 |
| Q/KV heads 不同（GQA）| KV Cache 减少 4x，内存带宽压力减少 |
| cu_seqlens_q 连续存储 | FlashAttention 内部快速 binary search 定位序列边界 |

---

*涉及文件：`vllm/model_executor/models/llama.py`，`vllm/v1/attention/backends/flash_attn.py`，`csrc/attention/attention_kernels.cuh`*
*更新：2026-03*
