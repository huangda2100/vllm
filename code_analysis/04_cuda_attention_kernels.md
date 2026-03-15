# vLLM PagedAttention CUDA 内核深度解析

> 目录：`csrc/attention/`
> 核心文件：`attention_kernels.cuh`（691 行）、`paged_attention_v1.cu`、`paged_attention_v2.cu`
> 定位：PagedAttention 的 CUDA 底层实现，支持非连续 KV Cache 的 Decode 阶段注意力计算

---

## 一、为什么需要自定义 CUDA 内核？

标准 FlashAttention 假设 KV Cache 是**连续的内存**。PagedAttention 允许 KV Cache 分散在不连续的 Block 中，需要根据 `block_table`（逻辑 block 号 → 物理 block 号的映射表）随机访问内存。

此外，Decode 阶段的特性与 Prefill 完全不同：
- Prefill：Q 的序列长度 = 输入长度（可能数千），适合 FlashAttention 的 tiling
- **Decode：每个请求只有1个新 token（Q 长度 = 1），但需要访问整个 KV 序列**

Decode 阶段是典型的 **Memory-Bound** 操作：计算量极少（只有1个 Q），但需要从 GPU HBM 读取大量 KV 数据。需要针对这个特性专门优化。

---

## 二、KV Cache 的内存布局设计

### Key Cache 布局（非对称！）

```
k_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
```

其中 `x = 16 / sizeof(cache_t)`（FP16 时 x=8，FP8 时 x=16）。

**为什么不是直觉上的 `[num_blocks, num_kv_heads, block_size, head_size]`？**

这个奇怪的布局是为了配合硬件的 **向量化内存访问（Coalesced Access）**。

每个 Thread Group（由 `THREAD_GROUP_SIZE = WARP_SIZE/BLOCK_SIZE` 个线程组成）同时处理一个 token 的 key 向量。它们协同读取时，每次取 16 bytes：

```
线程 0 读取 key[0..7]（FP16，8个元素 = 16 bytes）
线程 1 读取 key[8..15]
...
```

如果 Key 按 `[block_size, head_size]` 排列，这些线程会读取同一行的不同部分，**内存地址不连续** → 不合并访问，需要多次内存事务。

而 `[head_size/x, block_size, x]` 排列，使得同一 block 内、同一 head_dim 维度上的 x 个相邻元素**物理上连续**，Thread Group 的协同读取恰好对齐到 16 bytes 边界，实现合并访问。

### Value Cache 布局（不同于 Key！）

```
v_cache: [num_blocks, num_kv_heads, head_size, block_size]
```

V Cache 访问模式不同：计算 `Σ softmax_score_i * V_i` 时，需要按 **head_dim 的一行** 遍历所有 token：

```
输出 out[row] = Σ_i (logits[i] * v_cache[block_i][head][row][i % block_size])
```

此时 `block_size` 在最内层，每个 warp 按 `V_VEC_SIZE = 16/sizeof(scalar_t)` 对齐访问同一行的连续元素，同样实现合并访问。

**Key 和 Value 用不同 layout 的根本原因**：K 用于 Q·K 点积（列访问），V 用于加权求和（行访问），访问模式不同，需要不同的 layout 来优化内存局部性。

---

## 三、Thread Group 设计

```cpp
constexpr int THREAD_GROUP_SIZE = MAX(WARP_SIZE / BLOCK_SIZE, 1);
// 例如：WARP_SIZE=32, BLOCK_SIZE=16 → THREAD_GROUP_SIZE=2
```

一个 Thread Group 协作处理**一个 token 的 QK 点积**：

```
BLOCK_SIZE=16 的 KV block，由32个线程（1个 warp）处理：
  thread group 0（线程 0,1）  → token 0
  thread group 1（线程 2,3）  → token 1
  ...
  thread group 15（线程 30,31）→ token 15
```

每个 Thread Group 内的线程分工：
```cpp
constexpr int VEC_SIZE = MAX(16 / (THREAD_GROUP_SIZE * sizeof(scalar_t)), 1);
// FP16, THREAD_GROUP_SIZE=2: VEC_SIZE = 16/(2*2) = 4

constexpr int NUM_ELEMS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE;
// HEAD_SIZE=128, GROUP=2: 每线程负责64个元素

constexpr int NUM_VECS_PER_THREAD = NUM_ELEMS_PER_THREAD / VEC_SIZE;
// 64/4 = 16 个向量
```

同一 Thread Group 内的线程持有**同一个 token 的不同维度部分的 key**，最后通过 `Qk_dot` 中的 warp reduce 求和得到完整的 qk dot product。

---

## 四、主循环：QK 计算与 Online Softmax

### 分配共享内存

```cpp
extern __shared__ char shared_mem[];
float* logits = reinterpret_cast<float*>(shared_mem);
// logits 数组大小 = padded_max_seq_len，存储所有 token 的注意力分数
```

**用 FP32 存 logits 而非 FP16**：softmax 计算对数值精度敏感（需要求 exp），用 FP16 容易下溢/溢出，必须用 FP32。

### 遍历 Key Blocks

```cpp
for (int block_idx = start_block_idx + warp_idx; block_idx < end_block_idx;
     block_idx += NUM_WARPS) {
    // 通过 block_table 查找物理 block 号（非连续内存的核心）
    const int64_t physical_block_number =
        static_cast<int64_t>(block_table[block_idx]);

    // 加载 K 向量（向量化，16 bytes/次）
    for (int j = 0; j < NUM_VECS_PER_THREAD; j++) {
        const cache_t* k_ptr = k_cache + physical_block_number * kv_block_stride + ...;
        k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + ...);
    }

    // QK 点积 + Thread Group 内部 reduce
    float qk = scale * Qk_dot<scalar_t, THREAD_GROUP_SIZE>::dot(q_vecs, k_vecs);
    logits[token_idx - start_token_idx] = qk;
    qk_max = fmaxf(qk_max, qk);
}
```

关键：`block_table[block_idx]` 将逻辑 block 索引映射到物理内存地址，实现**非连续内存的随机访问**。

**int64 类型转换**：

```cpp
const int64_t physical_block_number = static_cast<int64_t>(block_table[block_idx]);
```

注释说明：block_table 存 int32，但乘以 `kv_block_stride` 后可能溢出 int32（大模型有数万个 blocks），强制转 int64 避免溢出。这是一个不容忽视的细节，真实工程中的 off-by-one / overflow bug 常来源于此。

### Online Softmax（数值稳定）

```cpp
// 第一步：找全局最大值 qk_max（用于数值稳定性）
for (int mask = WARP_SIZE / 2; mask >= THREAD_GROUP_SIZE; mask /= 2)
    qk_max = fmaxf(qk_max, VLLM_SHFL_XOR_SYNC(qk_max, mask));
// 跨 warp 归约
if (lane == 0) red_smem[warp_idx] = qk_max;
__syncthreads();
qk_max = VLLM_SHFL_SYNC(qk_max, 0); // 广播给所有线程

// 第二步：计算 exp(qk - qk_max) 并累加
float exp_sum = 0.f;
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS) {
    float val = __expf(logits[i] - qk_max);
    logits[i] = val;   // 就地替换！节省内存
    exp_sum += val;
}
exp_sum = block_sum<NUM_WARPS>(&red_smem[NUM_WARPS], exp_sum);

// 第三步：归一化
const float inv_sum = __fdividef(1.f, exp_sum + 1e-6f);
for (int i = thread_idx; i < num_tokens; i += NUM_THREADS)
    logits[i] *= inv_sum;
```

**为什么减 qk_max？**

直接计算 `exp(qk_i)` 时，若 `qk_i` 较大（如 30），`exp(30) ≈ 1e13`，极易数值溢出。
减去最大值后 `exp(qk_i - qk_max) ≤ 1`，数值稳定。

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
             = exp(x_i - max) / Σ exp(x_j - max)  （数学等价）
```

**1e-6f 的作用**：防止 `exp_sum = 0`（序列为空时）导致除零。

---

## 五、共享内存复用：logits → output

```cpp
// 注意力阶段：共享内存存 logits（FP32，大小 = padded_max_seq_len * 4 bytes）
float* logits = reinterpret_cast<float*>(shared_mem);

// softmax 计算完毕，现在需要用共享内存存 output 的中间结果
// 必须有 __syncthreads() 隔离！
__syncthreads();

// output 阶段：同一块共享内存复用为 output accumulation buffer
float* out_smem = reinterpret_cast<float*>(shared_mem);
```

共享内存极其珍贵（一般只有几十 KB/SM）。logits 和 output 的计算时间不重叠，因此可以**复用同一块共享内存**，节省约 `max_seq_len * 4 bytes` 的共享内存压力。

代码注释特别指出：中间必须有 `__syncthreads()` 作为内存屏障，确保所有线程读完 logits 后再覆写。

---

## 六、Value 加权求和

```cpp
constexpr int V_VEC_SIZE = MIN(16 / sizeof(scalar_t), BLOCK_SIZE);
// FP16: V_VEC_SIZE = min(8, BLOCK_SIZE)

for (int block_idx = ...; ) {
    // 从共享内存读取 softmax 后的 logits
    L_vec logits_vec = *reinterpret_cast<Float_L_vec*>(logits + token_idx);

    // 从 V Cache 读取 value
    const cache_t* v_ptr = v_cache + physical_block_number * kv_block_stride + ...;
    V_vec v_vec = *reinterpret_cast<const V_vec*>(v_ptr + row * BLOCK_SIZE + offset);

    // FP8 解量化（如果 KV Cache 是 FP8 格式）
    if constexpr (KV_DTYPE != Fp8KVCacheDataType::kAuto) {
        v_vec = fp8::scaled_convert<V_vec, V_quant_vec, KV_DTYPE>(v_quant_vec, *v_scale);
    }

    // 处理序列尾部的 padding（防止 NaN 污染）
    if (block_idx == num_seq_blocks - 1) {
        for (int j = 0; j < V_VEC_SIZE; j++)
            v_vec_ptr[j] = token_idx + j < seq_len ? v_vec_ptr[j] : zero_value;
    }

    accs[i] += dot(logits_vec, v_vec);
}
```

**NaN 处理**：最后一个 block 可能未填满（如序列长度 = 17，block_size = 16，最后一个 block 只有1个有效 token）。padding 位置的内存值是未初始化的垃圾数据，可能是 NaN/Inf。必须显式置零，否则与 logits=0 的权重相乘仍会产生 `0 * NaN = NaN`，导致整个输出污染。

---

## 七、V1 vs V2：解决超长序列问题

### V1（单分区）

```
Grid: (num_heads, num_seqs, 1)

每个 thread block 处理一个 (head, seq) 对的完整序列
```

**限制**：序列很长（如 32K tokens）时，单个 thread block 需要遍历所有 block，共享内存需要 `32K * 4 bytes = 128KB`（超过限制），且无法利用多个 SM 并行处理同一序列。

### V2（分区并行）

```
Grid: (num_heads, num_seqs, max_num_partitions)

每个 thread block 只处理序列的一段（partition）
然后由 reduce kernel 合并所有分区的结果
```

**关键挑战**：如何合并多个分区的 softmax 结果？

### V2 reduce kernel：跨分区合并 Online Softmax

每个分区独立计算 `max_logit_p` 和 `exp_sum_p`，输出局部的 `tmp_out_p`。

Reduce kernel 用**两阶段 online softmax** 合并：

```cpp
// 第一步：找全局 max_logit = max(max_logit_0, max_logit_1, ...)
float max_logit = -FLT_MAX;
for (int i = threadIdx.x; i < num_partitions; i += blockDim.x)
    max_logit = fmaxf(max_logit, max_logits_ptr[i]);
// ... warp reduce ...

// 第二步：rescale 每个分区的 exp_sum
for (int i = threadIdx.x; i < num_partitions; i += blockDim.x) {
    float l = shared_max_logits[i];
    float rescaled_exp_sum = exp_sums_ptr[i] * expf(l - max_logit);
    // 每个分区的 exp_sum 是在自己的 max 下算的
    // 换算到全局 max：乘以 exp(local_max - global_max)
    global_exp_sum += rescaled_exp_sum;
    shared_exp_sums[i] = rescaled_exp_sum;
}

// 第三步：用 rescaled 权重加权 tmp_out
for (int i = threadIdx.x; i < HEAD_SIZE; i += NUM_THREADS) {
    float acc = 0.0f;
    for (int j = 0; j < num_partitions; ++j)
        acc += to_float(tmp_out_ptr[j * HEAD_SIZE + i])
               * shared_exp_sums[j] * inv_global_exp_sum;
    from_float(out_ptr[i], acc);
}
```

数学原理：
```
设分区 j 的 local max 为 m_j，全局 max 为 M = max(m_j)

分区 j 的部分输出：
  tmp_out_j = Σ_{i in partition_j} softmax_j(i) * V_i
            = Σ_{i in partition_j} [exp(qk_i - m_j) / Σ exp(qk_k - m_j)] * V_i

全局输出：
  out = Σ_j [exp(m_j - M) * exp_sum_j / global_exp_sum] * tmp_out_j
```

这是 FlashAttention 论文中 Online Softmax 的多分区扩展版本。

---

## 八、编译时特化（Template 展开）

```cpp
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          vllm::Fp8KVCacheDataType KV_DTYPE, bool IS_BLOCK_SPARSE,
          int NUM_THREADS = 128>
void paged_attention_v1_launcher(...) {
    switch (head_size) {
        case 32:  LAUNCH_PAGED_ATTENTION_V1(32); break;
        case 64:  LAUNCH_PAGED_ATTENTION_V1(64); break;
        case 128: LAUNCH_PAGED_ATTENTION_V1(128); break;
        // ...
    }
}
```

将 `HEAD_SIZE`、`BLOCK_SIZE`、`KV_DTYPE` 全部作为模板参数：

1. **HEAD_SIZE 编译时已知**：`NUM_VECS_PER_THREAD = HEAD_SIZE / THREAD_GROUP_SIZE` 等常量在编译时计算，数组大小确定，寄存器分配最优
2. **循环展开**：`#pragma unroll` 对编译时已知大小的循环展开，消除循环开销，增加 instruction-level parallelism
3. **死代码消除**：`if constexpr (KV_DTYPE == kAuto)` 在编译时选择代码路径，运行时无条件判断开销

**代价**：编译时间较长（每个 {T, CACHE_T, BLOCK_SIZE, HEAD_SIZE, KV_DTYPE} 组合都生成一个独立的内核）。注释中已说明 `// NOTE(woosuk): To reduce the compilation time, we only compile for the head sizes that we use`。

---

## 九、FP8 KV Cache 支持

```cpp
if constexpr (KV_DTYPE == Fp8KVCacheDataType::kAuto) {
    k_vecs[j] = *reinterpret_cast<const K_vec*>(k_ptr + ...);
} else {
    // FP8 → FP16/BF16 转换
    Quant_vec k_vec_quant = *reinterpret_cast<const Quant_vec*>(k_ptr + ...);
    k_vecs[j] = fp8::scaled_convert<K_vec, Quant_vec, KV_DTYPE>(k_vec_quant, *k_scale);
}
```

FP8 KV Cache 将 KV 从 BF16（2 bytes/element）压缩到 FP8（1 byte/element），节省 50% 的 KV Cache 内存，同时减少内存带宽消耗（Decode 阶段带宽敏感）。

加载时通过 `k_scale` / `v_scale` 进行反量化（FP8 × scale → FP16），计算仍用 FP16/BF16 精度。

---

## 十、block_sum：两级 warp reduce

```cpp
template <int NUM_WARPS>
inline __device__ float block_sum(float* red_smem, float sum) {
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;

    // 第一级：warp 内用 shuffle 归约（无共享内存，延迟低）
    for (int mask = WARP_SIZE / 2; mask >= 1; mask /= 2)
        sum += VLLM_SHFL_XOR_SYNC(sum, mask);

    // 跨 warp 通信（必须经过共享内存）
    if (lane == 0) red_smem[warp] = sum;
    __syncthreads();

    // 第二级：用第一个 warp 对各 warp 的结果再归约
    if (lane < NUM_WARPS) sum = red_smem[lane];
    for (int mask = NUM_WARPS / 2; mask >= 1; mask /= 2)
        sum += VLLM_SHFL_XOR_SYNC(sum, mask);

    return VLLM_SHFL_SYNC(sum, 0);  // 广播给所有线程
}
```

Warp shuffle（`__shfl_xor_sync`）是 GPU 上最快的线程间通信方式：**寄存器直传，无需共享内存，延迟仅 1~2 cycles**。只有跨 warp 时才需要通过共享内存中转（需要 `__syncthreads()`）。

这种两级设计将 reduce 操作对 `__syncthreads()` 的调用从 O(log N) 降为 O(1)，减少同步等待。

---

## 十一、关键设计模式总结

| 设计 | 原因 | 收益 |
|------|------|------|
| Key: `[..., head_size/x, block_size, x]` | 列访问合并 | 减少内存事务次数 |
| Value: `[..., head_size, block_size]` | 行访问合并 | 减少内存事务次数 |
| Thread Group 协作 | 1 token 的计算对齐到硬件 | 高效利用 warp |
| FP32 logits | softmax 数值精度 | 防止溢出/下溢 |
| 共享内存复用 | logits 和 output 不重叠 | 节省有限的共享内存 |
| `int64` block number | 大模型防止 int32 溢出 | 正确性保证 |
| 编译时模板特化 | HEAD_SIZE 编译时已知 | 循环展开，最优寄存器分配 |
| Online Softmax | 数值稳定性 | 防止 exp 溢出 |
| V2 两阶段 reduce | 超长序列并行化 | 大上下文场景线性扩展 |
| FP8 KV 支持 | KV Cache 内存紧张 | 节省 50% KV Cache 显存 |
| `if constexpr` 分支 | 编译时选择代码路径 | 无运行时 branch overhead |
| 尾部 block 清零 | 未初始化内存含 NaN | 防止 NaN 污染输出 |

---

*目录：`csrc/attention/`*
*更新：2026-03*
