# FlashAttention：原理与 vLLM 实现

> **核心问题**：标准 Attention 的内存用量是 O(N²)，序列长 4096 时就需要 64M 个元素的矩阵，
> 显存成为瓶颈，计算速度远低于硬件峰值。FlashAttention 如何打破这一限制？
>
> **论文**：[FlashAttention（Dao et al., NeurIPS 2022）](https://arxiv.org/abs/2205.14135)、
> [FlashAttention-2（Dao, ICLR 2024）](https://arxiv.org/abs/2307.08691)、
> [FlashAttention-3（Shah et al., 2024）](https://arxiv.org/abs/2407.08608)
> **代码路径**：`vllm/v1/attention/backends/flash_attn.py`

---

## 一、背景：标准 Attention 为什么慢

### 1.1 Attention 的数学定义

Transformer 自注意力的计算：

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d}}\right) V$$

用矩阵写出来：

```
Q: [N, d]    N = 序列长度，d = 每头的维度（如 128）
K: [N, d]
V: [N, d]

步骤 1：S = Q × Kᵀ           形状 [N, N]   ← 注意力得分矩阵
步骤 2：P = softmax(S / √d)   形状 [N, N]   ← 注意力权重
步骤 3：O = P × V             形状 [N, d]   ← 输出
```

### 1.2 内存壁垒：N² 的代价

```
序列长度 N = 4096，d = 128，BF16（2 bytes/元素）：

  S 矩阵（注意力得分）：4096 × 4096 × 2 bytes = 32 MB
  P 矩阵（注意力权重）：4096 × 4096 × 2 bytes = 32 MB

  单个头：64 MB
  LLaMA-3-8B（32 头）：64 MB × 32 = 2 GB！
  → 仅注意力中间矩阵就需要 2 GB 显存

  N = 32768（长上下文）：
  单头 S/P = 32768 × 32768 × 2 bytes = 2 GB
  32 头 = 64 GB → 超过单卡显存！
```

**根本原因**：这些中间矩阵必须全部写回 HBM（GPU 主内存），代价极高。

### 1.3 GPU 的内存层次

理解 FlashAttention 需要先理解 GPU 的存储层次：

```
GPU 内存层次（A100）：

  ┌──────────────────────────────────────────────────────────────────┐
  │   计算单元（SM，Streaming Multiprocessors，108 个）               │
  │                                                                   │
  │   ┌────────────────────────────────────────────────────────────┐ │
  │   │  寄存器（Register）每个 SM ~256KB，延迟 < 1 cycle          │ │
  │   └────────────────────────────────────────────────────────────┘ │
  │                                                                   │
  │   ┌────────────────────────────────────────────────────────────┐ │
  │   │  SRAM / 共享内存（Shared Memory）每个 SM 192KB，延迟 ~5 cy │ │
  │   │  所有 SM 合计 ≈ 108 × 192KB ≈ 20 MB                       │ │
  │   └────────────────────────────────────────────────────────────┘ │
  │                 ↕  带宽 19.5 TB/s（寄存器 ↔ SRAM）              │
  │   ┌────────────────────────────────────────────────────────────┐ │
  │   │  HBM（High Bandwidth Memory，"显存"）80 GB，延迟 ~200 cy   │ │
  │   │  带宽：2 TB/s                                              │ │
  │   └────────────────────────────────────────────────────────────┘ │
  └──────────────────────────────────────────────────────────────────┘

关键：
  SRAM 带宽是 HBM 的 10×，但容量只有 HBM 的 1/4000
  运算单元的峰值算力（312 TFLOP/s）远超 HBM 带宽
  → GPU 是"内存带宽受限"的，计算单元常常在等数据
```

### 1.4 标准 Attention 的 I/O 分析

```
标准 Attention 的内存读写（N=4096，d=128，单头，BF16）：

  读 Q、K：2 × N × d × 2 bytes = 2 × 4096 × 128 × 2 = 2 MB
  写 S = QKᵀ：N × N × 2 bytes = 32 MB              ← HBM 写
  读 S（做 softmax）：32 MB                          ← HBM 读
  写 P（softmax 结果）：32 MB                        ← HBM 写
  读 P、V（做 PV）：32 MB + 1 MB                    ← HBM 读
  写 O：N × d × 2 bytes = 1 MB

  总 HBM I/O ≈ 130 MB（per head，per layer）

  LLaMA-3-8B（32 层，32 头）：
  130 MB × 32 层 × 32 头 = 133 GB 的 HBM 访问！
  → 以 2 TB/s 带宽，光读写就需要 66ms
  → 实际算术操作（GEMM）只需 <5ms
  → 利用率 < 10%！

问题：GPU 99% 的时间在等内存，不在算数。
```

---

## 二、FlashAttention 的核心原理

### 2.1 关键洞察：避免写回 N² 矩阵

FlashAttention 的核心思想：**不物化（materialize）N×N 的 S 和 P 矩阵，在 SRAM 内完成所有计算**。

```
标准 Attention（HBM-bound）：
  HBM → Q,K,V → SRAM → 算 QKᵀ → HBM（写 S）
  HBM → S → SRAM → 算 softmax → HBM（写 P）
  HBM → P,V → SRAM → 算 PV → HBM（写 O）
  三轮 HBM I/O

FlashAttention（SRAM-bound）：
  HBM → Q 分块, K 分块, V 分块 → SRAM
  SRAM 内：算 QKᵀ + softmax + PV（全部在 SRAM 里！）
  SRAM → 最终 O → HBM
  一轮 HBM I/O
```

### 2.2 问题：Softmax 需要全局信息

直接分块的障碍：**softmax 需要对整行求和**，但分块时只能看到部分 K。

```
标准 softmax：
  P[i,j] = exp(S[i,j]) / Σₖ exp(S[i,k])
                          ↑
                     需要整行 K 的结果才能归一化！
```

如果我们把 K 分成 2 块（K₁ 和 K₂），先算第一块：

```
第一块：S₁ = Q × K₁ᵀ   → 得到部分分数，无法做准确的 softmax
        P₁ = softmax(S₁)  ← 这个 softmax 是错的！因为没看到 K₂
```

### 2.3 Online Softmax：分块计算的数学基础

FlashAttention 用**在线（online）方式**来分块计算 softmax。

**关键技巧：Softmax 的数值稳定版本 + 增量更新**

标准 softmax 的数值稳定形式：
```
m = max(S[i,:])   ← 减去最大值，防止 exp 溢出
P[i,j] = exp(S[i,j] - m) / Σₖ exp(S[i,k] - m)
```

**分两块时的推导**：

```
第一块（K₁ 覆盖 j ∈ [0, N/2)）：
  m₁ = max(S₁[i,:])              ← 第一块的最大值
  ℓ₁ = Σⱼ exp(S₁[i,j] - m₁)    ← 第一块的归一化因子
  O₁ = Σⱼ (exp(S₁[i,j] - m₁) / ℓ₁) × V₁[j,:]  ← 第一块的输出

第二块（K₂ 覆盖 j ∈ [N/2, N)）：
  m₂ = max(S₂[i,:])              ← 第二块的最大值
  ℓ₂ = Σⱼ exp(S₂[i,j] - m₂)

  全局最大值：m = max(m₁, m₂)
  修正因子：
    e₁ = exp(m₁ - m)             ← 第一块相对全局最大值的修正
    e₂ = exp(m₂ - m)             ← 第二块相对全局最大值的修正
  全局归一化因子：
    ℓ = ℓ₁ × e₁ + ℓ₂ × e₂

  最终输出（合并两块）：
    O = (O₁ × ℓ₁ × e₁ + O₂_unnorm × e₂) / ℓ

其中 O₂_unnorm = Σⱼ exp(S₂[i,j] - m₂) × V₂[j,:]（第二块未归一化的输出）
```

**这个公式的意义**：
- 只需保存 `m`（最大值）和 `ℓ`（归一化因子），不需要保存整个 P 矩阵
- 每次看到新的 K 块，就更新这两个标量，并修正已有的输出 O
- 最终 O 正确等价于对整行做 softmax 后的结果

### 2.4 分块算法（Tiling）

```
FlashAttention 前向传播：

输入：Q [N, d]，K [N, d]，V [N, d]（在 HBM 中）
SRAM 块大小：Bc = Br = ⌈SRAM_size / (4 × d)⌉

# 把 K, V 分成列块（每块 Bc 列）
for j = 1 to N/Bc:                        # 遍历 K/V 块
    从 HBM 加载 K_j、V_j 到 SRAM          # ← HBM 读（小块）

    # 把 Q 分成行块（每块 Br 行）
    for i = 1 to N/Br:                    # 遍历 Q 块
        从 HBM 加载 Q_i 到 SRAM           # ← HBM 读（小块）
        从 HBM 加载 O_i、ℓ_i、m_i        # ← 上一轮的累积状态

        # 在 SRAM 中计算（无 HBM 访问！）
        S_ij = Q_i × K_jᵀ               # [Br, Bc]
        m̃ = row_max(S_ij)               # [Br]
        P̃ = exp(S_ij - m̃)              # [Br, Bc]，未归一化
        ℓ̃ = row_sum(P̃)                 # [Br]

        # 修正历史累积值（online softmax 的核心）
        m_new = max(m_i, m̃)
        ℓ_new = exp(m_i - m_new) × ℓ_i + exp(m̃ - m_new) × ℓ̃
        O_i = (O_i × exp(m_i - m_new) × ℓ_i + P̃ × V_j) / ℓ_new

        # 写回 HBM（只写小块）
        写 O_i、ℓ_i = ℓ_new、m_i = m_new 到 HBM  # ← HBM 写（小块）

# 最终 O 已经正确归一化，无需额外 pass
```

**复杂度对比**：

| | 标准 Attention | FlashAttention |
|--|--|--|
| HBM 读写量 | O(N²) | O(N²/M) × M = O(N) \* |
| SRAM 需求 | O(N²) | O(N × M)，M = SRAM 大小 |
| 数值精度 | 完全精确 | 完全精确（不是近似！） |
| 额外 pass 数 | 3（QKᵀ、softmax、PV）| 1（融合） |

\* FlashAttention 的 HBM 读写量从 O(N²) 降到 O(N × d)，省去了 N×N 矩阵的读写

### 2.5 Recomputation（重计算）：反向传播的内存优化

```
标准反向传播：
  前向时：保存 P 矩阵（[N, N]）用于反向计算 ∂L/∂Q 和 ∂L/∂K
  内存代价：O(N²)

FlashAttention 反向传播：
  前向时：只保存 O 和 ℓ（归一化因子，大小 O(N)）
  反向时：重新计算 S 和 P（从 Q、K、V 开始，在 SRAM 中计算）

  代价：额外的前向计算时间
  收益：内存从 O(N²) 降到 O(N)

  实际上：重计算的时间代价 < 反复读写 HBM 的时间代价
  → 总体上既省内存又省时间（内存带宽是瓶颈，不是算力）
```

---

## 三、FlashAttention 1/2/3 的演进

### 3.1 FlashAttention-1（2022）：奠定基础

核心贡献：
- 提出分块（tiling）+ online softmax 消除 N² 中间矩阵
- IO 复杂度分析框架
- 速度提升：2-4×（相比 PyTorch naive attention）
- 内存减少：5-20×

局限：
- GPU 利用率仍较低（~30%）
- 反向传播效率不足

### 3.2 FlashAttention-2（2023）：优化计算效率

主要改进：

```
① 减少非矩阵乘法（non-matmul）的操作量
  原因：现代 GPU 的矩阵乘法（Tensor Core）算力远高于标准算术
        A100 的 BF16 GEMM = 312 TFLOP/s
        A100 的 BF16 其他算术 ≈ 19.5 TFLOP/s（低 16×）
  方法：重排循环，减少 rescale 操作（exp、除法次数）

② 更好的并行策略
  FA-1：只对 batch × heads 维度并行（内循环是串行的）
  FA-2：query 维度也并行（不同 query 块分到不同线程块）
  → GPU 线程利用率从 ~30% 提升到 ~70%

③ 减少共享内存（SRAM）的读写
  合并操作，减少 SRAM 内部的数据搬运
  → warps 间通信减少

性能：速度再提升 2×（相比 FA-1），GPU 利用率约 70%
```

### 3.3 FlashAttention-3（2024）：硬件深度优化

针对 H100（Hopper 架构）的专项优化：

```
H100 的新特性：
  ① 异步内存拷贝（TMA，Tensor Memory Accelerator）
     可以在 GPU 计算的同时异步从 HBM 搬运数据
  ② WGMMA 指令（Warpgroup Matrix Multiply Accumulate）
     比 A100 的 HMMA 更宽的矩阵乘法指令
  ③ 低精度：FP8（4 bytes → 2 bytes → 1 byte）

FA-3 的优化：
  ① 计算与数据搬运的流水线并行（Pipelining）
       时刻 t:   计算 Q_i × K_jᵀ
       时刻 t+1: 计算 P × V（用前一步结果）+ 异步加载 K_{j+1}
       → 内存等待被计算填满，利用率进一步提升

  ② 低精度（FP8）支持
       Q/K/V 以 FP8 存储和计算，精度损失可接受
       → 内存减半，算力翻倍（FP8 GEMM = 2× FP16）

  ③ softmax 与矩阵乘法解耦
       利用 H100 的 2-stage pipeline 将 exp 计算与 GEMM 并行

性能：GPU 利用率约 75%（H100，FP16）
```

**不同版本在 A100 上的对比**：

```
序列长度 4096，batch=4，heads=16，dim=128，BF16：

                    FLOPs/s    相对 FA-1
  Naive Attention    10 TFLOP   0.35×
  FlashAttention-1   28 TFLOP   1.0×（基准）
  FlashAttention-2   56 TFLOP   2.0×
  FlashAttention-3   84 TFLOP   3.0×（H100）
  理论峰值          312 TFLOP   11×
```

---

## 四、Causal Mask 的实现

### 4.1 为什么需要 Causal Mask

```
语言模型生成时，位置 i 的 token 不能看到位置 i+1 以后的 token：

  [A] [B] [C] [D] [E]   ← 5 个 token
   ↓   ↓   ↓   ↓   ↓
  A   AB  ABC ABCD ABCDE  ← 每个 token 能访问的历史

注意力矩阵（下三角）：
       A  B  C  D  E
  A: [ 1  0  0  0  0 ]   ← A 只能看自己
  B: [ 1  1  0  0  0 ]   ← B 能看 A 和 B
  C: [ 1  1  1  0  0 ]   ← C 能看 A、B、C
  D: [ 1  1  1  1  0 ]
  E: [ 1  1  1  1  1 ]
```

### 4.2 FlashAttention 的 Causal Mask 实现

```
分块计算时，mask 的处理：

  Q 块 i（行 [i×Br, (i+1)×Br)）：
  K 块 j（列 [j×Bc, (j+1)×Bc)）：

  情况 1：整块都是过去（j×Bc < i×Br）
    → 所有位置都可以 attend，无需 mask
    → 完整计算 S_ij，效率最高

  情况 2：整块都是未来（j×Bc > (i+1)×Br）
    → 所有位置都被 mask，直接跳过
    → 不需要计算！

  情况 3：对角线上的块（j×Bc ≈ i×Br）
    → 需要逐元素检查位置关系
    → 在 SRAM 内施加 mask（将未来位置设为 -inf）

FA-2 对 causal mask 的优化：
  对角线块（情况 3）只有 N/Br 个，约占总块数的 1/Br
  → 大多数块走情况 1 或 2，快速路径
  → 实际开销 < 5%
```

---

## 五、vLLM 中的 FlashAttention 实现

### 5.1 架构全图

```
vllm/v1/attention/
├── backends/
│   ├── flash_attn.py       ← 主要后端（CUDA/NVIDIA GPU）
│   ├── flashmla.py         ← MLA（DeepSeek 压缩 KV）后端
│   ├── flex_attention.py   ← PyTorch FlexAttention 后端
│   └── utils.py            ← CommonAttentionMetadata 定义
│
└── layer.py                ← Attention 层（所有后端的统一入口）

vllm/attention/utils/
└── fa_utils.py             ← FA2/FA3 版本选择，函数导入

vllm/vllm_flash_attn/       ← vLLM 自己的 Flash Attention CUDA kernel
```

### 5.2 Flash Attention 函数签名

```python
# vllm/attention/utils/fa_utils.py
from vllm.vllm_flash_attn import flash_attn_varlen_func

flash_attn_varlen_func(
    q,                    # [total_tokens, num_heads, head_size]     Query 张量
    k,                    # KV Cache 的 K 部分（完整 cache）
    v,                    # KV Cache 的 V 部分（完整 cache）
    out,                  # [total_tokens, num_heads, head_size]     输出张量

    # === 变长序列的核心参数 ===
    cu_seqlens_q,         # [batch_size + 1] int32，Query 的累积长度
    seqused_k,            # [batch_size] int32，每个请求的完整 KV 长度
    max_seqlen_q,         # int，max(每请求 query 长度)
    max_seqlen_k,         # int，max(每请求 KV 总长度)

    # === Attention 参数 ===
    softmax_scale,        # float，通常 = 1/√d
    causal,               # bool，是否因果 mask

    # === Paged KV Cache ===
    block_table,          # [batch_size, max_blocks_per_req] int32

    # === 可选 ===
    alibi_slopes,         # ALiBi 位置编码斜率
    window_size,          # [left, right] 滑动窗口
    softcap,              # Gemma 的 logit soft-capping
    scheduler_metadata,   # FA3 专属：AOT 调度元数据
    fa_version,           # 2 或 3
    q_descale,            # FP8 量化的反量化因子
    k_descale,
    v_descale,
)
```

**"varlen"** 表示 variable-length：一次调用可以处理不同长度的序列，对应 Continuous Batching 中 prefill + decode 混合的需求。

### 5.3 KV Cache 的存储格式

```python
# KV Cache 张量的物理布局
# 形状：[2, num_blocks, block_size, num_kv_heads, head_size]
#        ↑  ↑            ↑           ↑              ↑
#        K/V 块索引    块内位置     KV 头数          头维度

# 解包
key_cache, value_cache = kv_cache.unbind(0)
# key_cache:   [num_blocks, block_size, num_kv_heads, head_size]
# value_cache: [num_blocks, block_size, num_kv_heads, head_size]

# Block Table：[batch_size, max_blocks_per_req]
# block_table[i, j] = 第 i 个请求的第 j 个物理块的 block_id

# FA kernel 内部的访问逻辑（伪代码）：
# 对于请求 i，位置 pos：
#   block_idx = pos // block_size
#   block_offset = pos % block_size
#   physical_block = block_table[i, block_idx]
#   k = key_cache[physical_block, block_offset, :, :]
```

### 5.4 写入新 KV 到 Cache：reshape_and_cache_flash

在注意力计算**之前**，先把本步新计算的 K/V 写入 KV Cache：

```python
# vllm/v1/attention/backends/flash_attn.py
from vllm._custom_ops import reshape_and_cache_flash

# 1. 写入新 KV（散射写，slot_mapping 指定物理位置）
reshape_and_cache_flash(
    key,           # [total_tokens, num_kv_heads, head_size]（本步新 KV）
    value,
    key_cache,     # [num_blocks, block_size, num_kv_heads, head_size]（目标）
    value_cache,
    slot_mapping,  # [total_tokens]，每个 token → 物理槽位号
    kv_cache_dtype,
    k_scale,
    v_scale,
)
# slot_mapping[i] = block_id × block_size + offset_within_block

# 2. 然后用完整的 KV Cache 做 attention（包含历史 + 本步）
flash_attn_varlen_func(q, key_cache, value_cache, ..., block_table=block_table)
```

### 5.5 FlashAttentionMetadata：完整字段

```python
# vllm/v1/attention/backends/flash_attn.py

@dataclass
class FlashAttentionMetadata:
    # === 基本信息 ===
    num_actual_tokens: int          # 本 batch 的 token 总数（不含 padding）
    max_query_len: int              # max(每请求本步 query 长度)
    query_start_loc: torch.Tensor   # [batch_size + 1]，query 起始位置（累计和）
    max_seq_len: int                # max(每请求完整序列长度)
    seq_lens: torch.Tensor          # [batch_size]，每请求完整序列长度
    block_table: torch.Tensor       # [batch_size, max_blocks]，KV Cache 块表
    slot_mapping: torch.Tensor      # [total_tokens]，KV Cache 写入位置

    # === Cascade Attention（共享前缀优化）===
    use_cascade: bool               # 是否启用 Cascade Attention
    common_prefix_len: int          # 所有请求共享的前缀长度
    cu_prefix_query_lens: Optional  # 前缀阶段的 query 累积长度
    prefix_kv_lens: Optional        # 前缀阶段每请求的 KV 长度
    suffix_kv_lens: Optional        # 后缀阶段每请求的 KV 长度

    # === FA3 专属：AOT 调度 ===
    scheduler_metadata: Optional    # 提前计算的调度元数据（减少 kernel launch 开销）
    prefix_scheduler_metadata: Optional
    max_num_splits: int = 0         # CUDAGraph 下的分片数

    # === 其他 ===
    causal: bool = True             # 是否因果 mask
```

### 5.6 从 CommonAttentionMetadata 到 FlashAttentionMetadata 的转换

```python
# vllm/v1/attention/backends/flash_attn.py FlashAttentionMetadataBuilder.build()

def build(self, common_attn_metadata):
    query_start_loc = common_attn_metadata.query_start_loc  # 已在 GPU 上
    seq_lens        = common_attn_metadata.seq_lens
    num_reqs        = common_attn_metadata.num_reqs
    num_actual_tokens = common_attn_metadata.num_actual_tokens

    # 获取 Block Table 和 Slot Mapping（来自 KVCacheManager 分配的结果）
    block_table = input_batch.block_table.get_device_tensor()[:num_reqs]
    slot_mapping = input_batch.block_table.slot_mapping[:num_actual_tokens]

    # 判断是否启用 Cascade Attention
    use_cascade, common_prefix_len = self._check_cascade(...)

    # 构建 FA3 调度器元数据（如果使用 FA3）
    scheduler_metadata = None
    if fa_version == 3:
        scheduler_metadata = get_scheduler_metadata(
            batch_size=num_reqs,
            seqlens_q=num_scheduled_tokens,
            seqlens_k=seq_lens,
            ...
        )

    return FlashAttentionMetadata(
        num_actual_tokens=num_actual_tokens,
        max_query_len=common_attn_metadata.max_query_len,
        query_start_loc=query_start_loc,
        max_seq_len=common_attn_metadata.max_seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=use_cascade,
        common_prefix_len=common_prefix_len,
        scheduler_metadata=scheduler_metadata,
        causal=True,
    )
```

### 5.7 前向传播：完整调用链

```python
# vllm/v1/attention/backends/flash_attn.py FlashAttentionImpl.forward()

def forward(self, query, key, value, kv_cache, attn_metadata, ...):

    # ① KV Cache 解包
    key_cache, value_cache = kv_cache.unbind(0)

    # ② 写入新 KV（本步计算的 K/V 散射到物理槽位）
    reshape_and_cache_flash(
        key, value,
        key_cache, value_cache,
        attn_metadata.slot_mapping,
        ...
    )

    # ③ 注意力计算（两种路径）
    if attn_metadata.use_cascade:
        # Cascade Attention：共享前缀 + 各自后缀，分两次 FA 调用
        output = self._run_cascade_attention(query, key_cache, value_cache, attn_metadata)
    else:
        # 普通路径：单次 FA 调用处理所有请求
        flash_attn_varlen_func(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=attn_metadata.query_start_loc,
            seqused_k=attn_metadata.seq_lens,
            max_seqlen_q=attn_metadata.max_query_len,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=self.scale,
            causal=attn_metadata.causal,
            block_table=attn_metadata.block_table,
            scheduler_metadata=attn_metadata.scheduler_metadata,
            fa_version=self.fa_version,
        )

    return output
```

---

## 六、Prefill + Decode 混合处理：一次调用的细节

这是 Continuous Batching 的核心——如何用一次 `flash_attn_varlen_func` 同时处理两种请求？

### 6.1 参数构造示例

```
Batch（3 个请求）：
  Req A：prefill，本步 10 个 token，序列总长 10（纯 prefill）
  Req B：decode，本步 1 个 token，序列总长 151（150 历史 + 1 新）
  Req C：prefill，本步 5 个 token，序列总长 5（纯 prefill）

构建参数：

  cu_seqlens_q = [0, 10, 11, 16]
  # Req A 的 query: input[0:10]
  # Req B 的 query: input[10:11]（只有 1 个 token）
  # Req C 的 query: input[11:16]

  seqused_k = [10, 151, 5]
  # Req A：FA kernel 会 attend 10 个 KV（Q 和 KV 大小相同 = 纯 prefill）
  # Req B：FA kernel 会 attend 151 个 KV（1 个 query，151 个历史 KV）
  # Req C：FA kernel 会 attend 5 个 KV

  max_seqlen_q = 10   # max(10, 1, 5)
  max_seqlen_k = 151  # max(10, 151, 5)

  causal = True（对所有请求统一）
```

### 6.2 Causal Mask 在混合 Batch 中的行为

```
Req A（prefill，10 tokens）：
  causal=True → 下三角 mask
  注意力矩阵：10×10 下三角
  token 0 只看 token 0
  token 9 看 token 0-9

Req B（decode，1 token）：
  causal=True → 但 query 只有 1 个 token
  注意力矩阵：1×151（向量）
  唯一的 query token 可以看所有 151 个历史 KV
  → 退化为"全局 attend"（causal 无实际效果）

Req C（prefill，5 tokens）：
  注意力矩阵：5×5 下三角
  与 Req A 类似

FA 内部：
  varlen 机制让每个请求独立处理自己的 mask
  不同请求之间完全隔离（不会 attend 到其他请求的 token）
```

---

## 七、Cascade Attention：共享前缀优化

### 7.1 场景

```
所有请求都有相同的 system prompt（如 2048 个 token）：
  Req 0: [system prompt 2048 tok][user A  128 tok] → decode token
  Req 1: [system prompt 2048 tok][user B   64 tok] → decode token
  Req 2: [system prompt 2048 tok][user C  256 tok] → decode token
  ...
  Req 31: [system prompt 2048 tok][user Z   96 tok] → decode token

  common_prefix_len = 2048
  后缀长度各不同：128, 64, 256, ..., 96
```

### 7.2 算法

```
标准 FlashAttention（不启用 Cascade）：
  每个请求独立 attend 全部 KV
  Req 0：attend 2048+128 = 2176 个 KV
  Req 1：attend 2048+64  = 2112 个 KV
  ...
  → 32 个请求各自处理 ~2100 个 KV，大量重复计算

Cascade Attention：
  步骤 1：所有请求共同处理前缀 KV（共享 Q 对 prefix KV）
    flash_attn_func(
        q=all_queries,
        k=prefix_key_cache,
        v=prefix_value_cache,
        seqused_k=[2048] × 32,   # 每个请求都 attend 相同的 2048 前缀
        causal=False,             # 前缀对所有 query 都是"过去"，无需因果 mask
        return_softmax_lse=True,  # 返回 log-sum-exp（用于合并）
    )
    → prefix_output [32, d]，prefix_lse [32]

  步骤 2：每个请求处理自己的后缀 KV
    flash_attn_varlen_func(
        q=all_queries,
        k=suffix_key_cache,
        v=suffix_value_cache,
        seqused_k=[128, 64, 256, ..., 96],  # 后缀长度各不同
        causal=True,
        return_softmax_lse=True,
    )
    → suffix_output [32, d]，suffix_lse [32]

  步骤 3：合并两部分（基于 LSE 重新加权）
    merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse)
    # 公式：
    # lse_total = log(exp(prefix_lse) + exp(suffix_lse))
    # output = (prefix_output × exp(prefix_lse - lse_total)
    #         + suffix_output × exp(suffix_lse - lse_total))
```

### 7.3 性能分析

```
何时 Cascade Attention 更快？

  判断条件（vllm/v1/attention/backends/flash_attn.py _check_cascade()）：
    common_prefix_len ≥ 256
    num_reqs ≥ 8

  性能优势：
    标准方式：每个请求独立 attend 全部 2176 个 KV = 32 × 2176 次操作
    Cascade：前缀 2048 × 32（步骤 1）+ 后缀 ≈96 × 32（步骤 2）
             ≈ 2048×32 + 96×32 = 2144×32（但步骤 1 的前缀无因果 mask，效率更高）

    当后缀很短时（decode 阶段），Cascade 节省的是重复的前缀计算
    common_prefix_len/total_seq_len 越大，收益越大

  何时不启用：
    common_prefix_len 很短（< 256）：合并开销 > 收益
    请求数很少（< 8）：并行优势不明显
```

---

## 八、FA2 vs FA3 版本选择

```python
# vllm/attention/utils/fa_utils.py

def get_flash_attn_version():
    device_capability = torch.cuda.get_device_capability()

    # H100（SM90）默认用 FA3
    if device_capability.major == 9 and is_fa_version_supported(3):
        fa_version = 3
    else:
        # A100（SM80）等用 FA2
        fa_version = 2

    # 可被环境变量覆盖
    if VLLM_FLASH_ATTN_VERSION is not None:
        fa_version = VLLM_FLASH_ATTN_VERSION

    return fa_version
```

**FA3 的独有特性**：

```
scheduler_metadata（AOT 调度）：
  在 _prepare_inputs() 阶段提前计算调度信息
  → kernel launch 时无需在 GPU 上计算调度
  → 减少 kernel 启动延迟（对于小 batch 的 decode 阶段特别重要）

  get_scheduler_metadata(
      batch_size,
      seqlens_q,     # 每请求 query 长度
      seqlens_k,     # 每请求 KV 总长度
      ...
  ) → scheduler_metadata（GPU tensor）

FP8 支持（q_descale/k_descale/v_descale）：
  Flash Attention 直接接受 FP8 输入
  反量化因子以 tensor 形式传入，在 kernel 内部使用
  → 减少 I/O（FP8 比 BF16 小 2×），同等精度（Hopper FP8 GEMM 精度可接受）
```

---

## 九、GQA（Grouped Query Attention）的处理

### 9.1 GQA 的内存优势

```
标准 MHA（Multi-Head Attention）：
  Q 头数 = K 头数 = V 头数 = 32（LLaMA-3-8B）

GQA（Grouped Query Attention，LLaMA-3 使用）：
  Q 头数 = 32
  K 头数 = V 头数 = 8（4 个 Q 头共享 1 个 K/V 头）

KV Cache 大小：
  MHA：32 × 128 × 2 bytes = 8 KB/token/���
  GQA：8 × 128 × 2 bytes  = 2 KB/token/层（减少 4×！）
```

### 9.2 Flash Attention 对 GQA 的原生支持

```python
# Flash Attention varlen 函数天然支持 GQA
# q: [total_tokens, num_q_heads, head_dim]    num_q_heads = 32
# k: [num_blocks, block_size, num_kv_heads, head_dim]  num_kv_heads = 8
# v: 同 k

# FA kernel 内部自动处理：
# Q head i 使用 K/V head (i // (num_q_heads / num_kv_heads))
# 即 Q heads {0,1,2,3} 共享 KV head 0
#    Q heads {4,5,6,7} 共享 KV head 1
#    ...

# 无需在外部展开或复制 KV
```

---

## 十、性能数字与实际效果

### 10.1 内存节省

```
序列长度 4096，LLaMA-3-8B（32 层，32 heads，GQA 8 KV heads）：

  标准 Attention 的中间矩阵（S + P）：
    32 层 × 32 heads × 2 × 4096 × 4096 × 2 bytes ≈ 34 GB！
    → 根本无法运行

  FlashAttention：
    不存储 S/P 矩阵
    只需 O 矩阵：32 层 × 32 heads × 4096 × 128 × 2 bytes = 1 GB
    → 正常运行
```

### 10.2 速度提升

```
在 A100（BF16，causal，N=2048，d=128）的 Attention 延迟：

  PyTorch Naive：~8ms（内存带宽受限）
  FlashAttention-1：~2ms
  FlashAttention-2：~1ms
  理论下限：~0.4ms（算力受限）

  FlashAttention-2 MFU（模型浮点利用率）≈ 73%（H100）
  → 接近硬件极限
```

### 10.3 vLLM 中 FA 的实际作用

```
vLLM 推理（LLaMA-3-70B，batch=32，A100-80G）：

  单步 forward 时间分布：
    GEMM（Linear 层）：~30ms
    FlashAttention（所有层）：~15ms
    其他（RMSNorm、激活、采样）：~3ms
    总计：~48ms

  如果用标准 Attention（假设 N=1024 平均长度）：
    Attention 单独就需要 ~100ms（IO bound）
    总计 >130ms

  FlashAttention 的贡献：
  ① 速度：Attention 从 100ms → 15ms（7×）
  ② 内存：释放数十 GB 的中间矩阵显存，可用于 KV Cache
  ③ 长上下文：使 N=32K/128K 的推理成为可能
```

---

## 十一、总结

```
FlashAttention 的本质：把 Attention 从"内存带宽受限"变成"算力受限"

核心技术：
  1. 分块（Tiling）：在 SRAM 内完成计算，避免 N² 矩阵写入 HBM
  2. Online Softmax：数学等价的增量 softmax，无需全局预知
  3. Recomputation：用少量重计算换取大量内存节省

演进历史：
  FA-1（2022）：奠定基础，2-4× 加速
  FA-2（2023）：优化并行度和 SRAM 使用，再 2× 加速
  FA-3（2024）：H100 专属，流水线并行、FP8 支持，再 1.5×

vLLM 的应用：
  1. flash_attn_varlen_func：变长序列，一次调用处理混合 batch
  2. Paged KV Cache + Block Table：与 PagedAttention 无缝结合
  3. Cascade Attention：共享前缀的二段式计算
  4. FA3 AOT Scheduling：提前调度，减少 kernel launch 开销
  5. FP8 支持：H100 上进一步提速

最终效果：
  内存：N² 中间矩阵 → 消除
  速度：Attention 计算 7-10× 加速
  上下文：使 32K/128K 长序列推理成为可能
```

| 技术 | 解决的问题 | 关键点 |
|------|-----------|--------|
| 分块（Tiling）| N² 矩阵占满 HBM | 在 SRAM 内计算，分块写回 |
| Online Softmax | Softmax 需要全局求和 | 维护 m（最大值）和 ℓ（归一化因子），增量更新 |
| Recomputation | 反向传播需保存 P 矩阵 | 只保存 O 和 ℓ，反向时重算 S/P |
| varlen_func | Continuous Batching 中混合请求 | cu_seqlens_q 分割不同长度的请求 |
| Block Table | PagedAttention 非连续 KV | FA kernel 内部按块表查询物理块 |
| Cascade Attention | 重复计算共享前缀 | 前缀一次，后缀分别，LSE 合并 |
| FA3 | H100 利用率不够高 | TMA 异步、WGMMA、FP8、AOT 调度 |

---

*参考资料：*
- *[FlashAttention（Dao et al., NeurIPS 2022）](https://arxiv.org/abs/2205.14135)*
- *[FlashAttention-2（Dao, ICLR 2024）](https://arxiv.org/abs/2307.08691)*
- *[FlashAttention-3（Shah et al., 2024）](https://arxiv.org/abs/2407.08608)*
- *[Online Softmax（Milakov & Gimelshein, 2018）](https://arxiv.org/abs/1805.02867)*
- *[vLLM 源码：flash_attn.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/attention/backends/flash_attn.py)*
- *[vLLM 源码：fa_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/attention/utils/fa_utils.py)*
*更新：2026-03*
