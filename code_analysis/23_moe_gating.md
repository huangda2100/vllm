# vLLM 中的门控网络与 MoE（Mixture of Experts）

## 目录
1. [MoE 基础原理](#1-moe-基础原理)
2. [门控网络数学推导](#2-门控网络数学推导)
3. [vLLM MoE 架构概览](#3-vllm-moe-架构概览)
4. [标准 Softmax 门控（Mixtral 风格）](#4-标准-softmax-门控mixtral-风格)
5. [分组 TopK 门控（DeepSeek 风格）](#5-分组-topk-门控deepseek-风格)
6. [带纠正偏置的门控（noaux_tc）](#6-带纠正偏置的门控noaux_tc)
7. [共享专家（Shared Experts）](#7-共享专家shared-experts)
8. [FusedMoE 前向传播流程](#8-fusedmoe-前向传播流程)
9. [CUDA/Triton 内核实现](#9-cudatriton-内核实现)
10. [专家并行（Expert Parallelism）](#10-专家并行expert-parallelism)
11. [专家并行负载均衡（EPLB）](#11-专家并行负载均衡eplb)
12. [量化支持](#12-量化支持)
13. [MoE vs Dense 对比](#13-moe-vs-dense-对比)

---

## 1. MoE 基础原理

### 1.1 动机

标准 Transformer 的 FFN 层对**每个 token** 激活全部参数：

```
Dense FFN:  output = FFN(x)          参数量 = H × I × 2
MoE FFN:    output = Σ gate_i × FFN_i(x)   参数量 = E × H × I × 2
```

MoE 的核心洞察：**扩大参数量，但保持计算量恒定**。

| 模型 | 参数量 | 激活参数 | 专家数 | Top-K |
|------|--------|---------|-------|-------|
| LLaMA-3-8B（Dense） | 8B | 8B | - | - |
| Mixtral-8x7B | ~47B | ~13B | 8 | 2 |
| DeepSeek-V2 | ~236B | ~21B | 160 | 6 |
| DeepSeek-V3 | ~671B | ~37B | 256 | 8 |
| Qwen2-57B-A14B | 57B | 14B | 64 | 8 |

**收益**：
- 推理计算量 ≈ Dense (active_params/total_params 小，通信代价可忽略)
- 模型容量大幅增加 → 更强的"记忆"能力
- 不同 token 由不同专家处理 → 隐式的功能专门化

### 1.2 MoE 层结构

```
输入: x ∈ R^{B×H}   (B=批次token数, H=hidden_size)
         │
         ▼
┌─────────────────┐
│  Gate Network   │  Linear(H, E) → router_logits ∈ R^{B×E}
│  (门控网络)      │
└────────┬────────┘
         │ Top-K 选择
         ▼
    topk_ids    ∈ Z^{B×K}   (每个token选中的K个专家ID)
    topk_weights ∈ R^{B×K}  (对应的门控权重)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Experts (E个FFN，每个结构相同但参数独立)                  │
│                                                         │
│  Expert_i: Gate(x) → SiLU → Up(x) → Down(x)            │
│  （SwiGLU结构）                                          │
└────────┬────────────────────────────────────────────────┘
         │ 加权求和
         ▼
输出: y = Σ_{i∈TopK(x)} w_i × Expert_i(x)   ∈ R^{B×H}
```

---

## 2. 门控网络数学推导

### 2.1 标准 Softmax TopK 门控

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

### 2.2 分组 TopK 门控（DeepSeek-V2/V3）

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

### 2.3 noaux_tc 纠正偏置门控

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

## 3. vLLM MoE 架构概览

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

## 4. 标准 Softmax 门控（Mixtral 风格）

### 4.1 Mixtral 模型结构

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

### 4.2 专家权重存储格式

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

### 4.3 ���个专家的 FFN 结构（SwiGLU）

```
x ∈ R^H
  │
  ├── Gate proj: x @ W1  → gate ∈ R^I
  └── Up proj:   x @ W3  → up   ∈ R^I
        │
        ▼
  SiLU(gate) × up        ��� hidden ∈ R^I   （SwiGLU 激活）
        │
        ▼
  Down proj: hidden @ W2  → output ∈ R^H
```

---

## 5. 分组 TopK 门控（DeepSeek 风格）

### 5.1 DeepSeek-V2 MoE 结构

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

### 5.2 grouped_topk() 实现

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

## 6. 带纠正偏置的门控（noaux_tc）

### 6.1 fused_topk_bias()

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

### 6.2 偏置学习机制

`e_score_correction_bias` 在训练时通过梯度下降学习：
- 若专家 i 被过度选择（负载高），其偏置逐渐变小（降低被选概率）
- 若专家 i 很少被选择（负载低），其偏置逐渐变大（提高被选概率）
- 偏置与主任务梯度**分离**，不影响权重计算，无需手动调 aux_loss 系数

---

## 7. 共享专家（Shared Experts）

### 7.1 概念

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
          └─���── + ────────────────┘
                ↓
           final_output
```

### 7.2 SharedFusedMoE 实现

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

### 7.3 Qwen2-MoE 的 sigmoid 共享门控

Qwen2-MoE 对共享专家额外加一个 sigmoid 门：

```python
# vllm/model_executor/models/qwen2_moe.py

# sigmoid gate 控制共享专家的贡献
shared_expert_gate = nn.Linear(hidden_size, 1, bias=False)
gate_value = torch.sigmoid(shared_expert_gate(hidden_states))  # (N, 1)

shared_output = shared_expert(hidden_states) * gate_value   # 按 sigmoid 缩放
```

---

## 8. FusedMoE 前向传播流程

### 8.1 完整调用链

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

### 8.2 fused_experts() 内核核心逻辑

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

步���5 - 加权聚合（scatter + weighted sum）:
  对每个 token t，将其 top-k 个专家输出加权求和：
  y[t] = Σ_{k=1}^{K} topk_weights[t,k] × expert_output[t, topk_ids[t,k]]
```

### 8.3 权重应用时机控制

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

## 9. CUDA/Triton 内核实现

### 9.1 TopK Softmax CUDA 内核

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

### 9.2 分组 TopK CUDA 内核

**文件**：`csrc/moe/grouped_topk_kernels.cu`

```
使用 Bitonic Sort 和 WarpSelect：

1. 将 scores reshape 为 (num_tokens, num_groups, experts_per_group)
2. 每个 warp 在一个 group 内做 top-2 selection
3. 对 group scores 做 Bitonic Sort 选出 topk_group 个组
4. 在选中的组内做最终 TopK
```

### 9.3 Token-Expert 对齐（moe_align_block_size）

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

## 10. 专家并行（Expert Parallelism）

### 10.1 概念

**Tensor Parallel (TP)**：每个 GPU 持有所有专家的部分权重（沿 intermediate 维度切分）

**Expert Parallel (EP)**：每个 GPU 持有部分专家的完整权���

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

### 10.2 专家映射（determine_expert_map）

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

### 10.3 All-to-All 通信

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

## 11. 专家并行负载均衡（EPLB）

### 11.1 问题

MoE 的核心挑战：专家负载不均衡。

```
理想：每个专家处理 N×K/E 个 token（均匀分布）
实际：热门专家可能处理 10× 平均量，冷门专家几乎不���选择

→ 导致：部分 GPU 成为瓶颈，整体吞吐降低
```

### 11.2 三层专家 ID 体系

```
逻辑专家 ID（Logical ID）
  ↓  由训练决定，模型文件中存储 0..E-1
物理专家 ID（Physical ID）
  ↓  包含冗余副本，0..E+R-1（R = 冗余专家数）
GPU 本地专家 ID（Local ID）
     每个 GPU 只持有自己负责的专家
```

### 11.3 冗余专家机制

**热门专家多副本**：
```
逻辑专家 3（被频繁访问）→ 物理专家 3, 17（两个副本，分布在不同 GPU）
逻辑专家 7（被较少访问）→ 物理专家 7（单副本）

eplb_map_to_physical_and_record() 在 routing 时：
  1. 查询 logical_replica_count[logical_id]（该专家有几个副本）
  2. replica_idx = position % replica_count（伪随机选副本，均摊负载）
  3. physical_id = logical_to_physical_map[logical_id, replica_idx]
```

### 11.4 动态重新平衡

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

## 12. 量化支持

### 12.1 FusedMoEQuantConfig

```python
# vllm/model_executor/layers/fused_moe/config.py

@dataclass
class FusedMoEQuantConfig:
    use_fp8_w8a8: bool       # FP8 权重 + FP8 ���活（最高性能，H100）
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

### 12.2 内核选择策略

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

## 13. MoE vs Dense 对比

### 13.1 计算特性对比

| 特性 | Dense FFN | MoE FFN |
|------|-----------|---------|
| 前向计算 | 所有 token × 全部参数 | 所有 token × K/E 参数 |
| 总参数量 | H×I×2 | E×H×I×2 |
| 活跃参数 | H×I×2（100%） | K/E×H×I×2（通常 25%） |
| 内存访问 | 顺序，对 cache 友好 | 稀疏分散，需对齐 |
| 通信量 | All-reduce (TP) | All-to-All (EP) + All-reduce (TP) |
| 负载均衡 | 天然均衡 | 需显式处理（aux_loss 或 EPLB） |

### 13.2 推理挑战

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

### 13.3 vLLM MoE 设计总结

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
