# vLLM LLaMA 模型实现深度解析

> 文件：`vllm/model_executor/models/llama.py`（708 行）
> 定位：模型执行层，推理专用（无训练代码），支持张量并行、流水线并行、LoRA、量化

---

## 一、整体结构

```
LlamaForCausalLM          ← 对外暴露的顶层类，实现 SupportsLoRA/SupportsPP/SupportsEagle3
└── LlamaModel             ← @support_torch_compile，包含所有 Decoder 层
    └── LlamaDecoderLayer  ← 单个 Transformer Block
        ├── LlamaAttention ← Multi-head / GQA 注意力
        └── LlamaMLP       ← SwiGLU FFN
```

这5个类对应5个层次，每层有严格的职责分离。vLLM 和 HuggingFace 的最大区别在于：**所有层都是推理专用的，并在初始化时就根据并行策略划分权重**。

---

## 二、LlamaMLP：SwiGLU + 张量并行

### 代码
```python
self.gate_up_proj = MergedColumnParallelLinear(
    input_size=hidden_size,
    output_sizes=[intermediate_size] * 2,  # gate 和 up 合并
    ...
)
self.act_fn = SiluAndMul()

def forward(self, x):
    x, _ = self.gate_up_proj(x)   # 一次 GEMM 得到 [gate, up]
    x = self.act_fn(x)             # SiLU(gate) * up
    x, _ = self.down_proj(x)
    return x
```

### 原理：SwiGLU 激活函数

LLaMA 使用的不是普通 ReLU，而是 **SwiGLU**（Swish + Gated Linear Unit）：

```
FFN_SwiGLU(x) = (SiLU(W_gate · x) ⊙ W_up · x) · W_down
```

- **为什么用 SwiGLU？** 相比 ReLU，SwiGLU 在相同参数量下能达到更好的模型质量（PaLM 论文实证）。它引入门控机制，允许网络学习"哪些特征需要通过"。
- **SiLU = x · σ(x)**，平滑、有梯度，避免 ReLU 的死神经元问题。

### 关键设计：gate 和 up 合并成一个权重矩阵

HuggingFace 存两个独立矩阵 `gate_proj` 和 `up_proj`，vLLM 把它们合并成 `gate_up_proj`：

```
[gate_proj]   →   [gate_up_proj]  (output_size = 2 * intermediate_size)
[up_proj  ]
```

**原因：减少 GEMM 调用次数**。一次 `[hidden, 2×intermediate]` 的矩阵乘法比两次 `[hidden, intermediate]` 的矩阵乘法快，因为：
1. 减少 GPU 内核启动开销
2. 单个大 GEMM 比两个小 GEMM 更容易达到硬件峰值利用率（更好的 Tensor Core 利用率）

### 张量并行拆分

`MergedColumnParallelLinear` 在列维度切分：每个 GPU 持有 `[hidden, 2×intermediate/tp_size]` 的权重，各自独立计算，无需通信。`RowParallelLinear`（down_proj）在行维度切分，计算结束后做一次 `AllReduce`。

---

## 三、LlamaAttention：GQA + RoPE + 融合 QKV

### 代码
```python
self.qkv_proj = QKVParallelLinear(
    hidden_size=hidden_size,
    head_size=self.head_dim,
    total_num_heads=self.total_num_heads,
    total_num_kv_heads=self.total_num_kv_heads,
    ...
)

def forward(self, positions, hidden_states):
    qkv, _ = self.qkv_proj(hidden_states)
    q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
    q, k = self.rotary_emb(positions, q, k)
    attn_output = self.attn(q, k, v)
    output, _ = self.o_proj(attn_output)
    return output
```

### 原理一：GQA（Grouped Query Attention）

LLaMA-2/3 使用 GQA，Q heads 数量远大于 KV heads：

```
LLaMA-3-8B:  num_heads=32, num_kv_heads=8  → 每组4个Q共享1对KV
LLaMA-3-70B: num_heads=64, num_kv_heads=8  → 每组8个Q共享1对KV
```

**为什么 GQA？** KV Cache 是推理内存瓶颈。GQA 将 KV 头数从32减到8，KV Cache 内存降低4倍，同时准确率接近 MHA（Multi-Head Attention）。

### 张量并行下的 GQA 处理

```python
if self.total_num_kv_heads >= tp_size:
    # KV heads 多于 TP 数：正常切分
    self.num_kv_heads = self.total_num_kv_heads // tp_size
else:
    # KV heads 少于 TP 数：每个 GPU 复制一份 KV heads
    self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
```

当 TP=8，KV heads=8 时，每个 GPU 分到1个 KV head，4个 Q heads 共享它。当 TP=8，KV heads=4 时，每个 GPU 必须有完整的1个 KV head（不能再切分），因此部分 KV heads 被复制。

### 原理二：RoPE（旋转位置编码）

RoPE 不像绝对位置编码那样直接加到 embedding 上，而是在计算 QK 点积时通过旋转矩阵引入相对位置信息：

```
q_rotated = q ⊗ cos(θ) + rotate_half(q) ⊗ sin(θ)
k_rotated = k ⊗ cos(θ) + rotate_half(k) ⊗ sin(θ)
```

其中 `θ_i = base^(-2i/d)`，`base=10000`（LLaMA-3 用更大的 base 支持更长上下文）。

**为什么 RoPE 更好？**
1. 相对位置信息：`<q_rotated, k_rotated>` 只依赖位置差，不依赖绝对位置
2. 外推能力：训练时 4096 长度，推理时可外推到更长（用 RoPE scaling）
3. 不占参数：不需要像 learned positional embedding 那样存储位置参数

### 关键设计：`positions` 作为显式输入

注意 `forward(self, positions, hidden_states)` 中的 `positions` 参数。这是为了支持 **连续批处理**：不同请求的 token 拼接成一个 flat 的 1D 张量，每个 token 的位置不是简单的 `0,1,2,...`，而是各自在其原始序列中的绝对位置。

---

## 四、LlamaDecoderLayer：融合残差加法

### 代码
```python
def forward(self, positions, hidden_states, residual):
    # 第一次调用时 residual=None，直接算
    if residual is None:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
    else:
        # 后续调用：融合 Add+RMSNorm
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)

    # 融合 Add+RMSNorm
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual
```

### 原理：融合 Add + RMSNorm

标准 Transformer 的 Pre-LayerNorm 写法是：
```python
# 标准写法（HuggingFace）
hidden_states = layernorm(hidden_states + residual)
```

vLLM 的优化写法是把 `Add + RMSNorm` 融合成一个操作，并将 `residual` 作为单独的 tensor 传递：
```python
hidden_states, residual = layernorm(hidden_states, residual)
# 等价于：
# new_residual = hidden_states + residual
# hidden_states = rms_norm(new_residual)
# residual = new_residual
```

**为什么这样做？**
1. **减少内存读写**：单独的 Add 需要读 hidden_states 和 residual，写结果；单独的 RMSNorm 再读一次结果。融合后只读写一次，减少 HBM 带宽消耗。
2. **保持精度**：`residual` 始终以全精度（FP32 或 BF16 原始精度）保存，不会因为多次 normalization 而损失精度。
3. **利于编译器优化**：`@support_torch_compile` 装饰器后，编译器能识别这种 fused pattern 并生成更优的 Triton 代码。

---

## 五、LlamaModel：流水线并行 + EAGLE3

### 流水线并行

```python
self.start_layer, self.end_layer, self.layers = make_layers(
    config.num_hidden_layers,
    lambda prefix: layer_type(vllm_config=vllm_config, prefix=prefix),
    prefix=f"{prefix}.layers",
)
```

`make_layers` 根据当前进程所在的流水线并行组（PP rank）只创建该 rank 负责的层。例如：
- 32层模型，4个 PP rank → 每个 rank 只创建8层
- rank 0 持有 embed_tokens + layers[0:8]
- rank 1~2 只持有 layers[8:16], layers[16:24]
- rank 3 持有 layers[24:32] + lm_head + norm

```python
def forward(self, ...):
    if get_pp_group().is_first_rank:
        hidden_states = self.get_input_embeddings(input_ids)
        residual = None
    else:
        # 从上游 rank 接收中间张量
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    for layer in self.layers[start:end]:
        hidden_states, residual = layer(positions, hidden_states, residual)

    if not get_pp_group().is_last_rank:
        # 传给下游 rank
        return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})

    return self.norm(hidden_states, residual)
```

### EAGLE3 辅助隐藏状态

```python
self.aux_hidden_state_layers = tuple[int, ...]()

for idx, layer in enumerate(layers):
    if idx in self.aux_hidden_state_layers:
        aux_hidden_states.append(hidden_states + residual)  # 保存特定层的输出
    hidden_states, residual = layer(...)
```

EAGLE3 是一种推测解码方法：它用大模型的中间层输出（而不是只用最后输出）来训练 draft 模型，使 draft 模型能更准确地预测下一个 token。默认收集层 `(2, num_layers//2, num_layers-3)` 覆盖早、中、晚层的特征，提供丰富的多层次信息。

---

## 六、LlamaForCausalLM：权重加载与格式兼容

### stacked_params_mapping：HF → vLLM 的权重合并

```python
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),   # HF 的 q_proj → vLLM 的 qkv_proj[q部分]
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0), # HF 的 gate_proj → vLLM 的 gate_up_proj[前半]
    (".gate_up_proj", ".up_proj", 1),
]
```

HuggingFace 的 LLaMA 把 Q、K、V 存为三个独立矩阵，但 vLLM 运行时使用合并的 `qkv_proj`。加载时通过这个映射表，将 HF 格式的独立权重合并写入 vLLM 的联合权重，**不需要在磁盘上重新保存一份新格式的模型**。

```python
for param_name, weight_name, shard_id in stacked_params_mapping:
    if weight_name not in name:
        continue
    name = name.replace(weight_name, param_name)
    param.weight_loader(param, loaded_weight, shard_id)  # 写入对应的 shard
    break
```

### Mistral 格式重映射

旧版 Mistral/LLaMA1/2 使用不同的参数命名（`wq`/`wk` 而非 `q_proj`/`k_proj`）。`maybe_remap_mistral` 做名字转换，还额外做了一个 `permute` 操作：

```python
def permute(w, n_heads, attn_out):
    attn_in = self.config.head_dim * n_heads
    return (
        w.view(n_heads, attn_in // n_heads // 2, 2, attn_out)
         .transpose(1, 2)
         .reshape(attn_in, attn_out)
    )
```

这是因为旧版 Mistral 的 Q/K 权重是以 "interleaved" 格式存储的（RoPE 的两个分量交错），而 vLLM 用的是 "rotary" 格式（两个分量拼接），两者需要做维度变换才能等价。

---

## 七、关键设计模式总结

| 设计 | 原因 | 收益 |
|------|------|------|
| gate_up_proj 合并 | 减少 GEMM 调用 | 更高的 Tensor Core 利用率 |
| QKV 合并投影 | 减少 GEMM 调用 | 同上 |
| 残差流分离 (`residual` 单独传递) | 融合 Add+RMSNorm | 减少 HBM 读写次数 |
| `positions` 显式传入 | 支持连续批处理 | 混合长度序列可在同一 batch 中处理 |
| `make_layers` PP 切分 | 仅创建本 rank 的层 | 不浪费显存和初始化时间 |
| `stacked_params_mapping` | HF格式直接加载 | 无需重转储模型 |
| `@support_torch_compile` | 开启图编译 | 消除 Python 开销，启用内核融合 |

---

*文件：`vllm/model_executor/models/llama.py`*
*更新：2026-03*
