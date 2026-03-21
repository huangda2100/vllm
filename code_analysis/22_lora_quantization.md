# vLLM 中的 LoRA 与量化

## 目录
1. [LoRA 基础原理](#1-lora-基础原理)
2. [vLLM LoRA 系统架构](#2-vllm-lora-系统架构)
3. [LoRA GPU 推理实现](#3-lora-gpu-推理实现)
4. [Multi-LoRA 服务](#4-multi-lora-服务)
5. [量化基础原理](#5-量化基础原理)
6. [vLLM 量化框架](#6-vllm-量化框架)
7. [AWQ 量化](#7-awq-量化)
8. [GPTQ 量化](#8-gptq-量化)
9. [FP8 量化](#9-fp8-量化)
10. [BitsAndBytes 量化](#10-bitsandbytes-量化)
11. [GGUF 量化](#11-gguf-量化)
12. [量化集成模式](#12-量化集成模式)
13. [LoRA 与量化对比](#13-lora-与量化对比)

---

## 1. LoRA 基础原理

### 1.1 动机：高效微调

全参数微调（Full Fine-tuning）代价极高。对于 LLaMA-3-70B：
- 权重：~140GB (fp16)
- 梯度：~140GB
- Adam 优化器状态：~280GB
- 合计 >560GB，需要 8×80GB A100

**LoRA（Low-Rank Adaptation）** 的洞察：微调过程中权重更新矩阵 ΔW 具有**低秩结构**。

### 1.2 数学原理

原始线性层：
```
y = xW    (W ∈ R^{d×k})
```

LoRA 引入低秩分解：
```
y = xW + x(BA)·(α/r)

其中：
  A ∈ R^{r×d}   (lora_a，输入投影)
  B ∈ R^{k×r}   (lora_b，输出投影)
  r ≪ min(d, k)  (秩，通常 r=8 或 16)
  α              (缩放超参数)
```

参数量对比（d=k=4096，r=16）：
- 全参数：4096² = 16.7M 参数
- LoRA：r×(d+k) = 16×8192 = 131K 参数 → 减少 **128x**

**权重初始化**：
- A：随机高斯初始化
- B：全零初始化（使训练开始时 ΔW = BA = 0，不破坏预训练模型）

### 1.3 缩放策略

**Standard LoRA**：`scaling = alpha / r`

**rsLoRA（rank-stabilized）**：`scaling = alpha / sqrt(r)`
- 随 r 增大时梯度更稳定（vLLM 通过 `PEFTHelper` 解析 `use_rslora` 字段）

---

## 2. vLLM LoRA 系统架构

### 2.1 核心数据结构

```
vllm/lora/
├── request.py          # LoRARequest：用户请求中的 LoRA 标识
├── models.py           # LoRAModel, LoRAModelManager, LRUCacheLoRAModelManager
├── lora_weights.py     # LoRALayerWeights, PackedLoRALayerWeights
├── peft_helper.py      # PEFTHelper：解析 adapter_config.json
├── worker_manager.py   # WorkerLoRAManager：磁盘加载与 Worker 管理
├── punica_wrapper/
│   ├── punica_gpu.py   # PunicaWrapperGPU：多 LoRA GPU 内核封装
│   └── punica_base.py
└── layers/
    ├── base_linear.py
    ├── column_parallel_linear.py
    └── row_parallel_linear.py
```

### 2.2 LoRARequest

```python
# vllm/lora/request.py
@dataclass
class LoRARequest:
    lora_name: str           # 用户定义的名称
    lora_int_id: int         # 内部整数 ID（用于 GPU slot 索引）
    lora_path: str           # 磁盘路径（safetensors/bin/pt）
    lora_local_path: Optional[str] = None
    base_model_name: Optional[str] = None
```

### 2.3 LoRAModel 与管理器

```python
# vllm/lora/models.py
class LoRAModel:
    id: int                              # 对应 lora_int_id
    rank: int                            # LoRA 秩
    loras: Dict[str, LoRALayerWeights]   # 层名 → 权重
```

**LoRAModelManager**：
- `_create_lora_modules()`：遍历模型，将目标 `Linear` 层替换为 `ColumnParallelLinearWithLoRA` / `RowParallelLinearWithLoRA`
- `activate_adapter(lora_id)`：将 LoRA 权重写入 GPU slot，并跟踪哪个 slot 存储哪个 adapter
- `deactivate_adapter(lora_id)`：释放 GPU slot

**LRUCacheLoRAModelManager**：
- 继承自 LoRAModelManager
- 维护 LRU 缓存，自动淘汰最久未使用的 adapter
- 支持同时服务**数百个 LoRA**（内存中仅保留 `max_loras` 个）

---

## 3. LoRA GPU 推理实现

### 3.1 GPU Slot 系统

vLLM 预分配固定数量的 GPU slot（`max_loras`，默认 1-4 个）：

```python
# 每个支持 LoRA 的线性层维护：
lora_a_stacked: Tensor  # shape: [max_loras, 1, rank, input_dim]
lora_b_stacked: Tensor  # shape: [max_loras, 1, output_dim, rank]
lora_bias_stacked: Optional[Tensor]  # shape: [max_loras, 1, output_dim]
```

激活 adapter 时，将权重复制到对应 slot：
```python
# vllm/lora/layers/base_linear.py
def set_lora(self, index, lora_a, lora_b, embeddings_tensor):
    self.lora_a_stacked[index, 0, :lora_a.shape[0], :] = lora_a
    self.lora_b_stacked[index, 0, :, :lora_b.shape[0]] = lora_b
```

**关键**：LoRA 不修改基础权重（W 保持不变），在 forward pass 中动态计算 ΔW·x。

### 3.2 前向传播 apply()

```python
# vllm/lora/layers/base_linear.py
def apply(self, x, bias=None):
    # 1. 基础权重计算（原始 Linear 层）
    output = self.base_layer(x, bias)

    # 2. LoRA 增量计算（Punica SGMV 内核）
    self.punica_wrapper.add_lora_linear(
        output,
        x,
        self.lora_a_stacked,
        self.lora_b_stacked,
        self.lora_bias_stacked,
        1.0,              # scale（已融合到 lora_b 中）
        ...
    )
    return output
```

### 3.3 Punica SGMV 内核

**SGMV（Sparse Gather Matrix-Vector）**：批量处理多个不同 LoRA 的矩阵乘法。

```python
# vllm/lora/punica_wrapper/punica_gpu.py
class PunicaWrapperGPU:
    def add_lora_linear(self, y, x, lora_a, lora_b, ...):
        # Step 1: Shrink（输入投影）
        # buffer[i] = x[i] @ lora_a[slot_mapping[i]]
        add_shrink(buffer, x, lora_a, self.token_lora_indices, ...)

        # Step 2: Expand（输出投影）
        # y[i] += buffer[i] @ lora_b[slot_mapping[i]] * scaling
        add_expand(y, buffer, lora_b, self.token_lora_indices, ...)
```

两步 Triton 内核实现（以 r=16 为例）：
```
x[i] ∈ R^d  ──shrink──→  buf[i] ∈ R^r  ──expand──→  delta_y[i] ∈ R^k
              (d×r 矩阵)                   (r×k 矩阵)
```

**metadata 更新**（每 step 调用）：
```python
def update_metadata(self, mapping: LoRAMapping, ...):
    # mapping.index_mapping: [num_tokens] → slot_index
    # 区分 prefill tokens 和 decode tokens
    self.token_lora_indices = mapping.prompt_mapping
```

### 3.4 LoRA 权重优化

**scaling 融合**（`LoRALayerWeights.optimize()`）：
```python
# 将 scaling 乘到 lora_b 中，避免推理时每次乘法
lora_b = lora_b * scaling  # 预计算
```

**rsLoRA**（`PEFTHelper.scaling_factor`）：
```python
if use_rslora:
    scaling = lora_alpha / math.sqrt(r)
else:
    scaling = lora_alpha / r
```

### 3.5 张量并行（TP）下的 LoRA 分片

**Column Parallel Linear（Q/K/V/gate/up 投影）**：
```
W: [d, k]  分片为  [d, k/tp]（每个 GPU 持有部分列）

lora_a: [r, d]    → 无需分片（每个 GPU 相同）
lora_b: [k, r]    → 按行分片  [k/tp, r]（输出维度切分）

前向：x @ lora_a = buf（所有 GPU 相同），buf @ lora_b[shard] = delta_y[shard]
```

**Row Parallel Linear（o/down 投影）**：
```
W: [d, k]  分片为  [d/tp, k]（每个 GPU 持有部分行）

lora_a: [r, d]    → 按列分片  [r, d/tp]（输入维度切分）
lora_b: [k, r]    → 无需分片（每个 GPU 相同，最后 all-reduce 合并）

前向：x[shard] @ lora_a[shard] = buf（partial），all-reduce buf，buf @ lora_b = delta_y
```

**QKV Packed（MergedQKVParallelLinearWithLoRA）**：
- Q/K/V 有独立的 lora_a 和 lora_b（因为它们的 rank 可以不同）
- `PackedLoRALayerWeights` 将多个 lora 按 output 维度拼接，统一处理

---

## 4. Multi-LoRA 服务

### 4.1 整体流程

```
Request 1 (LoRA A) ──┐
Request 2 (LoRA B) ──┤ Scheduler ──→ GPU Batch
Request 3 (LoRA A) ──┤
Request 4 (no LoRA)──┘

              ↓
        LoRAMapping
  token_index → slot_index
  [0,1,2,3,4,   5,6,7,8,   9,10,11, 12,13,14]
  [slot_A,slot_A,...  slot_B,...   slot_A,...  0,0,0]
              ↓
       PunicaWrapperGPU
   per-token 选择对应 LoRA slot
```

### 4.2 LoRAMapping

```python
@dataclass
class LoRAMapping:
    # 每个 token 对应的 LoRA slot index（0 = base model）
    index_mapping: Tuple[int, ...]  # [num_tokens]
    prompt_mapping: Tuple[int, ...]  # 用于 prompt logprobs
```

### 4.3 加载策略

**WorkerLoRAManager._load_adapter()**：
1. 从磁盘加载 safetensors/bin/pt 文件
2. 通过 `PEFTHelper` 解析 adapter_config.json（rank, lora_alpha, target_modules）
3. 构建 `LoRAModel` 对象（包含各层的 A/B 矩阵）
4. 调用 `activate_adapter()` 写入 GPU slot

**LRU 淘汰**：当 GPU slot 满时，淘汰最久未使用的 adapter（从 GPU slot 清除，但 CPU 内存中的 LoRAModel 可保留）。

---

## 5. 量化基础原理

### 5.1 为什么需要量化

LLM 推理的瓶颈是**内存带宽**（Memory Bandwidth），而非计算（对于小 batch size）：

```
LLaMA-3-70B 参数量：70B × 2 bytes (fp16) = 140GB
A100 HBM 带宽：2TB/s

每 token decode：需要读取所有权重 → 140GB / 2TB/s = 70ms/token (理论)
```

**量化收益**：
| 精度  | 大小 | 带宽节省 | ��度损失 |
|-------|------|---------|---------|
| FP16  | 2B/param | 1x | 无 |
| INT8  | 1B/param | 2x | 极小 |
| FP8   | 1B/param | 2x | 极小 |
| INT4  | 0.5B/param | 4x | 小 |
| INT2  | 0.25B/param | 8x | 大 |

### 5.2 量化基本概念

**均匀量化（Uniform Quantization）**：
```
q = round(x / scale) + zero_point
x_dequant = (q - zero_point) × scale

scale = (x_max - x_min) / (2^bits - 1)
```

**对称量化（Symmetric）**：`zero_point = 0`，适合权重（分布对称）

**非对称量化（Asymmetric）**：`zero_point ≠ 0`，适合激活值（如 ReLU 后输出 ≥ 0）

**Group Quantization（分组量化）**：
- 每 N 个权重共享一组 scale/zero_point（N=128 常见）
- 精度高于 per-tensor，成本低于 per-channel

**Quantization Granularity**（从粗到细）：
```
per-tensor → per-channel (per-axis) → per-group → per-element
  最低精度                                           最高精度
  最低开销                                           最高开销
```

---

## 6. vLLM 量化框架

### 6.1 基础抽象

```python
# vllm/model_executor/layers/quantization/base_config.py

class QuantizationConfig(ABC):
    @classmethod
    @abstractmethod
    def get_name(cls) -> str:
        """量化方案名称，如 'awq', 'gptq', 'fp8'"""

    @classmethod
    @abstractmethod
    def from_config(cls, config: Dict) -> "QuantizationConfig":
        """从 config.json 的 quantization_config 字段构建"""

    @abstractmethod
    def get_quant_method(self, layer, prefix: str) -> Optional[QuantizeMethodBase]:
        """为指定层返回量化方法"""


class QuantizeMethodBase(ABC):
    @abstractmethod
    def create_weights(self, layer, input_size_per_partition, ...):
        """创建量化权重 tensor（qweight/scales/zeros）"""

    @abstractmethod
    def apply(self, layer, x, bias=None) -> Tensor:
        """量化感知前向传播"""

    def process_weights_after_loading(self, layer):
        """权重加载后的后处理（如 reorder, pack）"""
```

### 6.2 注册与加载

```python
# vllm/model_executor/layers/linear.py
class LinearBase(nn.Module):
    def __init__(self, ..., quant_config=None, prefix=""):
        if quant_config is not None:
            self.quant_method = quant_config.get_quant_method(self, prefix)
        else:
            self.quant_method = UnquantizedLinearMethod()

    def create_weights(self, ...):
        self.quant_method.create_weights(self, ...)

    def forward(self, x):
        return self.quant_method.apply(self, x, self.bias)
```

量化方案注册表（从 `config.json` 的 `quantization_config.quant_type` 字段加载）：
```python
_QUANTIZATION_CONFIG_REGISTRY = {
    "awq": AWQConfig,
    "gptq": GPTQConfig,
    "fp8": Fp8Config,
    "bitsandbytes": BitsAndBytesConfig,
    "gguf": GGUFConfig,
    ...
}
```

---

## 7. AWQ 量化

### 7.1 原理（Activation-aware Weight Quantization）

AWQ 的核心洞察：**并非所有权重都同等重要**。激活值分布中的**显著通道**（salient channels，对应激活值较大的输入维度）对精度影响更大。

**校准过程**（离线）：
1. 用少量校准数据分析激活值分布
2. 找到每个 channel 的激活 scale s
3. 将权重除以 s（降低量化误差），激活乘以 s（保持等价）
4. 再对调整后的权重做 INT4 量化

数学上：`W·x = (W·diag(s)) · (diag(s)^{-1}·x)` — 右边的 W' 量化精度更高。

### 7.2 vLLM 实现

**权重格式**：
```python
# vllm/model_executor/layers/quantization/awq.py
qweight:  Tensor  # [input_dim, output_dim // pack_factor]，INT32，pack_factor=8（8个INT4打包）
qzeros:   Tensor  # [input_dim // group_size, output_dim // pack_factor]，每组 zero_point
scales:   Tensor  # [input_dim // group_size, output_dim]，FP16
```

**前向传播**（`AWQLinearMethod.apply()`）：
```python
def apply(self, layer, x, bias=None):
    if x.shape[0] >= 256:  # 大 batch：解量化后用标准 matmul
        out = awq_dequantize(qweight, scales, qzeros)  # → FP16
        out = x @ out
    else:                  # 小 batch：专用 gemm 内核（避免解量化开销）
        out = awq_gemm(x, qweight, scales, qzeros, pack_factor)
    return out + bias
```

**关键设计**：
- INT4 打包：8个 INT4 → 1个 INT32，节省内存
- 分组量化（group_size=128）：平衡精度与内存
- 大/小 batch 分支：decode（小 batch）用 `awq_gemm`，prefill（大 batch）用矩阵解量化

---

## 8. GPTQ 量化

### 8.1 原理（Generative Pre-trained Transformer Quantization）

GPTQ 基于 OBQ（Optimal Brain Quantization）：将量化视为逐列优化问题，利用 Hessian 矩阵补偿量化误差。

**核心步骤**：
1. 按列顺序量化权重 W
2. 每量化一列后，用 Hessian 逆矩阵更新剩余权重：
   ```
   W_{:, j:} -= (W_{:,j} - Quant(W_{:,j})) / [H^{-1}]_{jj} × [H^{-1}]_{j, j:}
   ```
3. 误差传播使总量化误差最小

**激活顺序（act-order / desc_act）**：
- 按激活值重要性重排权重列顺序（激活值大的列先量化）
- 降低量化误差，但需要额外的 `g_idx` 张量记录原始顺序

### 8.2 vLLM 实现

```python
# vllm/model_executor/layers/quantization/gptq.py

# 权重格式（packed_dim=0，按行打包）
qweight:  Tensor  # [input_dim // pack_factor, output_dim]，INT32
scales:   Tensor  # [num_groups, output_dim]，FP16
qzeros:   Tensor  # [num_groups, output_dim // pack_factor]，INT32
g_idx:    Tensor  # [input_dim]，每个输入��度对应的 group 索引（仅 desc_act 时需要）
```

**Exllama 内核加速**：
```python
# process_weights_after_loading：
gptq_shuffle(qweight, g_idx, bits)  # 按 g_idx 重排权重（将随机访问变为顺序访问）

# apply：
gptq_gemm(x, qweight, scales, qzeros, g_idx)  # Exllama CUDA 内核
```

**支持位宽**：2/3/4/8-bit，通过 `bits` 参数控制打包比例。

---

## 9. FP8 量化

### 9.1 原理

FP8 使用 IEEE 浮点格式，有 E4M3（较大范围）和 E5M2（较大精度）两种子格式：
- **E4M3**：指数4位 + 尾数3位，范围 ±448，常用于**权重**
- **E5M2**：指数5位 + 尾数2位，范围 ±57344，常用于**梯度**

FP8 相对 INT8 的优势：无需 zero_point（浮点天然对称），支持更大动态范围。

### 9.2 vLLM 实现

**两种模式**：

**W8A8（权重FP8 + 激活FP8，计算时FP8）**：
```python
# 激活量化：
if dynamic:
    scale = amax(x) / 448.0    # 动态：每次推理计算
else:
    scale = loaded_scale        # 静态：校准时预计算

x_fp8 = x.to(torch.float8_e4m3fn) / scale

# GEMM：
out = cutlass_scaled_mm(x_fp8, w_fp8, out_dtype=fp16)
```

**W8A16（权重FP8，激活FP16，计算时FP16/BF16）**：
```python
# 使用 Marlin FP8 内核：
out = fp8_marlin_gemm(x_fp16, w_fp8_packed, scales, ...)
```

**Block Quantization（块量化）**：
- 每 `block_size`（如128）个权重共享一个 scale
- 更细粒度，精度更高，适合大模型

**H100 FP8 原生支持**（FA3 + FP8）：
- FlashAttention3 支持 FP8 attention，vLLM 生成 AOT（Ahead-Of-Time）调度元数据
- Hopper 架构 TMA 异步加载 + WGMMA 指令，FP8 throughput 可达 FP16 的 2x

---

## 10. BitsAndBytes 量化

### 10.1 INT4（NF4/FP4）

**NF4（Normal Float 4-bit）**：
- 基于正态分布的非均匀量化级别（16个量化点按正态分布等概率分配）
- 更适合权重分布（近似正态）
- 支持**双重量化（Double Quantization）**：对 scale 再量化，进一步节省内存

**FP4**：标准 4-bit 浮点格式。

```python
# vllm/model_executor/layers/quantization/bitsandbytes.py

# INT4 前向：
out = matmul_4bit(x, weight_int4, quant_state)
# quant_state 包含：scale, zero_point, NF4/FP4 查找表
```

### 10.2 INT8（LLM.int8()）

**动机**：权重分布接近正态，但激活值有**异常值（outliers）**（少数维度激活值极大）。

**处理策略（`llm_int8_threshold=6.0`）**：
```python
# 将 x 分为正常部分和 outlier 部分：
mask = |x| > threshold           # outlier 维度
x_normal = x[:, ~mask]           # 正常：INT8 量化计算
x_outlier = x[:, mask].float()   # 异常：FP16 精确计算

# 两部分分别 matmul 后相加
out = MatmulLt(x_normal, W_int8) + x_outlier @ W_outlier_fp16
```

**特点**：
- 零精度损失（相比纯 INT8）
- 但 outlier 处理引入额外开销
- `bnb_quant_state` 每个分片独立维护量化状态（支持 TP 分片）

---

## 11. GGUF 量化

### 11.1 K-quant 类型

GGUF（llama.cpp 格式）引入 K-quant（K 代表 "Kernel-optimized"）：

| 类型 | 位宽 | Block 大小 | 描述 |
|------|------|-----------|------|
| Q2_K | 2.56 bit | 256 | 极度压缩，精度损失大 |
| Q3_K | 3.44 bit | 256 | 平衡压缩率与精度 |
| Q4_K | 4.50 bit | 256 | **推荐**，接近 FP16 精度 |
| Q5_K | 5.50 bit | 256 | 高精度 |
| Q6_K | 6.56 bit | 256 | 接近无损 |

K-quant 特点：使用超块结构（super-block），每 super-block 内有多个 sub-block，sub-block 共享 scale，super-block 共享 super-scale。

### 11.2 vLLM 实现

**内核选择**：
```python
# vllm/model_executor/layers/quantization/gguf.py

if batch_size <= 4:   # 小 batch（decode 阶段）
    out = ggml_mul_mat_vec_a8(qweight, x)   # MMVQ：矩阵-向量乘法
elif支持 MMQ:          # 中等 batch
    out = ggml_mul_mat_a8(qweight, x)       # MMQ：优化矩阵乘法
else:                  # 大 batch（prefill 阶段）
    w_fp16 = dequantize(qweight)            # 解量化后标准 matmul
    out = x @ w_fp16
```

**自定义算子注册**：
```python
# 通过 direct_register_custom_op 注册 GGUF 内核为 PyTorch 自定义算子
# 支持 torch.compile 追踪
```

---

## 12. 量化集成模式

### 12.1 统一接口流程

```
模型加载：
  config.json
      ↓ quantization_config.quant_type
  _QUANTIZATION_CONFIG_REGISTRY[quant_type].from_config(config)
      ↓
  quant_config 对象（保存量化参数）

层初始化（LinearBase.__init__）：
  quant_config.get_quant_method(layer, prefix)
      ↓
  AWQLinearMethod / GPTQLinearMethod / Fp8LinearMethod / ...
      ↓
  create_weights()  → 创建 qweight/scales/zeros 等 tensor

权重加载后：
  process_weights_after_loading()  → pack/reorder/precompute

推理（LinearBase.forward）：
  quant_method.apply(layer, x, bias)
      ↓
  量化 GEMM 内核（awq_gemm / gptq_gemm / scaled_mm / ...）
```

### 12.2 各方案对比

| 方案 | 权重精度 | 激活精度 | 主要内核 | 适用场景 |
|------|---------|---------|---------|---------|
| AWQ | INT4 | FP16 | awq_gemm | 通用 GPU，高吞吐 |
| GPTQ | INT2-8 | FP16 | gptq_gemm (Exllama) | 极低内存，旧 GPU |
| FP8 W8A8 | FP8 | FP8 | cutlass scaled_mm | H100/A100，训练+推理 |
| FP8 W8A16 | FP8 | FP16 | Marlin FP8 | 中等 GPU���无激活量化 |
| BNB INT4 | INT4 | FP16 | matmul_4bit (bnb) | 研究/测试，CPU offload |
| BNB INT8 | INT8 | INT8+FP16 | MatmulLt | 精度优先场景 |
| GGUF Q4_K | ~4.5bit | FP16 | MMVQ/MMQ | llama.cpp 兼容 |

---

## 13. LoRA 与量化对比

### 13.1 目标与方法

| 维度 | LoRA | 量化 |
|------|------|------|
| 目标 | 高效微调/个性化 | 减小模型大小，加速推理 |
| 修改 | 添加低秩增量（不改原始权重） | 压缩原始权重（不可逆） |
| 精度 | 无损（数学等价） | 有损（近似） |
| 推理开销 | 小量额外计算（Punica SGMV） | 减少内存带宽，可能加速 |
| 多用户 | 支持多 LoRA 并发服务 | 单一量化模型 |
| 组合 | LoRA + 量化可叠加使用 | - |

### 13.2 组合使用（QLoRA 风格）

vLLM 支持量化基础模型 + LoRA 增量（类似 QLoRA 推理）：
```
推理：
  x  →  量化基础层（INT4 权重）  →  y_base
  x  →  LoRA 增量（FP16 A/B）   →  delta_y
  output = y_base + delta_y
```

### 13.3 工程权衡

**选择 LoRA**：
- 需要服务多个客户的不同微调版本
- 基础模型固定，只有 adapter 变化
- 对推理延迟要求高（减少 adapter 数量可接近 baseline）

**选择量化**：
- 单一模型，内存/带宽受限
- H100 → FP8（几乎无精度损失）
- 消费级 GPU → AWQ/GPTQ INT4（4x 内存节省）
- 研究/边缘设备 → BitsAndBytes / GGUF

**同时使用**：量化 + LoRA 可进一步降低内存，同时保持个性化能力。

---

## 总结

vLLM 的 LoRA 系统通过 **GPU slot 预分配 + Punica SGMV 内核**实现了高效的多 LoRA 并发服务，核心设计是不修改基础权重，在 forward 动态叠加增量。量化系统通过统一的 `QuantizationConfig/QuantizeMethodBase` 抽象支持 AWQ/GPTQ/FP8/BnB/GGUF 等多种方案，每种方案针对不同 GPU 架构和精度要求优化了专用 CUDA/Triton 内核，最终都通过 `LinearBase` 的统一接口无缝集成到模型推理流程中。
