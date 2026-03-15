# vLLM 模型加载机制深度解析

> 涉及文件：`vllm/model_executor/model_loader/`（整目录）、`vllm/model_executor/layers/linear.py`、`vllm/model_executor/model_loader/utils.py`
> 定位：从磁盘 checkpoint 到可推理的 GPU 模型的完整转化过程

---

## 一、为什么模型加载是一个难题？

### 规模问题

LLaMA-3-70B 的参数量：
```
70B parameters × 2 bytes (BF16) = 140 GB
```

单卡 A100 仅有 80GB 显存，必须用 TP=2（张量并行）才能装载。加载过程中还需要一份 CPU 内存拷贝，峰值内存需求可达 300GB+。

### 多维度复杂性

```
┌──────────────────────────────────────────────────────────────┐
│  加载来源复杂：HuggingFace / ModelScope / 本地磁盘             │
│  格式多样：safetensors / pytorch.bin / GGUF / BitsAndBytes    │
│  参数名冲突：HF 分开存 q/k/v，vLLM 需要合并成 qkv_proj         │
│  并行切分：TP=8 时每个 rank 只需权重的 1/8，如何精确切片？       │
│  量化适配：AWQ/GPTQ/FP8 各自需要不同的 scales 和后处理         │
│  内存管理：140GB 权重如何流式加载、不爆 CPU 内存？               │
└──────────────────────────────────────────────────────────────┘
```

---

## 二、加载流程全景：四阶段模型

```
阶段 1：架构识别
  hf_config.architectures → ModelRegistry → 找到对应的 Python 类
        ↓
阶段 2：模型骨架初始化
  initialize_model() → 根据 TP/PP 配置创建层，分配 GPU 显存（空壳）
        ↓
阶段 3：权重注入
  model.load_weights(迭代器) → 逐个权重切片后 copy 到 GPU
        ↓
阶段 4：后处理
  process_weights_after_loading() → 量化后处理、权重重排、注意力初始化
```

### 代码入口

```python
# vllm/model_executor/model_loader/base_loader.py
def load_model(self, vllm_config, model_config) -> nn.Module:
    with set_default_torch_dtype(model_config.dtype):  # 全局设置 BF16/FP16
        with target_device:                             # 直接在 GPU 上分配
            model = initialize_model(vllm_config=vllm_config)

        self.load_weights(model, model_config)          # 权重注入
        process_weights_after_loading(model, model_config, target_device)  # 后处理
    return model.eval()
```

---

## 三、阶段 1 详解：架构识别

### 从 HF config 到 Python 类

```python
# vllm/model_executor/model_loader/utils.py
def initialize_model(vllm_config, ...):
    model_class, arch = get_model_architecture(model_config)
    # 例如：arch = "LlamaForCausalLM" → model_class = vllm 的 LlamaForCausalLM
```

`ModelRegistry` 维护一张表：`architectures 字符串 → vLLM 模型类`。这张表分两种来源：
- **内置模型**：在 `vllm/model_executor/models/__init__.py` 注册的 185 个模型
- **插件模型**：用户通过 `--model` 参数指定的自定义类

### 架构转换

加载时还会根据 `convert_type` 参数把同一模型转成不同用途：

```python
if convert_type == "embed":
    model_cls = as_embedding_model(model_cls)    # 用于文本 embedding
elif convert_type == "classify":
    model_cls = as_seq_cls_model(model_cls)      # 用于分类
elif convert_type == "reward":
    model_cls = as_reward_model(model_cls)       # 用于 RLHF reward
```

**同一套权重，套不同的"头"（Head），就成了不同任务的模型。**

---

## 四、阶段 2 详解：骨架初始化与 TP 切分

### 直接在 GPU 上分配空参数

```python
with set_default_torch_dtype(model_config.dtype):  # BF16/FP16
    with target_device:                             # with torch.device("cuda:0")
        model = model_class(vllm_config=vllm_config)
        # 模型 __init__ 中 nn.Parameter 自动分配到 target_device
```

`with target_device:` 上下文管理器让 PyTorch 的默认 device 变为 GPU，所有 `torch.empty()`、`nn.Parameter()` 都直接分配到 GPU 上，**不经过 CPU**。这避免了"先分配 CPU 再搬到 GPU"的两次内存分配。

### TP 切分在初始化时就发生

```python
# vllm/model_executor/layers/linear.py - ColumnParallelLinear
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, ...):
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()

        # 关键：直接只分配本 rank 需要的那部分权重
        self.output_size_per_partition = divide(output_size, self.tp_size)

        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,  # 不是 output_size！
                self.input_size_per_partition,
            )
        )
```

**在模型 __init__ 阶段，每个 GPU 的参数 shape 就已经是 1/tp_size 了**，后续加载时对应 slicing 到这个 shape 即可，无需先加载完整权重再切分。

---

## 五、阶段 3 详解：权重注入的核心机制

### 5.1 权重迭代器：流式不全量加载

```python
# default_loader.py
def get_all_weights(self, model_config, model):
    yield from self._get_weights_iterator(primary_source)
    for source in secondary_weights:   # 多模态模型的 encoder 权重等
        yield from self._get_weights_iterator(source)
```

迭代器是 Python Generator，每次只 yield 一个权重 tensor，处理完即释放，不需要把 140GB 全部加载到 CPU。

**不同格式对应不同迭代器**：

| 格式 | 迭代器 | 特点 |
|------|--------|------|
| `.safetensors` | `safetensors_weights_iterator` | 基础，lazy 读取 |
| `.safetensors` | `multi_thread_safetensors_weights_iterator` | 多线程并发加载 |
| `.safetensors` | `fastsafetensors_weights_iterator` | 直接加载到 GPU |
| `.bin / .pt` | `pt_weights_iterator` | PyTorch pickle |
| `.gguf` | `gguf_quant_weights_iterator` | 含量化类型元数据 |
| 已转 numpy | `np_cache_weights_iterator` | 内存映射加速 |

### 5.2 safetensors 懒加载原理

```python
# weight_utils.py
def safetensors_weights_iterator(hf_weights_files, ...):
    for st_file in hf_weights_files:
        with safe_open(st_file, framework="pt") as f:
            for name in f.keys():
                param = f.get_tensor(name)  # 仅此时才真正读取磁盘
                yield name, param
                # yield 之后 param 的引用消失，内存立即可被 GC
```

`safetensors` 格式（Meta 设计）在文件头部存储了每个 tensor 的偏移量，`get_tensor()` 可以精确跳转到对应位置只读该 tensor，不需要读整个文件。这是比 `.bin`（pickle 格式，必须顺序读）快得多的原因。

### 5.3 fastsafetensors：直接 GPU 加载

```python
# weight_utils.py
def fastsafetensors_weights_iterator(hf_weights_files, ...):
    device = torch.device(f"cuda:{local_rank}")
    loader = _init_loader(pg, device, f_list)
    fb = loader.copy_files_to_device()  # PCIe DMA 直传到 GPU
    for k in fb.key_to_rank_lidx.keys():
        yield k, fb.get_tensor(k)       # 已在 GPU 上的 tensor
```

传统路径：磁盘 → CPU 内存 → `tensor.to(device)` → GPU 内存（两次拷贝）
fastsafetensors：磁盘 → GPU 内存（一次 DMA，通过 GDRCopy 或 NVMe-GPU Direct）

### 5.4 weight_loader：精确切片的核心

每个模型参数都有对应的 `weight_loader` 函数，负责把从磁盘读出的**完整权重**切成本 rank 需要的**分片**：

```python
# linear.py - ColumnParallelLinear.weight_loader
def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
    output_dim = getattr(param, "output_dim", None)  # 沿哪个维度切
    is_sharded_weight = getattr(param, "is_sharded_weight", False)

    if output_dim is not None and not is_sharded_weight:
        shard_size = param.data.shape[output_dim]         # 本 rank 的大小
        start_idx = self.tp_rank * shard_size             # 在完整权重里的起始偏移
        loaded_weight = loaded_weight.narrow(             # zero-copy 切片！
            output_dim, start_idx, shard_size
        )

    assert param.data.shape == loaded_weight.shape
    param.data.copy_(loaded_weight)  # 写入 GPU 参数
```

**`narrow()` 是 zero-copy 操作**：它只是创建原 tensor 的一个视图（修改 storage offset 和 size），不分配新内存。真正的内存拷贝只有一次：`param.data.copy_()`，直接 H2D 传输。

### 5.5 参数名称映射：HF 格式 → vLLM 格式

HuggingFace 的 LLaMA checkpoint 存储的是独立的 `q_proj`/`k_proj`/`v_proj`，但 vLLM 运行时使用合并的 `qkv_proj`：

```python
# llama.py - LlamaModel.load_weights
stacked_params_mapping = [
    (".qkv_proj", ".q_proj", "q"),   # shard_id="q"
    (".qkv_proj", ".k_proj", "k"),
    (".qkv_proj", ".v_proj", "v"),
    (".gate_up_proj", ".gate_proj", 0),  # shard_id=0
    (".gate_up_proj", ".up_proj",   1),
]

for name, loaded_weight in weights:
    for param_name, weight_name, shard_id in stacked_params_mapping:
        if weight_name not in name:
            continue
        name = name.replace(weight_name, param_name)  # 重命名
        param.weight_loader(param, loaded_weight, shard_id)  # 写入对应 slot
        break
```

`QKVParallelLinear.weight_loader(param, loaded_weight, shard_id)` 会根据 `shard_id` 把 Q/K/V 各自写入合并权重的对应位置，最终在 GPU 上得到一个 `[q_size + k_size + v_size, hidden_size]` 的大矩阵，一次 GEMM 就能同时算出 Q、K、V。

**为什么要合并？** 三个独立矩阵乘 vs 一个大矩阵乘，在 GPU 上后者对 Tensor Core 的利用率更高，计算更高效。

---

## 六、阶段 4 详解：后处理

```python
# utils.py
def process_weights_after_loading(model, model_config, target_device):
    # 1. 遍历所有量化层，执行量化后处理
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # 2. 处理注意力层的权重后处理（如 MLA 压缩）
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)):
            module.process_weights_after_loading(model_config.dtype)
```

### 量化后处理做了什么？

不同量化方法需要不同的后处理：

| 量化方法 | 后处理内容 |
|---------|-----------|
| **GPTQ / AWQ** | 权重重排（Repack）：将量化权重从 HF 存储格式重排为 GPU 内核期望的访问模式，以对齐 Tensor Core 读取 |
| **Marlin** | 将 4bit 权重按 Marlin 内核要求的 tile 格式重新排列 |
| **FP8** | 计算 per-tensor 或 per-channel 的量化 scales |
| **BitsAndBytes** | 将 8bit 权重转换为 BNB 内部格式 |
| **Online Quantization** | 第一次加载时记录权重的统计信息（min/max/scale） |

### device_loading_context：CPU Offload 兼容

```python
@contextmanager
def device_loading_context(module, target_device):
    # CPU offload 场景：权重平时在 CPU，量化后处理需要在 GPU 上执行
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)  # 临时搬到 GPU
    try:
        yield module  # 在 GPU 上执行 process_weights_after_loading
    finally:
        for name, p in module.named_parameters():
            if name in original_device_states:
                p.data = p.data.to("cpu")  # 搬回 CPU
```

---

## 七、核心难点深度分析

### 难点 1：张量并行下的权重正确切分

**问题**：TP=8 时，8 个进程同时加载同一份 checkpoint，每个进程应取哪一片？

```
完整权重 W: shape [8192, 4096]
TP=8 时 ColumnParallel 切分后每个 rank: [1024, 4096]

rank 0 取 W[0:1024, :]
rank 1 取 W[1024:2048, :]
...
rank 7 取 W[7168:8192, :]
```

**实现方式**：每个 rank 都完整读取一遍磁盘上的 `W`（140GB 模型，8 个 rank 各读一遍，I/O 压力极大），然后用 `narrow()` 只保留自己的部分。

**优化方案**：`ShardedStateLoader` — 提前把权重按 TP rank 切好存成 8 个文件，每个 rank 只读自己的分片文件，I/O 减少 8 倍。

```python
# sharded_state_loader.py
DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

def load_weights(self, model, model_config):
    rank = get_tensor_model_parallel_rank()
    # 只加载 rank-specific 的文件
    filepaths = glob.glob(pattern.format(rank=rank, part="*"))
    for key, tensor in self.iterate_over_files(filepaths):
        state_dict[key].data.copy_(tensor)
```

### 难点 2：GQA 下 KV heads 的 TP 切分异常

标准 MHA（Multi-Head Attention）中，Q/K/V heads 数量相同，TP 切分很自然。但 GQA（Grouped Query Attention）中：

```
LLaMA-3-70B: num_q_heads=64, num_kv_heads=8, TP=8

Q heads 切分：每个 rank 有 64/8 = 8 个 Q heads  ✓
K/V heads 切分：每个 rank 有 8/8 = 1 个 KV head  ✓

但如果 TP=16：
K/V heads 切分：8/16 = 0.5 → 不能整除！
→ 每个 rank 必须复制同一个 KV head（每2个 rank 共享1个 KV head）
```

```python
# llama.py - LlamaAttention.__init__
if self.total_num_kv_heads >= tp_size:
    # KV heads >= TP size：正常切分
    assert self.total_num_kv_heads % tp_size == 0
    self.num_kv_heads = self.total_num_kv_heads // tp_size
else:
    # KV heads < TP size：必须复制
    assert tp_size % self.total_num_kv_heads == 0
    self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
    # 此时 num_kv_heads = 1，多个 rank 持有完全相同的 KV 权重
```

加载时，`QKVParallelLinear.weight_loader` 必须感知这种"复制"逻辑，给需要复制的 rank 加载完整的 KV 权重而非切片。

### 难点 3：量化 scales 的名称混乱

FP8 量化的 scale 参数名在不同训练框架/checkpoint 格式中命名极不一致：

```
# 可能出现的名字：
"model.layers.0.self_attn.k_proj.weight_scale"
"model.layers.0.self_attn.kv_scales"
"model.layers.0.self_attn.attn.kv_scale"
"model.layers.0.self_attn.k_fake_quantizer.qscale_act"  # Mistral 格式
```

vLLM 需要把这些全部映射到统一的内部命名：

```python
# weight_utils.py
def maybe_remap_kv_scale_name(name: str, params_dict: dict) -> str | None:
    """FP8 k/v_scale 参数名重映射"""
    if name.endswith(".k_scale"):
        remapped = name.replace(".k_scale", ".attn.kv_scales")
        if remapped in params_dict:
            return remapped
    # ... 多种格式的 fallback
```

### 难点 4：GGUF 格式的参数名完全不同

GGUF（llama.cpp 格式）使用完全不同的命名体系：

```
GGUF 格式:          HF/vLLM 格式:
blk.0.attn_q.weight → model.layers.0.self_attn.q_proj.weight
blk.0.ffn_gate.weight → model.layers.0.mlp.gate_proj.weight
token_embd.weight → model.embed_tokens.weight
output_norm.weight → model.norm.weight
```

`GGUFModelLoader` 需要动态构建一张从 GGUF 命名到 HF 命名的映射表：

```python
# gguf_loader.py
def _get_gguf_weights_map(self, model_config):
    arch = gguf.MODEL_ARCH_NAMES[model_type]
    name_map = gguf.get_tensor_name_map(arch, num_layers)
    # name_map 是 gguf 库提供的标准映射

    gguf_to_hf_name_map = {}
    for hf_name in dummy_model.state_dict():
        name, suffix = hf_name.rsplit(".", 1)
        gguf_name = name_map.get_name(name)
        if gguf_name:
            gguf_to_hf_name_map[f"{gguf_name}.{suffix}"] = hf_name
```

### 难点 5：流水线并行下的权重"缺失"

PP（流水线并行）场景下，rank 0 只有前 N 层，rank 1 有中间层，rank 3 有最后层。当 rank 0 加载权重时，遇到 `model.layers.24.xxx`（属于 rank 1 的层），需要直接跳过而不报错：

```python
# models/utils.py
def is_pp_missing_parameter(name: str, model: nn.Module) -> bool:
    """检查参数是否属于其他 PP rank（不应该在本 rank 加载）"""
    ...

# llama.py - LlamaModel.load_weights
for name, loaded_weight in weights:
    if is_pp_missing_parameter(name, self):
        continue  # 跳过，不报错
    ...
```

`PPMissingLayer` 是一个占位符类，当 PP rank 不是第一个时，`embed_tokens` 被替换成 `PPMissingLayer()`，任何对它的权重加载调用都会被静默忽略。

### 难点 6：Tied Weights（权重绑定）

LLaMA 的 `embed_tokens`（输入 embedding）和 `lm_head`（输出投影）在某些模型中共享同一份权重（tied weights）：

```python
# llama.py - LlamaForCausalLM.__init__
if config.tie_word_embeddings:
    self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
    # lm_head 的权重指向 embed_tokens 的权重，不占额外显存
```

加载时需要跳过 `lm_head` 的权重（因为 `embed_tokens` 已经加载了）：

```python
# llama.py - LlamaForCausalLM.load_weights
loader = AutoWeightsLoader(
    self,
    skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
)
```

如果不跳过，`lm_head` 的权重也被加载，会打破绑定关系（两份独立权重），不仅浪费显存，还可能导致模型行为不一致。

### 难点 7：在线量化的两阶段加载

在线量化（Online Quantization）是指：模型 checkpoint 存的是 BF16 权重，vLLM 启动时实时量化为 FP8 或 INT8 再推理。这需要两次加载：

```
第一次加载（首次运行）：
  - 正常读取 BF16 权重
  - 记录每个 tensor 的 shape、dtype、min/max 等统计信息
  - 设置 model.weight_metadata_and_attr_saved = True

第二次加载（后续调用）：
  - 读取 BF16 权重
  - 立即量化为 FP8
  - 基于第一次记录的 scales 进行量化
```

```python
# default_loader.py - load_weights
if not getattr(model, "weight_metadata_and_attr_saved", False):
    # 第一次：正常加载
    loaded_weights = model.load_weights(self.get_all_weights(...))
else:
    # 后续：加载 + 量化
    loaded_weights = load_weights_and_online_quantize(self, model, ...)
```

---

## 八、模型特性与加载的关系

| 模型特性 | 加载时的具体影响 |
|---------|--------------|
| **GQA（Grouped Query Attention）** | KV heads < TP size 时需要复制而非切分，`QKVParallelLinear.weight_loader` 特殊处理 |
| **Tied Embeddings** | `lm_head` 跳过加载，与 `embed_tokens` 共享内存指针 |
| **Sliding Window Attention** | 部分层用不同 attention 类型，`layer_types` 配置决定每层的 sliding_window 值 |
| **MoE（Mixture of Experts）** | 专家权重按 EP rank 切分，加载逻辑类似 TP 但在专家维度 |
| **MLA（DeepSeek）** | 压缩 KV 矩阵，加载后需要解压/重组，`process_weights_after_loading` 执行分解 |
| **RoPE Scaling** | 无需加载额外权重，运行时动态计算；但 `inv_freq` 需要跳过（不必要加载） |
| **Vision Encoder** | 作为 `secondary_weights` 从单独文件加载，独立于 LLM backbone |
| **LoRA** | 在基础模型加载完成后，额外加载 adapter weights 并 merge 或保持分离 |
| **BitsAndBytes 量化** | 权重在保存时已按 TP rank 分片（`is_sharded_weight=True`），加载时不再切分 |
| **GGUF 量化** | 带有量化类型元数据（Q4_K、Q8_0等），先 yield 类型再 yield 权重 |

---

## 九、内存使用分析

### 正常加载的内存峰值

```
GPU 峰值（单 rank，TP=8，70B BF16 模型）：
  模型参数：140GB / 8 = 17.5 GB
  加载中间状态：约 2~3 GB（当前正在处理的 tensor）
  CUDA 上下文：约 1 GB
  合计：约 21 GB  ← 可以装在 A100 80GB 上

CPU 峰值（TP=8 时各 rank 都需要从同一 checkpoint 读）：
  单个权重 tensor 最大可达 2~4 GB（大的线性层）
  使用 safetensors 懒加载时：CPU 内存约等于最大单个 tensor 的大小
  若用 .bin（pickle）格式：必须全量加载到 CPU → 可能需要 140GB CPU 内存
```

### 各方案的内存优化效果

```
方案                     CPU 内存    加载速度    GPU 内存
────────────────────────────────────────────────────────
safetensors（默认）       ~几GB       中等        最小
multi_thread_safetensors  ~几GB×N线程 最快        最小
fastsafetensors           ~几MB      极快        最小
ShardedStateLoader        最小        快          最小
npcache（numpy mmap）     内存映射   较快        最小
pytorch .bin              全量 ×N    慢           最小
```

---

## 十、完整数据流追踪

以加载一个 TP=2 的 LLaMA-3-8B BF16 模型、`q_proj` 权重为例：

```
checkpoint 中：
  "model.layers.0.self_attn.q_proj.weight"
  shape: [4096, 4096], dtype: bfloat16

safetensors_weights_iterator 读取：
  name = "model.layers.0.self_attn.q_proj.weight"
  loaded_weight = tensor([4096, 4096], bfloat16, device="cpu")

LlamaModel.load_weights 名称映射：
  ".q_proj" 命中 stacked_params_mapping
  name → "model.layers.0.self_attn.qkv_proj"
  shard_id = "q"

QKVParallelLinear.weight_loader(param, loaded_weight, "q"):
  # TP=2, rank=0
  q_shard_size = param.q_size = 4096 / 2 = 2048

  q部分的切片：
  loaded_weight.narrow(0, rank*q_shard_size, q_shard_size)
  → tensor([2048, 4096], bfloat16, device="cpu")  ← zero-copy narrow

  param.data[q_offset : q_offset + q_shard_size].copy_(sliced)
  → H2D 拷贝：2048*4096*2 bytes = 16MB 到 GPU

最终 GPU 上 qkv_proj.weight:
  shape: [2048 + 1024 + 1024, 4096] = [4096, 4096]  (rank=0的完整qkv)
  包含: Q[0:2048], K[0:1024], V[0:1024]  of the full weight
```

---

## 十一、关键设计模式总结

| 设计 | 解决的问题 | 原理 |
|------|-----------|------|
| Generator 迭代器 | 避免全量加载 OOM | 惰性求值，逐 tensor 处理 |
| `safetensors` 格式 | 任意 tensor 随机访问 | 文件头存偏移表，O(1) 定位 |
| `with target_device:` 上下文 | 直接在 GPU 分配参数 | PyTorch device context |
| `narrow()` zero-copy 切片 | TP 切分不增加内存 | view 操作，共享 storage |
| `weight_loader` 回调 | 各层自主决定如何切分 | Strategy 设计模式 |
| `stacked_params_mapping` | HF格式直接加载合并权重 | 名称重映射 + shard_id 路由 |
| `ShardedStateLoader` | 消除 TP 冗余 I/O | 提前切好分片文件 |
| `process_weights_after_loading` | 量化后处理解耦 | 加载与量化分离 |
| `PPMissingLayer` | PP rank 跳过不属于自己的层 | Null Object 设计模式 |
| `device_loading_context` | CPU offload 兼容量化后处理 | 临时搬到 GPU，处理完归还 |

---

*文件：`vllm/model_executor/model_loader/`*
*更新：2026-03*
