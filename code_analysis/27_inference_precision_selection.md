# 推理加速中的精度选择指南

## 1. 推理中的精度是什么？

### 1.1 核心概念

在推理加速中，**精度（Precision）** 指的是模型权重、激活值和中间计算结果的数值表示格式。精度直接影响：

- **计算速度**：低精度计算更快
- **内存占用**：低精度占用更少内存
- **模型准确性**：低精度可能损失精度
- **硬件利用率**：不同精度对硬件的利用率不同

### 1.2 推理 vs 训练的精度需求

| 维度 | 训练 | 推理 |
|------|------|------|
| 精度要求 | 高（需要精确梯度） | 中等（只需前向传播） |
| 常用精度 | FP32, BF16, FP16 | FP16, INT8, INT4 |
| 精度容忍度 | 低（影响收敛） | 高（1-3% 损失可接受） |
| 优化目标 | 收敛速度 | 推理速度 + 成本 |
| 内存需求 | 大（存储梯度） | 小（只存储权重） |

**关键差异：**
```
训练：
- 需要反向传播，梯度计算对精度敏感
- 小的数值误差会累积，影响收敛
- 通常使用 FP32 或混合精度（FP16+FP32）

推理：
- 只需前向传播，对精度容忍度更高
- 1-3% 的精度损失通常可以接受
- 可以使用更低精度（INT8, INT4）大幅加速
```

## 2. 推理中的常见精度类型

### 2.1 浮点精度

#### FP32（单精度浮点）

**格式：** [1位符号] [8位指数] [23位尾数]

**特点：**
```
优点：
✓ 精度高（~7位十进制有效数字）
✓ 数值范围大（±3.4×10³⁸）
✓ 无需额外校准
✓ 与训练精度一致

缺点：
✗ 内存占用大（4 bytes/参数）
✗ 计算速度慢
✗ 带宽需求高
✗ 无法利用 Tensor Core 加速

适用场景：
- 基线测试
- 精度敏感的应用
- 调试验证
```

**性能数据：**
```
LLaMA-7B FP32 推理（A100 GPU）：
- 内存占用：28 GB
- 推理速度：~50 tokens/s
- 延迟：~20ms/token
- GPU 利用率：~30%（无 Tensor Core）
```

#### FP16（半精度浮点）

**格式：** [1位符号] [5位指数] [10位尾数]

**特点：**
```
优点：
✓ 内存减半（2 bytes/参数）
✓ 计算速度快 2-3×
✓ 支持 Tensor Core（10× 加速）
✓ 精度损失小（<1%）

缺点：
✗ 数值范围小（±65504）
✗ 容易溢出
✗ 小数精度有限（~3位十进制）

适用场景：
- 通用推理（最常用）
- GPU 推理
- 在线服务
```

**性能数据：**
```
LLaMA-7B FP16 推理（A100 GPU）：
- 内存占用：14 GB
- 推理速度：~150 tokens/s
- 延迟：~6.7ms/token
- GPU 利用率：~80%（Tensor Core）
- 加速比：3× vs FP32
```

#### BF16（Brain Float 16）

**格式：** [1位符号] [8位指数] [7位尾数]

**特点：**
```
优点：
✓ 数值范围与 FP32 相同（±3.4×10³⁸）
✓ 不易溢出
✓ 支持 Tensor Core
✓ 与 FP32 转换简单（截断即可）

缺点：
✗ 精度低于 FP16（~2位十进制）
✗ 需要硬件支持（Ampere 及以上）

适用场景：
- 大模型推理（不易溢出）
- 训练+推理一致性
- 新一代 GPU（A100, H100）
```

**性能数据：**
```
LLaMA-7B BF16 推理（A100 GPU）：
- 内存占用：14 GB
- 推理速度：~140 tokens/s
- 延迟：~7.1ms/token
- GPU 利用率：~75%
- 精度损失：<1% vs FP32
```

### 2.2 整数精度（量化）

#### INT8（8位整数）

**格式：** 有符号整数 [-128, 127]

**量化原理：**
```
量化公式：
  Q = round(FP32 / scale) + zero_point

反量化公式：
  FP32 = (Q - zero_point) × scale

参数：
- scale: 缩放因子
- zero_point: 零点偏移
```

**特点：**
```
优点：
✓ 内存减少 4×（1 byte/参数）
✓ 计算速度快 3-4×
✓ 支持 INT8 Tensor Core
✓ 精度损失可控（1-3%）

缺点：
✗ 需要校准数据集
✗ 量化过程复杂
✗ 部分层可能精度损失大

适用场景：
- 生产部署（最佳性价比）
- 边缘设备
- 高吞吐量服务
```

**性能数据：**
```
LLaMA-7B INT8 推理（A100 GPU）：
- 内存占用：7 GB
- 推理速度：~200 tokens/s
- 延迟：~5ms/token
- GPU 利用率：~85%
- 精度损失：1-2% vs FP32
- 加速比：4× vs FP32
```

#### INT4（4位整数）

**格式：** 有符号整数 [-8, 7]

**特点：**
```
优点：
✓ 内存减少 8×（0.5 byte/参数）
✓ 极致压缩
✓ 适合超大模型

缺点：
✗ 精度损失较大（3-5%）
✗ 量化难度高
✗ 需要高级量化技术（GPTQ, AWQ）

适用场景：
- 超大模型（70B+）
- 内存受限环境
- 边缘设备
```

**性能数据：**
```
LLaMA-7B INT4 推理（A100 GPU）：
- 内存占用：3.5 GB
- 推理速度：~250 tokens/s
- 延迟：~4ms/token
- 精度损失：3-4% vs FP32
- 加速比：5× vs FP32

LLaMA-70B INT4 推理（A100 80GB）：
- 内存占用：35 GB（单卡可运行！）
- 推理速度：~30 tokens/s
- 精度损失：4-5% vs FP32
```

### 2.3 混合精度

**核心思想：** 不同层使用不同精度

```
策略：
- 敏感层（Attention, LayerNorm）：FP16/BF16
- 非敏感层（FFN）：INT8
- 极不敏感层：INT4

优势：
- 平衡精度和速度
- 精度损失 <2%
- 加速 3-4×
```

**示例配置：**
```python
# 混合精度配置
mixed_precision_config = {
    "attention": "fp16",      # 注意力层用 FP16
    "layernorm": "fp16",      # 归一化层用 FP16
    "ffn": "int8",            # 前馈网络用 INT8
    "embedding": "int8",      # 嵌入层用 INT8
}

# 性能：
# - 精度损失：<2%
# - 加速：3.5×
# - 内存：减少 3×
```

## 3. 如何选择推理精度？

### 3.1 决策流程图

```
开始
  ↓
是否有 GPU？
  ├─ 否 → CPU 推理
  │        ├─ 小模型 → FP32
  │        └─ 大模型 → INT8/INT4 (llama.cpp)
  │
  └─ 是 → GPU 推理
           ↓
      是否有 Tensor Core？
           ├─ 否（旧 GPU）→ FP32
           │
           └─ 是（新 GPU）
                    ↓
              内存是否充足？
                    ├─ 是 → FP16/BF16（推荐）
                    │
                    └─ 否 → 量化
                             ├─ 精度优先 → INT8
                             └─ 内存优先 → INT4
```

### 3.2 按场景选择

#### 场景 1：在线服务（延迟敏感）

**需求：**
- P99 延迟 < 100ms
- 高并发（1000+ QPS）
- 精度损失 < 2%

**推荐精度：**
```
首选：FP16
- 速度快（Tensor Core 加速）
- 精度高（<1% 损失）
- 延迟低

备选：INT8（如果内存不足）
- 内存减半
- 速度更快
- 精度损失 1-2%

配置示例：
```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="half",  # FP16
    max_num_seqs=128,
    gpu_memory_utilization=0.9
)
```

**性能预期：**
```
LLaMA-7B FP16（A100）：
- P50 延迟：6ms
- P99 延迟：15ms
- 吞吐量：150 tokens/s
- 并发：128 请求
```

#### 场景 2：离线批处理（吞吐量优先）

**需求：**
- 最大化吞吐量
- 延迟不敏感（秒级可接受）
- 成本优化

**推荐精度：**
```
首选：INT8
- 内存占用小（可增大批量）
- 速度快
- 成本低

配置示例：
```python
from vllm import LLM

llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq",  # INT4
    max_num_batched_tokens=8192,  # 大批量
    max_num_seqs=512
)
```

**性能预期：**
```
LLaMA-7B INT8（A100）：
- 批量大小：512
- 吞吐量：200 tokens/s
- 延迟：50-100ms（可接受）
- 成本：降低 50%
```

#### 场景 3：边缘设备（资源受限）

**需求：**
- 内存 < 8GB
- 功耗 < 10W
- 模型大小 < 2GB

**推荐精度：**
```
首选：INT4
- 极致压缩（8× 减少）
- 低功耗
- 可在移动设备运行

备选：INT8（如果精度要求高）

配置示例：
```python
# 使用 llama.cpp（CPU 推理）
from llama_cpp import Llama

llm = Llama(
    model_path="llama-2-7b.Q4_K_M.gguf",  # INT4 量化
    n_ctx=2048,
    n_threads=4
)
```

**性能预期：**
```
LLaMA-7B INT4（iPhone 15 Pro）：
- 模型大小：3.5 GB
- 推理速度：~10 tokens/s
- 功耗：~5W
- 精度损失：3-4%
```

#### 场景 4：超大模型（70B+）

**需求：**
- 模型 > 50B 参数
- 单卡内存有限
- 需要多卡或量化

**推荐精度：**
```
方案 1：FP16 + 多卡张量并行
- 精度高
- 速度快
- 成本高（需要多卡）

方案 2：INT4 量化 + 单卡
- 成本低
- 精度损失 3-5%
- 速度中等

配置示例：
```python
# 方案 1：FP16 多卡
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 4 卡
    dtype="half"
)

# 方案 2：INT4 单卡
llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",  # INT4
    dtype="half"
)
```

**性能对比：**
```
LLaMA-70B 推理：

FP16 4卡（4×A100 80GB）：
- 内存：每卡 35 GB
- 速度：~80 tokens/s
- 成本：4 卡
- 精度：无损失

INT4 单卡（A100 80GB）：
- 内存：35 GB
- 速度：~30 tokens/s
- 成本：1 卡
- 精度损失：4-5%
```

### 3.3 按硬件选择

#### NVIDIA GPU

**Ampere 架构（A100, A30）：**
```
推荐精度：
1. FP16（首选）
   - Tensor Core 加速 10×
   - 最佳性价比

2. BF16（备选）
   - 数值范围大
   - 不易溢出

3. INT8
   - INT8 Tensor Core 支持
   - 加速 4×

配置：
torch.backends.cuda.matmul.allow_tf32 = True
```

**Hopper 架构（H100）：**
```
推荐精度：
1. FP8（最新）
   - H100 专属
   - 加速 2× vs FP16
   - 精度损失 <1%

2. INT4
   - 极致压缩
   - 适合超大模型

配置：
使用 Transformer Engine
```

**旧架构（V100, T4）：**
```
推荐精度：
1. FP16（V100 有 Tensor Core）
2. FP32（T4 推理优化）
3. INT8（T4 INT8 优化）
```

#### CPU

**Intel/AMD CPU：**
```
推荐精度：
1. INT8（AVX-512 加速）
2. INT4（极致压缩）
3. FP32（基线）

工具：
- ONNX Runtime
- llama.cpp
- OpenVINO
```

#### 移动端

**iOS（Apple Neural Engine）：**
```
推荐精度：
1. INT8（ANE 优化）
2. FP16（Metal GPU）

工具：
- Core ML
- llama.cpp
```

**Android（Qualcomm NPU）：**
```
推荐精度：
1. INT8（NPU 优化）
2. INT4（极致压缩）

工具：
- TensorFlow Lite
- ONNX Runtime Mobile
```

## 4. 精度转换实践

### 4.1 FP32 → FP16

**最简单的转换：**

```python
import torch

# 加载 FP32 模型
model = torch.load("model_fp32.pth")

# 转换为 FP16
model_fp16 = model.half()

# 保存
torch.save(model_fp16, "model_fp16.pth")

# 推理
with torch.cuda.amp.autocast():
    output = model_fp16(input_data)
```

**vLLM 中使用 FP16：**

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    dtype="half"  # 自动转换为 FP16
)
```

### 4.2 FP32 → INT8

**使用 PyTorch 动态量化：**

```python
import torch
from torch.quantization import quantize_dynamic

# 加载模型
model = MyModel()

# 动态量化（推理时量化）
quantized_model = quantize_dynamic(
    model,
    {torch.nn.Linear},  # 量化 Linear 层
    dtype=torch.qint8
)

# 测试精度
accuracy_fp32 = evaluate(model)
accuracy_int8 = evaluate(quantized_model)
print(f"精度损失: {accuracy_fp32 - accuracy_int8:.2%}")
```

**使用 vLLM + SmoothQuant：**

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    quantization="smoothquant",  # INT8 量化
    dtype="half"
)
```

### 4.3 FP32 → INT4

**使用 GPTQ 量化：**

```python
# 1. 安装 auto-gptq
# pip install auto-gptq

from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

# 2. 配置量化参数
quantize_config = BaseQuantizeConfig(
    bits=4,  # 4位量化
    group_size=128,
    desc_act=False
)

# 3. 加载模型并量化
model = AutoGPTQForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantize_config=quantize_config
)

# 4. 量化（需要校准数据）
model.quantize(calibration_data)

# 5. 保存
model.save_quantized("llama-2-7b-gptq")
```

**使用 vLLM 加载 GPTQ 模型：**

```python
from vllm import LLM

llm = LLM(
    model="TheBloke/Llama-2-7B-GPTQ",
    quantization="gptq"
)
```

## 5. 精度对比总结

### 5.1 性能对比表

| 精度 | 内存 | 速度 | 精度损失 | 硬件要求 | 适用场景 |
|------|------|------|---------|---------|---------|
| FP32 | 1× | 1× | 0% | 通用 | 基线/调试 |
| FP16 | 0.5× | 3× | <1% | Tensor Core | 通用推理（推荐）|
| BF16 | 0.5× | 2.8× | <1% | Ampere+ | 大模型推理 |
| INT8 | 0.25× | 4× | 1-3% | INT8 支持 | 生产部署 |
| INT4 | 0.125× | 5× | 3-5% | 高级量化 | 超大模型/边缘 |

### 5.2 LLaMA-7B 实测数据（A100 GPU）

| 精度 | 内存(GB) | 速度(tokens/s) | 延迟(ms) | 精度损失 | 吞吐量(QPS) |
|------|---------|---------------|---------|---------|------------|
| FP32 | 28 | 50 | 20 | 0% | 50 |
| FP16 | 14 | 150 | 6.7 | <1% | 150 |
| BF16 | 14 | 140 | 7.1 | <1% | 140 |
| INT8 | 7 | 200 | 5.0 | 1.5% | 200 |
| INT4 | 3.5 | 250 | 4.0 | 3.5% | 250 |

### 5.3 推荐配置

**通用推荐（80% 场景）：**
```
FP16 + vLLM + A100
- 最佳平衡点
- 精度高、速度快
- 部署简单
```

**成本优化（生产环境）：**
```
INT8 + vLLM + A100
- 内存减半
- 速度更快
- 精度损失可接受（1-2%）
```

**极致压缩（超大模型）：**
```
INT4 + AWQ/GPTQ + A100
- 内存减少 8×
- 可单卡运行 70B 模型
- 精度损失 3-5%
```

## 6. 常见问题

**Q1: FP16 和 BF16 如何选择？**
```
A:
- 新 GPU（A100+）→ BF16（数值范围大，不易溢出）
- 旧 GPU（V100）→ FP16（更好的硬件支持）
- 大模型 → BF16（避免溢出）
- 小模型 → FP16（精度略高）
```

**Q2: 量化会损失多少精度？**
```
A: 实测数据（LLaMA-7B）：
- FP32 → FP16: <0.5%
- FP32 → INT8: 1-2%
- FP32 → INT4: 3-5%

建议：先测试，如果损失 >5%，考虑混合精度
```

**Q3: 如何验证精度损失？**
```python
# 对比不同精度的输出
def compare_precision(model_fp32, model_quantized, test_data):
    outputs_fp32 = model_fp32(test_data)
    outputs_quant = model_quantized(test_data)

    # 计算差异
    diff = torch.abs(outputs_fp32 - outputs_quant).mean()
    print(f"平均差异: {diff:.6f}")

    # 计算准确率
    acc_fp32 = evaluate(model_fp32, test_data)
    acc_quant = evaluate(model_quantized, test_data)
    print(f"精度损失: {acc_fp32 - acc_quant:.2%}")
```

**Q4: 混合精度如何配置？**
```python
# 示例：敏感层用 FP16，其他用 INT8
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    load_in_8bit=True,  # 默认 INT8
    llm_int8_skip_modules=["lm_head", "attention"]  # 这些层保持 FP16
)
```

推理精度选择是性能优化的关键，需要根据具体场景权衡精度、速度和成本！
