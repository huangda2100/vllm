# 模型权重的具体内容是什么样子的？

> 本文通过在线资源和代码示例，展示 LLM 模型权重的实际数值面貌
> 核心问题：权重到底是什么？数值长什么样？训练前后有何变化？

---

## 一、权重的本质：就是一堆浮点数

模型权重本质上是一个**多维浮点数数组**（张量）。没有任何神秘之处，就是普通的数字矩阵：

```python
# 用 safetensors 库加载一个真实的 LLaMA 权重文件
from safetensors import safe_open
import torch

with safe_open("model.safetensors", framework="pt") as f:
    # 取出第0层注意力的 Q 投影权重
    wq = f.get_tensor("model.layers.0.self_attn.q_proj.weight")
    print(wq.shape)   # torch.Size([4096, 4096])
    print(wq.dtype)   # torch.bfloat16
    print(wq[:3, :5]) # 打印前3行5列
```

实际输出大概是这样：
```
tensor([[-0.0042,  0.0031, -0.0019,  0.0058, -0.0024],
        [ 0.0067, -0.0083,  0.0011,  0.0029, -0.0045],
        [-0.0015,  0.0038, -0.0071,  0.0022,  0.0019]], dtype=torch.bfloat16)
```

**关键观察**：数值都非常小，大多在 ±0.01 范围内，看起来像随机噪声——但它们蕴含了整个模型的"知识"。

---

## 二、LLaMA-7B 的所有权重一览

LLaMA-7B（32层，hidden=4096，num_heads=32，FFN=11008）的权重文件完整列表：

```
权重名称                                  形状              字节大小（BF16）
────────────────────────────────────────────────────────────────────────
model.embed_tokens.weight               [32000, 4096]       262 MB
model.norm.weight                       [4096]              8 KB
lm_head.weight                          [32000, 4096]       262 MB（与 embed_tokens 共享）

# 以下每层重复 × 32 层：
model.layers.0.input_layernorm.weight   [4096]              8 KB
model.layers.0.self_attn.q_proj.weight  [4096, 4096]        32 MB
model.layers.0.self_attn.k_proj.weight  [4096, 4096]        32 MB
model.layers.0.self_attn.v_proj.weight  [4096, 4096]        32 MB
model.layers.0.self_attn.o_proj.weight  [4096, 4096]        32 MB
model.layers.0.post_attention_layernorm.weight [4096]        8 KB
model.layers.0.mlp.gate_proj.weight     [11008, 4096]       86 MB
model.layers.0.mlp.up_proj.weight       [11008, 4096]       86 MB
model.layers.0.mlp.down_proj.weight     [4096, 11008]       86 MB
────────────────────────────────────────────────────────────────────────
总计（BF16）：约 13 GB
```

---

## 三、各类权重的实际数值特征

### 3.1 Embedding 权重（embed_tokens.weight）

**形状**：`[vocab_size, hidden_size]` = `[32000, 4096]`（LLaMA-7B）

每一行是一个 token 对应的嵌入向量。GPT-2 的 embedding 权重（`[50257, 768]`）统计示例：

```python
from transformers import GPT2Model
model = GPT2Model.from_pretrained('gpt2')
wte = model.transformer.wte.weight  # [50257, 768]

print(f"Shape : {wte.shape}")          # torch.Size([50257, 768])
print(f"Mean  : {wte.mean():.6f}")     # ≈ 0.000012（极接近0）
print(f"Std   : {wte.std():.6f}")      # ≈ 0.017（初始化时 ≈ 0.02，训练后略有变化）
print(f"Min   : {wte.min():.4f}")      # ≈ -0.15
print(f"Max   : {wte.max():.4f}")      # ≈ 0.15
```

具体数值示例（取前 3 个 token，前 8 维）：

```
Token 0 ("!"): [ 0.0119, -0.0023,  0.0087, -0.0134,  0.0201,  0.0045, -0.0067,  0.0189, ...]
Token 1 ("""): [-0.0089,  0.0156, -0.0211,  0.0033, -0.0088,  0.0124,  0.0178, -0.0045, ...]
Token 2 ("#"): [ 0.0034, -0.0112,  0.0076,  0.0098, -0.0167,  0.0211, -0.0023,  0.0055, ...]
```

**直觉**：每个 token 的 768/4096 维向量就是它的"词义坐标"，相近语义的词（"cat" 和 "dog"）的向量余弦相似度接近 1。

### 3.2 注意力 Q/K/V 投影权重

**形状**：`[out_features, in_features]` = `[4096, 4096]`（LLaMA-7B，Q proj）

```python
# 用 safetensors 检查第0层 Q proj
with safe_open("model-00001-of-00002.safetensors", framework="pt") as f:
    wq = f.get_tensor("model.layers.0.self_attn.q_proj.weight")

# 典型统计
print(f"Shape : {wq.shape}")       # [4096, 4096]
print(f"Mean  : {wq.float().mean():.6f}")  # ≈ -0.000003（极接近0）
print(f"Std   : {wq.float().std():.6f}")   # ≈ 0.0045（小，但比初始化时略大）
print(f"Min   : {wq.float().min():.4f}")   # ≈ -0.08
print(f"Max   : {wq.float().max():.4f}")   # ≈ 0.08
```

实际数值（前5行5列）：

```
wq[:5, :5] =
[[-0.0031,  0.0052, -0.0018,  0.0067,  0.0023],
 [ 0.0044, -0.0071,  0.0039, -0.0028,  0.0056],
 [-0.0019,  0.0033, -0.0088,  0.0015, -0.0043],
 [ 0.0072, -0.0016,  0.0054, -0.0061,  0.0028],
 [-0.0048,  0.0085, -0.0022,  0.0037, -0.0069]]
```

**注意**：所有数值都很小（量级 ~0.001 ~ 0.01）。这不是随机巧合，而是设计使然——如果权重太大，前向传播中矩阵连乘会让激活值指数级增大，导致梯度爆炸。

### 3.3 RMSNorm 权重（最特殊的）

**形状**：`[hidden_size]` = `[4096]`（一维向量！）

```python
# input_layernorm.weight 的初始值
norm_w = model.state_dict()["model.layers.0.input_layernorm.weight"]
print(f"Shape : {norm_w.shape}")   # [4096]
print(f"初始值: 全 1.0")

# 训练后的典型值
print(norm_w[:10])
# tensor([1.0234, 0.9921, 1.0156, 0.9805, 1.0312, 0.9844, 1.0078, 1.0000,
#         0.9922, 1.0195])
```

**RMSNorm 公式**：`output = (x / RMS(x)) * weight`

初始化为全 1（不缩放），训练后数值在 0.8 ~ 1.2 左右，偏离 1.0 不多。这些小数值代表模型学会了在归一化后对每个维度进行微调。

### 3.4 FFN 权重（最大的矩阵）

**形状**：`[ffn_size, hidden_size]` = `[11008, 4096]`（LLaMA-7B gate_proj）

```python
# gate_proj.weight 是 SwiGLU 的门控部分
gate_w = f.get_tensor("model.layers.0.mlp.gate_proj.weight")
print(f"Shape      : {gate_w.shape}")      # [11008, 4096]
print(f"参数量      : {gate_w.numel():,}")  # 45,088,768（4500万个数字！）
print(f"内存        : {gate_w.numel()*2/1e6:.1f} MB")  # 86.0 MB（BF16）
print(f"Std        : {gate_w.float().std():.4f}")        # ≈ 0.0044
```

单个矩阵就有 4500 万个数字，但存储只需 86 MB（BF16 每个数字 2 字节）。

---

## 四、权重分布的可视化理解

### 4.1 初始化时（均匀分布的高斯噪声）

```
频率
 ▲
 │         ████
 │       ████████
 │     ████████████
 │   ████████████████
 │ ████████████████████
 └─────────────────────────→ 权重值
   -0.04  -0.02   0.0   0.02  0.04

Xavier 初始化（std ≈ 0.01 ~ 0.02）：完美的高斯钟形曲线，均值 = 0
```

### 4.2 训练后（略带重尾）

```
频率
 ▲
 │         ██████
 │       ██████████
 │     ██████████████      ← 主体仍是高斯
 │   ████████████████████
 │ ██████████████████████████  ← 尾部更重（少数权重值更大）
 └──────────────────────────────→ 权重值
   -0.15  -0.05  0.0  0.05  0.15

训练后：均值 ≈ 0，std 略有增大，出现少量"异常大"的权重
```

### 4.3 "超级权重"现象（训练后的特殊结构）

研究发现，在 LLaMA-7B 中存在**一个特殊的标量权重**（称为"Super Weight"），删除它会导致模型完全丧失生成能力，困惑度从正常值飙升到极高。

```python
# 在 LLaMA-7B 权重中，某一层的某个权重绝对值异常大
# 典型的"异常激活"出现在固定的 token 位置和固定的 hidden 维度
# 例如在 hidden state 的某几个固定维度上，激活值远大于其他维度：

# 正常 hidden state 分布：
# dim 0..4095：激活值 ≈ [-2.0, 1.5]

# 异常维度（约10个）：
# dim 42：激活值 ≈ 1500  ← 比正常值大 1000 倍！
# dim 1337：激活值 ≈ -800
```

这些"massive activations"是现代 LLM 的真实现象，它们让量化（INT8/INT4）变得困难——因为需要很大的数值范围来同时表示 0.001 和 1500。

---

## 五、用代码直接检查真实权重

### 5.1 标准方式（需要下载模型）

```python
from safetensors import safe_open
import torch
import numpy as np

# 方式1：用 safetensors 库直接读取
def inspect_weights(filepath):
    with safe_open(filepath, framework="pt") as f:
        print(f"文件中的张量数量：{len(f.keys())}")
        print(f"{'名称':<60} {'形状':<25} {'数据类型'}")
        print("-" * 100)
        for key in list(f.keys())[:20]:  # 只打印前20个
            tensor = f.get_tensor(key)
            print(f"{key:<60} {str(list(tensor.shape)):<25} {tensor.dtype}")

# 方式2：用 transformers 加载后检查
from transformers import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf",
                                          torch_dtype=torch.bfloat16)
for name, param in model.named_parameters():
    print(f"{name}: shape={param.shape}, "
          f"mean={param.float().mean():.4f}, "
          f"std={param.float().std():.4f}")
```

### 5.2 命令行工具快速检查

```bash
# 用 safetensors_explorer（交互式 TUI）
pip install safetensors-explorer
safetensors-explorer model.safetensors

# 输出示例：
# ▼ 📁 model (291 tensors, 12.9 GB)
#   ▼ 📁 layers
#     ▼ 📁 0 (13 tensors, 354 MB)
#       📄 self_attn.q_proj.weight  [BFloat16, (4096, 4096), 32.0 MB]
#       📄 self_attn.k_proj.weight  [BFloat16, (4096, 4096), 32.0 MB]
#       📄 self_attn.v_proj.weight  [BFloat16, (4096, 4096), 32.0 MB]
#       📄 self_attn.o_proj.weight  [BFloat16, (4096, 4096), 32.0 MB]
#       📄 mlp.gate_proj.weight     [BFloat16, (11008, 4096), 86.0 MB]
#       📄 mlp.up_proj.weight       [BFloat16, (11008, 4096), 86.0 MB]
#       📄 mlp.down_proj.weight     [BFloat16, (4096, 11008), 86.0 MB]
#       📄 input_layernorm.weight   [BFloat16, (4096,),  8.0 KB]
#       📄 post_attention_layernorm.weight [BFloat16, (4096,), 8.0 KB]
```

---

## 六、不同数据类型下的权重表示

同一个权重，用不同精度存储，数值范围和精度不同：

```python
w = torch.tensor([-0.00312, 0.00518, -0.00089])

# FP32（全精度）：4 bytes/元素
w_fp32 = w.to(torch.float32)
# tensor([-0.00312000, 0.00518000, -0.00089000])  精确

# BF16（Brain Float 16）：2 bytes/元素，vLLM 默认
w_bf16 = w.to(torch.bfloat16)
# tensor([-0.00311279, 0.00518799, -0.00088501])  轻微精度损失

# FP16：2 bytes/元素，精度与 BF16 相近但指数位更少
w_fp16 = w.to(torch.float16)
# tensor([-0.00312042, 0.00518036, -0.00089025])

# INT8（量化）：1 byte/元素，精度较低
# 通常通过 scale + zero_point 量化：
scale = w.abs().max() / 127
w_int8 = (w / scale).round().to(torch.int8)
# tensor([-77, 128, -22], dtype=torch.int8)  + scale=4.07e-5

# INT4（极端量化）：0.5 byte/元素，精度损失明显
# 只有 -8 ~ 7 共 16 个值
```

**BF16 vs FP16 的关键区别**：

```
FP16：1位符号 + 5位指数 + 10位尾数 → 能表示 [-65504, 65504]
BF16：1位符号 + 8位指数 +  7位尾数 → 能表示 [-3.4e38, 3.4e38]

→ BF16 数值范围与 FP32 相同，不容易溢出
→ FP16 精度略高，但范围小，大权重矩阵乘法容易溢出
→ vLLM 默认用 BF16（更稳定）
```

---

## 七、权重从"随机噪声"到"有知识"的变化

### 7.1 未训练的权重（随机初始化）

```python
import torch.nn as nn

# Xavier 均匀初始化（适合 Sigmoid/Tanh）
linear = nn.Linear(4096, 4096)
nn.init.xavier_uniform_(linear.weight)
# std ≈ sqrt(2 / (4096 + 4096)) ≈ 0.0156

# Kaiming 正态初始化（适合 ReLU/SiLU）
nn.init.kaiming_normal_(linear.weight, mode='fan_in', nonlinearity='relu')
# std ≈ sqrt(2 / 4096) ≈ 0.022

# GPT-2 的简单方式（全模型统一用小高斯）
nn.init.normal_(linear.weight, mean=0, std=0.02)

# 此时权重没有任何结构，就是纯高斯噪声：
# mean ≈ 0.000001（数值意义上的0）
# std  ≈ 0.02
# 模型输出：完全无意义的随机 token 概率
```

### 7.2 训练完成后的权重

权重在数值上变化不大（std 从 0.02 变成 ~0.005），但**内部结构完全不同**：

```python
# 训练后的 Q_proj 权重（4096×4096 矩阵）
# 做奇异值分解（SVD）可以看到结构：
U, S, Vh = torch.linalg.svd(wq.float())
print(S[:10])  # 前10个奇异值
# tensor([12.45, 9.83, 8.67, 7.21, 6.44, 5.98, 5.12, 4.87, 4.23, 3.99])
# 奇异值不均匀，说明矩阵有低秩结构（LoRA就是利用这一点做压缩）

# 对比：随机初始化权重的奇异值
U, S_rand, Vh = torch.linalg.svd(torch.randn(4096, 4096) * 0.02)
print(S_rand[:10])
# tensor([1.284, 1.282, 1.281, 1.279, 1.278, ...])  # 所有奇异值几乎相等（无结构）
```

**关键洞察**：训练后的权重矩阵有**低秩结构**——少数奇异值特别大，大部分很小。这正是 LoRA（Low-Rank Adaptation）能有效工作的原因：只需更新少量低秩分量就能适应新任务。

---

## 八、各层权重的直觉理解

### Embedding 矩阵：语义地图

```
"king" 的嵌入向量 ≈ "queen" 的嵌入向量 - "man"向量 + "woman"向量
（经典的 word2vec 类比关系，transformer embedding 也有类似结构）
```

权重值本身毫无直觉意义（-0.0031 这个数字什么都说明不了），但**向量间的相对关系**包含了语言学知识。

### Q/K 投影：注意力模式的编码

```
# 某个注意力头的 Q_head0 投影（取出第0个头的部分）
Q_head0 = wq[:128, :]  # [128, 4096]

# 这个 128×4096 矩阵"教会"了这个头去关注什么
# 例如某些头学会关注句法关系（主语-谓语），
# 另一些头学会关注语义相似性
# 但从数值上看，这些权重仍然是一堆 0.001 量级的小数
```

### FFN 权重：知识的存储

```
# GPT-2 的 FFN 权重被解读为"键值记忆"（Key-Value Memories）
# fc1（gate）权重的每一行 ≈ 一个"模式匹配器"
# fc2（down）权重的每一列 ≈ 一个"事实输出器"
# 例如：激活 ffn[i] ≈ 让模型输出"法国首都是巴黎"中的"巴黎"

# 但这只是一种解释视角，真实权重值仍然是普通的浮点数
```

### LayerNorm/RMSNorm 权重：学来的缩放比例

```
# 初始值：全 1
# 训练后：大约在 [0.8, 1.5] 范围
# 作用：对某些维度放大（值 > 1），对某些维度缩小（值 < 1）
# 这让模型能够"强调"它认为重要的 hidden state 维度
```

---

## 九、权重文件的物理组织

一个典型的 LLaMA-7B safetensors 文件结构：

```
model-00001-of-00002.safetensors  （约 9.9 GB）
model-00002-of-00002.safetensors  （约 3.5 GB）

# 文件格式：
# [8 bytes: header_size（小端序 uint64）]
# [header_size bytes: JSON header（每个 tensor 的 name/dtype/shape/offset）]
# [数据区: 所有 tensor 连续排列的原始字节]

# 例如 JSON header 片段：
{
  "model.layers.0.self_attn.q_proj.weight": {
    "dtype": "BF16",
    "shape": [4096, 4096],
    "data_offsets": [0, 33554432]   # 字节偏移 [start, end]
  },
  "model.layers.0.self_attn.k_proj.weight": {
    "dtype": "BF16",
    "shape": [4096, 4096],
    "data_offsets": [33554432, 67108864]
  }
  ...
}
```

因为有偏移表，可以直接跳转到任意权重的位置读取，**不需要顺序扫描整个文件**——这是 vLLM 加载速度快的底层原因之一。

---

## 十、一个可以实际运行的完整示例

如果你想在 Python 中看到真实的权重数值（无需下载大模型），可以用这个小例子：

```python
import torch
import torch.nn as nn
from safetensors.torch import save_file, load_file

# 1. 创建一个迷你 transformer 层（仿 LLaMA 结构）
class MiniAttention(nn.Module):
    def __init__(self, hidden=64, heads=4):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.RMSNorm(hidden)
        # Kaiming 初始化
        for layer in [self.q, self.k, self.v, self.o]:
            nn.init.kaiming_normal_(layer.weight, mode='fan_in')

model = MiniAttention(hidden=64, heads=4)

# 2. 保存权重
save_file(model.state_dict(), "/tmp/mini_attn.safetensors")

# 3. 加载并查看
weights = load_file("/tmp/mini_attn.safetensors")
for name, tensor in weights.items():
    t = tensor.float()
    print(f"\n{name}: {list(tensor.shape)}")
    print(f"  mean={t.mean():.4f}, std={t.std():.4f}, "
          f"min={t.min():.4f}, max={t.max():.4f}")
    # 打印 3×3 的数值样本
    if tensor.dim() == 2:
        print(f"  前3×3:\n{tensor[:3, :3].float()}")

# 4. 模拟一次简单的前向传播，观察激活值分布
x = torch.randn(1, 10, 64)  # [batch=1, seq=10, hidden=64]
q = model.q(x)
print(f"\nQ 激活值: mean={q.mean():.4f}, std={q.std():.4f}")
print(f"前 3 个 token 的 Q 向量前 5 维:")
print(q[0, :3, :5])
```

期望输出：

```
q.weight: [64, 64]
  mean=0.0001, std=0.1251, min=-0.3421, max=0.3567
  前3×3:
tensor([[ 0.1234, -0.0892,  0.2156],
        [-0.1678,  0.0445, -0.0923],
        [ 0.0789, -0.2134,  0.1567]])

norm.weight: [64]
  mean=1.0000, std=0.0000, min=1.0000, max=1.0000  ← 初始全1

Q 激活值: mean=-0.0023, std=0.9987  ← Kaiming 初始化保证激活方差≈1
```

---

## 十一、为什么权重值这么小？

这涉及**数值稳定性**的根本设计原则：

```
Transformer 有 N 层，每层都是矩阵乘法：
  h_out = W × h_in

如果 W 的方差是 σ²，h_in 的每个元素方差是 1，
那么 h_out 的每个元素方差 = input_dim × σ²

LLaMA-7B: hidden_size=4096, 32层
如果 σ = 0.1：
  第1层后方差 = 4096 × 0.01 = 40.96
  第2层后方差 = 4096 × 40.96 = 167,772
  第32层后：爆炸到无穷 → 梯度消失/爆炸

Xavier 均匀初始化：σ² = 2 / (fan_in + fan_out) = 2 / 8192 ≈ 0.000244
  → σ ≈ 0.0156
  → 每层后方差 = 4096 × 0.000244 ≈ 1.0  ← 保持方差稳定！

这就是为什么权重初始化的 std 必须 ∝ 1/√(input_dim)
```

---

## 十二、总结

| 权重类型 | 形状（7B为例） | 典型数值范围 | 初始值 | 训练后变化 |
|---------|--------------|------------|--------|-----------|
| Embedding | [32000, 4096] | ±0.15 | N(0, 0.02) | 出现语义结构，分布略变宽 |
| Q/K/V proj | [4096, 4096] | ±0.08 | N(0, 0.02) | 低秩结构形成，少数大奇异值 |
| O proj | [4096, 4096] | ±0.08 | N(0, 0.02) | 类似Q/K/V |
| FFN gate/up | [11008, 4096] | ±0.06 | N(0, 0.02) | 出现稀疏激活模式 |
| FFN down | [4096, 11008] | ±0.06 | N(0, 0.02) | 类似gate/up |
| RMSNorm | [4096] | [0.8, 1.5] | 全1.0 | 轻微偏离1，对重要维度放大 |
| LM head | [32000, 4096] | ±0.15 | 与embedding共享 | 同embedding |

**核心理解**：
- 权重就是浮点数数组，本身没有语义含义
- 数值小（~0.01量级）是数学稳定性的必然结果
- 权重的"知识"存储在矩阵的**相对结构**中，而非绝对数值
- 训练前后数值变化很小，但内在结构从随机变成有序

---

*参考资料：*
- *[GPT-2 嵌入权重探索](https://www.alignmentforum.org/posts/BMghmAxYxeSdAteDc/an-exploration-of-gpt-2-s-embedding-weights)*
- *[LLM 模型权重是什么？](https://blog.gopenai.com/what-are-the-llama-model-weights-e83a58cef1be)*
- *[Safetensors 官方文档](https://huggingface.co/docs/safetensors/en/index)*
- *[超级权重研究](https://arxiv.org/html/2411.07191v1)*
- *[大型权重研究（House of Cards）](https://arxiv.org/html/2410.01866v1)*
- *[karpathy/llm.c GPT-2 实现](https://github.com/karpathy/llm.c/blob/master/train_gpt2.py)*
- *[safetensors_explorer CLI 工具](https://github.com/EricLBuehler/safetensors_explorer)*
*更新：2026-03*
