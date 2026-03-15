# 优化器是什么？工业界有哪些优化器？

> 优化器（Optimizer）是神经网络训练的"驾驶员"：
> 梯度告诉你"坡在哪里"，优化器决定"怎么走"。

---

## 一、优化器的本质

### 1.1 定义

**优化器**：给定当前参数 $W$ 和梯度 $g = \nabla_W L$，计算下一步参数 $W'$ 的规则。

$$W' = \text{Optimizer}(W, g, \text{state})$$

其中 `state` 是优化器的历史记忆（如动量、梯度平方均值等）。

最简单的规则（梯度下降）：

$$W' = W - \eta \cdot g$$

优化器的进化方向：**用更少的步数、更稳定地找到更好的参数**。

### 1.2 优化器解决的三个核心问题

```
问题 1：方向问题（Direction）
  单个 batch 的梯度方向有噪声，不可靠
  → 解法：动量（Momentum）——平滑梯度方向

问题 2：步长问题（Step Size）
  不同参数的最优步长差异极大
  → 解法：自适应学习率（Adaptive LR）——每个参数独立步长

问题 3：曲率问题（Curvature）
  损失曲面在不同方向曲率不同，理想步长与曲率成反比
  → 解法：二阶优化（Second-Order）——利用 Hessian 信息
```

---

## 二、优化器的演化谱系

```
梯度下降（1847，Cauchy）
  └── SGD（随机梯度下降）
        └── SGD + Momentum（动量）
              ├── NAG（Nesterov 加速梯度）
              └── AdaGrad（2011，自适应学习率）
                    └── RMSProp（2012，Hinton，修复 AdaGrad）
                          └── Adam（2014，Kingma & Ba）
                                ├── AdamW（2017，解耦权重衰减）
                                ├── AMSGrad（2018，修复 Adam 不收敛）
                                ├── LAMB（2019，超大批量）
                                ├── AdaFactor（2018，节省内存）
                                ├── Lion（2023，Google Brain）
                                └── Muon（2024，基于 Nesterov）

二阶优化方向：
  牛顿法
    └── L-BFGS（拟牛顿）
          └── K-FAC（2015，Kronecker 因子近似曲率）
                └── Sophia（2023，Stanford，二阶近似）
                      └── SOAP（2024，二阶 + Adam 预条件）
```

---

## 三、基础优化器

### 3.1 SGD（随机梯度下降）

$$W_{t+1} = W_t - \eta \cdot g_t$$

```
"随机"的含义：
  不是用整个数据集的梯度（太慢）
  而是用随机抽取的一个 mini-batch 的梯度（"随机"近似）

优点：
  ✓ 极简，内存只需存参数本身
  ✓ 单步计算量最小
  ✓ 噪声有助于逃脱鞍点，找到更平坦的最小值

缺点：
  ✗ 收敛慢（Z 字形游走）
  ✗ 学习率极难调（对所有参数相同）
  ✗ 不适合非平稳目标（如 RL）

适用场景：
  图像分类（ResNet、ViT fine-tuning）
  研究中需要可控泛化性时
  不适合 LLM 训练
```

### 3.2 SGD + Momentum（动量）

$$v_t = \mu \cdot v_{t-1} + g_t$$
$$W_{t+1} = W_t - \eta \cdot v_t$$

```
物理直觉：
  球在曲面上滚动时有惯性（动量 μ，通常 0.9）
  方向一致的梯度会累积加速
  方向不一致的梯度会相互抵消

效果：
  ✓ 在"山谷"方向（梯度一致）加速下降
  ✓ 在垂直方向（梯度来回震荡）平滑振荡

对比 SGD：
  SGD：      Z─Z─Z─Z─Z─→ 慢，来回震荡
  Momentum：───────────→ 快，沿山谷直线前进

内存：v（动量，与 W 同形）→ 额外 1 份参数大小的内存
```

### 3.3 NAG（Nesterov Accelerated Gradient）

$$v_t = \mu \cdot v_{t-1} + g(W_t - \mu \cdot v_{t-1})$$
$$W_{t+1} = W_t - \eta \cdot v_t$$

```
核心改进：
  普通 Momentum：先看当前位置的梯度，再加速
  NAG：先"看"下一步位置的梯度（展望未来），再决定怎么走

直觉：
  走到快要撞墙时，提前刹车（而不是撞上再刹）
  → 收敛更平滑，振荡更少

Muon 优化器（2024）是 NAG 的现代版本，
在 LLM 训练中开始出现
```

---

## 四、自适应学习率优化器

### 4.1 AdaGrad（Adaptive Gradient）

$$G_t = G_{t-1} + g_t^2 \quad \text{（梯度平方的累积和）}$$
$$W_{t+1} = W_t - \frac{\eta}{\sqrt{G_t} + \epsilon} \cdot g_t$$

```
核心思想：
  频繁更新的参数（G 大）→ 学习率自动缩小
  稀少更新的参数（G 小）→ 学习率自动放大

适用场景：
  NLP 中的稀疏特征（词嵌入：常见词更新多，罕见词更新少）
  推荐系统（用户/物品 ID 特征极稀疏）

致命问题：
  G_t 是单调递增的（只加不减）
  → 训练足够长后，所有参数的 lr 趋近于 0
  → 训练停滞！

  无法用于长时间训练（LLM 训练数百万步），被 RMSProp/Adam 取代
```

### 4.2 RMSProp（Root Mean Square Propagation）

$$v_t = \rho \cdot v_{t-1} + (1 - \rho) \cdot g_t^2 \quad \text{（指数移动平均，而非累积和）}$$
$$W_{t+1} = W_t - \frac{\eta}{\sqrt{v_t} + \epsilon} \cdot g_t$$

```
Hinton 在 Coursera 课程上提出（从未正式发表过论文！）

核心修复：
  将 AdaGrad 的"累积和"改为"指数移动平均"
  → v_t 不会无限增长，会"遗忘"旧的梯度信息
  → 适应非平稳目标（lr 不会趋于 0）

参数 ρ = 0.9（保留 90% 的历史，丢弃 10%）

RMSProp = Adam 的"无动量版本"
  Adam = Momentum（m_t）+ RMSProp（v_t）的结合

仍在使用的场景：
  强化学习（RNN 控制器）
  某些在线学习场景
```

---

## 五、Adam 及其变体（当前主流）

### 5.1 Adam（Adaptive Moment Estimation）

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{（一阶矩，梯度均值）}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{（二阶矩，梯度方差）}$$
$$\hat{m}_t = m_t / (1 - \beta_1^t) \quad \text{（偏差修正）}$$
$$\hat{v}_t = v_t / (1 - \beta_2^t)$$
$$W_{t+1} = W_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$$

```
标准超参数：
  η  = 3e-4（学习率，需调整）
  β₁ = 0.9  （动量系数，很少改变）
  β₂ = 0.999（二阶矩系数，很少改变）
  ε  = 1e-8 （数值稳定项）

为什么需要偏差修正？
  初始时 m₀ = v₀ = 0（冷启动）
  前几步 m_t 和 v_t 被严重低估（偏向 0）
  除以 (1-β^t) 修正这个偏差

  第 1 步：(1-0.9¹) = 0.1，修正因子 = 1/0.1 = 10
  第 10 步：(1-0.9^10) ≈ 0.65，修正因子 ≈ 1.54
  第 100 步：修正因子 ≈ 1.00（几乎不需要修正了）

内存开销：
  参数 W：1 份
  一阶矩 m：1 份（与 W 同形，FP32）
  二阶矩 v：1 份（与 W 同形，FP32）
  合计：参数量 × 3（若含梯度则 × 4）

  对于 7B 模型（BF16 参数 = 14GB）：
  Adam 状态（FP32）= 7B × 2 × 4B = 56GB
  梯度（BF16）= 14GB
  总训练内存 ≈ 14 + 56 + 14 = 84GB（单卡 A100-80G 几乎放不下！）
  → 这就是为什么需要 ZeRO / 混合精度 / 梯度累积
```

**Adam 的问题（为什么需要 AdamW）**：

```
L2 正则化在 Adam 中失效：

标准 L2：loss = L(W) + λ||W||²
梯度：g = ∇L + 2λW

Adam 更新：W -= η × m / √v
其中 m 和 v 都包含了 2λW 这一项
→ 正则化效果被 1/√v 缩放，强度变得不可预测
→ 不同参数（v 不同）受到完全不同强度的正则化！
```

### 5.2 AdamW（Adam with Decoupled Weight Decay）★ 当前 LLM 首选

$$W_{t+1} = W_t - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon) - \eta \cdot \lambda \cdot W_t$$

```
修复：将 weight decay 从梯度中解耦，直接作用于参数

对比：
  Adam（L2 正则化）：∇loss = g + 2λW → 送入 m/v 估计
  AdamW（解耦 WD）：先 Adam 步，再减去 λW → 正则化固定强度

AdamW 的 weight decay：
  λ = 0.1（LLaMA-3 配置）
  效果：每步 W × (1 - η × λ) ≈ W × 0.99997（步长 3e-4 时）
  含义：权重在每步都轻微向 0 收缩（偏好小权重 = 平坦最小值）

工业界使用情况：
  LLaMA-1/2/3（Meta）：AdamW
  GPT-3/4（OpenAI）：Adam/AdamW
  Gemini（Google）：AdamW
  DeepSeek-V2/V3：AdamW
  Mistral/Mixtral：AdamW
  Falcon：AdamW
  几乎所有主流 LLM 预训练：AdamW
```

### 5.3 Adam 的已知问题

```
问题 1：不保证收敛（AMSGrad 修复）
  Adam 使用 v_t 的指数移动平均，可能在非凸问题中不收敛
  AMSGrad：用 max(v̂_{t-1}, v̂_t) 替代 v̂_t（单调不减）
  实践中差异不大，AMSGrad 未成为主流

问题 2：内存占用大（见上文，3× 参数大小）
  AdaFactor / CAME 等尝试解决

问题 3：泛化性比 SGD 差（某些场景）
  Adam 倾向于找到"更尖锐"的最小值
  LLM 超大规模训练中此问题不明显（随机性够大）

问题 4：对 lr 敏感
  lr 过大：不稳定；lr 过小：收敛慢
  → Warmup + Decay 是必须的配套方案
```

### 5.4 AdaFactor（节省内存的 Adam 替代）

```
核心思想：
  Adam 的 v_t 是全矩阵（参数量大小）
  AdaFactor：用低秩近似代替

  对于矩阵参数 W: [m, n]：
  Adam v_t: [m, n]（完整矩阵，m×n 个数）
  AdaFactor：v_t ≈ r_t · c_t^T（r: [m]，c: [n]，只需 m+n 个数）

内存节省：
  7B 参数模型：Adam v = 56GB → AdaFactor v ≈ 几百 MB

  代价：近似导致优化质量略下降

使用情况：
  T5（Google）：AdaFactor（当时 GPU 内存有限）
  现代 LLM（内存充足时）：普遍改回 AdamW
  内存极限场景（如 100B+ 模型单机训练）：仍有使用
```

---

## 六、大批量专用优化器

### 6.1 LARS（Layer-wise Adaptive Rate Scaling）

```
背景：
  大批量训练（Batch > 4096）时，线性 lr 缩放规则失效
  直接用大 lr 导致训练不稳定

核心思想：
  为每一层单独计算学习率
  lr_layer = η × ||W_layer|| / ||g_layer||

  含义：参数范数 / 梯度范数
  → 参数大而梯度小的层：lr 大
  → 参数小而梯度大的层：lr 小
  → 保持每层的更新幅度 / 参数幅度 ≈ 常数

使用场景：
  ImageNet 训练（ResNet，batch=32768+）
  Google 的 TPU 集群上的大规模视觉训练
  LLM 训练中较少使用（LLM 通常不用极端大批量）
```

### 6.2 LAMB（Layer-wise Adaptive Moments for Batch）

$$\text{LAMB} = \text{Adam 更新方向} + \text{LARS 的 per-layer lr 缩放}$$

```
公式：
  先按 Adam 计算更新 Δ = m / √v（方向 + 自适应幅度）
  再按 LARS 缩放：update = η × ||W|| / ||Δ|| × Δ

使用场景：
  BERT 预训练（超大批量，batch=65536）
  Google 的 LLM 超大批量训练
  TPU Pod 上的大规模并行训练（DP 极大时）

实验效果：
  BERT：batch=256 需要 3 天
  BERT + LAMB：batch=65536 → 76 分钟！
  → 训练速度提升 100 倍（靠更大批量 + LAMB 稳定训练）
```

---

## 七、新一代优化器（2022-2025）

### 7.1 Lion（EvoLved Sign Momentum，Google Brain 2023）

论文：*Symbolic Discovery of Optimization Algorithms*（Chen et al. 2023）

```
核心公式（极其简洁）：

  c_t = β₁ · m_{t-1} + (1 - β₁) · g_t    （候选动量）
  W_t = W_{t-1} - η · (sign(c_t) + λ · W_{t-1})  （只用符号！）
  m_t = β₂ · m_{t-1} + (1 - β₂) · g_t    （更新动量）

  β₁ = 0.9（用于计算更新方向）
  β₂ = 0.99（用于维护动量，与 Adam 的 β₁ 角色互换）

关键创新：
  只用梯度的"符号"（+1 或 -1），不用梯度的幅度
  → 所有参数更新幅度相同（均为 η 或 -η）

内存优势：
  Adam：m（FP32）+ v（FP32）= 2× 参数量
  Lion：只需 m（FP32）= 1× 参数量
  节省 33% 的优化器状态内存

效果：
  在视觉 Transformer、语言模型微调中
  相同计算量下，Lion 比 AdamW 效果更好或相当
  用更小的 lr（约 Adam 的 1/3~1/10）配合更大的 weight decay

工业界使用：
  Google DeepMind：内部部分模型使用
  字节跳动：Doubao/MegaScale 研究中有探索
  学术界：越来越多的论文采用

注意：
  LLM 超大规模预训练中，Lion 尚未完全取代 AdamW
  Adam 的"幅度感知"在大型 MoE 等复杂架构中仍有优势
```

### 7.2 Sophia（Second-Order Clipped Stochastic Optimization）

论文：*Sophia: A Scalable Stochastic Second-Order Optimizer for LLM Pre-training*（Liu et al. 2023，Stanford）

```
核心思想：用 Hessian 对角线近似曲率信息

Adam 的问题：v_t = E[g²] 估计的是"梯度幅度"，不是"曲率"
Sophia 的改进：直接估计 Hessian 对角线 h_t = ∂²L/∂W²（真实曲率）

更新规则：
  m_t = β₁ · m_{t-1} + (1 - β₁) · g_t    （动量，与 Adam 一样）
  h_t = 每 k 步更新一次（Gauss-Newton 近似）
  W_t = W_{t-1} - η · m_t / max(γ · h_t, ε)   （用曲率归一化）

"Clipped"的含义：
  max(γ · h_t, ε) 防止曲率很小时步长过大

效果（GPT-2 训练对比）：
  相同 loss 下，Sophia 比 AdamW 少 50% 的步数
  → 训练速度提升约 2× ！

代价：
  每 k 步（k=10）需要额外一次前向传播估计 Hessian
  → 单步计算量约 AdamW 的 1.1× （因为 k=10 摊销）
  → 内存：额外存储 h_t（与参数同形，FP32）

工业界应用：
  2023-2025 年部分研究团队的预训练实验
  尚未被顶级 LLM 厂商大规模采用（Adam 生态太成熟）
  → 是很有前途的方向，实际部署待观察
```

### 7.3 Muon（Momentum + Newton-Schulz，2024）

```
提出者：Kosson et al. / Jordan et al. 2024

核心思想：
  Nesterov 动量 + 正交化更新

  传统 Adam：更新 = m / √v（每个参数独立归一化）
  Muon：把更新矩阵"正交化"（Newton-Schulz 迭代）

正交化的含义：
  把矩阵更新 ΔW 变换为正交矩阵（行/列单位正交）
  → 每层的更新"均匀影响"所有方向
  → 不会在某些方向过度更新，某些方向几乎不更新

效果：
  在 LLM 训练（特别是 Transformer 的 MLP 层）中
  比 AdamW 更快收敛（相同步数更低的 loss）

工业界使用：
  Modular AI / DeepMind 研究团队有实验
  Kimi（月之暗面）等国内团队有探索
  仍处于研究阶段，未大规模商用

局限：
  正交化仅适用于矩阵参数（不适用于 bias、LayerNorm 等）
  实践中通常与 AdamW 混用：
    MLP/Attention 权重矩阵用 Muon
    其他参数用 AdamW
```

### 7.4 CAME（Confidence-guided Adaptive Memory Efficient Optimization）

```
背景：为了在内存受限场景下替代 AdamW

核心思想：
  AdaFactor 的低秩近似 + Adam 的二阶矩
  + "置信度引导"：用梯度的一致性程度调整更新

使用场景：
  T5、mT5 等 encoder-decoder 架构的训练
  内存受限的 LLM 微调
  不需要完整 Adam 状态的场景
```

### 7.5 GaLore（Gradient Low-Rank Projection，2024）

```
严格说不是一个全新的优化器，而是"梯度压缩"技术

核心思想：
  梯度矩阵是低秩的（训练中梯度主要分布在少数方向）
  → 在低秩子空间中运行 Adam
  → 大幅减少优化器状态内存

效果：
  7B 模型全量训练：Adam 内存 ≈ 80GB → GaLore ≈ 22GB
  同时保持接近全量训练的效果

代价：
  需要定期更新投影矩阵（SVD 分解，有计算开销）
  收敛质量略低于标准 AdamW

应用场景：
  消费级 GPU 上的 LLM 预训练（如单张 A100-80G 训练 7B）
  内存极度受限的学术研究场景
```

### 7.6 SOAP（2024，二阶 + 预条件）

```
核心思想：
  在 Adam 的基础上加入 Shampoo 风格的预条件矩阵
  让更新方向更接近 Newton 方向（最优曲率方向）

Shampoo（Google 2018）的原理：
  对每个参数矩阵 W: [m, n]
  维护左预条件 L: [m, m] 和右预条件 R: [n, n]
  update = L^{-1/2} · g · R^{-1/2}

SOAP = Adam（时间轴上的自适应）+ Shampoo（空间轴上的自适应）

效果：在 GPT-2 scale 的预训练上显著优于 AdamW
代价：L 和 R 矩阵的存储和更新（计算代价高）
应用：仍在研究阶段
```

---

## 八、内存优化技术（非独立优化器，但至关重要）

### 8.1 混合精度（Mixed Precision）

```
标准配置（LLM 训练）：
  参数（前向/反向）：BF16（2 bytes/element）
  Adam m、v 状态：FP32（4 bytes/element）← 需要高精度
  梯度：BF16（2 bytes/element，accumulate in FP32）

  7B 模型内存：
  参数：7B × 2 = 14 GB（BF16）
  m：  7B × 4 = 28 GB（FP32）
  v：  7B × 4 = 28 GB（FP32）
  梯度：7B × 2 = 14 GB（BF16）
  总计：84 GB（单卡 A100-80G 勉强能放下）

为什么 m/v 需要 FP32？
  FP32 的精度更高（23 位尾数 vs BF16 的 7 位）
  Adam 更新：W -= η × m / √v
  如果 η = 3e-4，m/√v ≈ 0.001，则更新量 = 3e-7
  BF16 最小精度 ≈ 2^{-7} × 2^{-14} ≈ 6e-6（无法表示 3e-7！）
  → FP32 m/v 保证更新量不被舍入为 0
```

### 8.2 ZeRO（Zero Redundancy Optimizer，DeepSpeed）

```
ZeRO 是 DP 场景下减少 Adam 状态冗余的技术：

DP=8（8 张卡，每张都有完整模型副本）的内存：
  标准 DP：每张卡都有完整的 W, m, v, g → 每卡 84GB（7B 模型）

  ZeRO-1（分片优化器状态）：
    m 和 v 分片：84GB → 每卡 (14+14+28×2/8) = 35 GB
    梯度仍然 AllReduce，参数仍然完整

  ZeRO-2（分片 m + v + 梯度）：
    每卡：14GB（参数）+ 84/8 = 10.5GB → 共 24.5GB

  ZeRO-3（分片 m + v + 梯度 + 参数）：
    每卡：84GB / 8 = 10.5 GB（极度节省！）
    代价：前向传播每层需要 AllGather 参数（通信增加）

  DeepSeek-V3 使用 ZeRO-1 + 其他优化
```

---

## 九、工业界优化器使用现状

### 9.1 各大公司的选择

```
公司/模型           优化器           特殊配置
─────────────────────────────────────────────────────────────────
Meta / LLaMA 1/2/3  AdamW          β₁=0.9, β₂=0.95, ε=1e-5, wd=0.1
OpenAI / GPT-3      Adam           β₁=0.9, β₂=0.95, ε=1e-8
OpenAI / GPT-4      未公开          推测 AdamW 类
DeepSeek / V2/V3    AdamW          β₁=0.9, β₂=0.95, ε=1e-8, wd=0.1
Mistral / Mixtral   AdamW          标准配置
Google / PaLM       Adafactor      TPU 内存节省
Google / Gemini     AdamW（推测）   —
Anthropic / Claude  AdamW（推测）   —
阿里 / Qwen        AdamW           标准配置
百度 / ERNIE        AdamW           标准配置
月之暗面 / Kimi     AdamW + 实验性  —

结论：AdamW 是当前 LLM 预训练的绝对主流
```

### 9.2 LLaMA-3 的完整优化器配置（官方）

```python
# 来自 LLaMA-3 技术报告
optimizer = AdamW(
    params=model.parameters(),
    lr=3e-4,         # peak learning rate（8B）/ 1.5e-4（70B）
    betas=(0.9, 0.95),  # β₁, β₂（注意 β₂=0.95，不是常见的 0.999！）
    eps=1e-5,           # ε（比常见的 1e-8 更大）
    weight_decay=0.1    # λ
)

# 梯度裁剪
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 学习率调度
scheduler = CosineAnnealingWithWarmup(
    optimizer,
    warmup_steps=2000,
    total_steps=1_000_000,
    min_lr=3e-5  # = peak_lr × 0.1
)
```

**为什么 LLaMA 用 β₂=0.95 而非 0.999？**

```
β₂=0.999（标准 Adam）：
  v_t 的"记忆长度"= 1/(1-0.999) = 1000 步
  → 非常稳定，但对最近梯度变化不敏感

β₂=0.95（LLaMA 配置）：
  v_t 的"记忆长度"= 1/(1-0.95) = 20 步
  → 更快响应梯度变化
  → 在 Warmup 结束后更快调整有效 lr

β₂=0.999 的问题（LLM 训练）：
  Warmup 期间（前 2000 步）梯度很大
  Warmup 结束后梯度变小
  v_t 需要 1000 步才能"忘记"大梯度
  → 前 3000 步有效 lr 偏小，浪费训练效率

β₂=0.95 的解决方案：
  20 步内就能调整，进入正常学习率状态更快
```

### 9.3 DeepSeek-V3 的优化器配置（官方）

```
来自技术报告：
  AdamW（β₁=0.9, β₂=0.95, ε=1e-8, weight_decay=0.1）
  梯度裁剪：max_norm=1.0
  学习率：max_lr=2.2e-4，WSD 调度
  Batch warmup：3072 → 15360 sequences

为什么 671B MoE 的 max_lr=2.2e-4 比 7B 模型更大？
  直觉上应该是大模型 lr 更小，但 MoE 每步激活参数只有 37B
  → 等效参数规模是 37B，不是 671B
  → lr 参照 37B 模型设置（比 7B 略小，比 70B 略大），合理

  另外 WSD 调度末期的快速衰减补偿了相对较大的 lr
```

---

## 十、如何选择优化器？决策框架

```
场景 1：LLM 预训练（主流选择）
  → AdamW（β₁=0.9, β₂=0.95, ε=1e-5, wd=0.1）
  → 理由：稳定、广泛验证、良好的工程支持

场景 2：LLM 预训练 + 内存极度受限
  → AdaFactor 或 AdamW + ZeRO-3 + GaLore
  → 理由：内存节省优先于绝对效果

场景 3：超大批量训练（Batch > 32k）
  → LAMB（BERT 训练）或 AdamW + 线性 LR 缩放
  → 理由：LAMB 在超大批量下更稳定

场景 4：图像/视频模型训练
  → SGD + Momentum（对泛化要求高）或 AdamW
  → 理由：SGD 在视觉任务中有时泛化更好

场景 5：强化学习（RLHF PPO）
  → Adam（小 lr：1e-6）或 RMSProp
  → 理由：RL 梯度方差极大，需要自适应 lr

场景 6：研究/探索新架构
  → Lion 或 Muon（可能更快收敛）
  → 理由：潜在性能提升，但需要验证

场景 7：微调（SFT / LoRA）
  → AdamW（小 lr：2e-5）或 Lion（更大 lr）
  → 理由：不需要大改动，稳定优先
```

---

## 十一、优化器对比速查表

| 优化器 | 年份 | 内存（×参数量）| 收敛速度 | 泛化性 | 主要使用场景 |
|--------|------|--------------|---------|-------|------------|
| SGD | 1847 | 1× | 慢 | 好 | 视觉模型 |
| SGD+Momentum | 1964 | 2× | 中 | 好 | 视觉模型 |
| AdaGrad | 2011 | 2× | 中 | 中 | 稀疏NLP特征 |
| RMSProp | 2012 | 2× | 快 | 中 | RNN/RL |
| Adam | 2014 | 3× | 快 | 中 | 通用 |
| **AdamW** | **2017** | **3×** | **快** | **好** | **LLM 首选** |
| AdaFactor | 2018 | 低秩≈0.01× | 中 | 中 | 内存受限 |
| LAMB | 2019 | 3× | 快 | 中 | 超大批量 |
| Lion | 2023 | 2× | 快 | 好 | 视觉/微调 |
| Sophia | 2023 | 4× | 很快 | 好 | LLM（研究）|
| Muon | 2024 | 2× | 快 | 好 | LLM（研究）|
| GaLore | 2024 | 低秩≈0.3× | 中 | 中 | 内存受限预训练 |
| SOAP | 2024 | 5×+ | 很快 | 好 | 研究阶段 |

---

## 十二、总结

```
优化器的本质：
  梯度 = 方向信息（当前位置的下坡方向）
  优化器 = 如何利用这个方向信息移动参数

优化器演化的三条主线：
  1. 更稳定的方向：Momentum → Adam 的 m_t
  2. 更智能的步长：AdaGrad → RMSProp → Adam 的 v_t
  3. 更好的曲率近似：SGD → Adam → Sophia → SOAP

工业界现状（2025）：
  预训练：AdamW 一统天下（稳定、成熟、全栈支持）
  新兴挑战：Lion（内存效率）、Sophia（收敛速度）、Muon（收敛质量）
  内存优化：ZeRO + GaLore 成为大规模训练的标配

未来趋势：
  二阶优化 + 内存效率的结合（Sophia 方向）
  更多融合"优化器"与"架构"的方案（Muon 针对 Transformer 定制）
  自动搜索优化器（Lion 就是被进化算法搜出来的）
```

---

*参考资料：*
- *[Adam：Kingma & Ba 2014](https://arxiv.org/abs/1412.6980)*
- *[AdamW：Loshchilov & Hutter 2017](https://arxiv.org/abs/1711.05101)*
- *[LAMB：You et al. 2019](https://arxiv.org/abs/1904.00962)*
- *[Lion：Chen et al. 2023，Google Brain](https://arxiv.org/abs/2302.06675)*
- *[Sophia：Liu et al. 2023，Stanford](https://arxiv.org/abs/2305.14342)*
- *[GaLore：Zhao et al. 2024](https://arxiv.org/abs/2403.03507)*
- *[Muon：Kosson et al. 2024](https://arxiv.org/abs/2409.20325)*
- *[SOAP：Vyas et al. 2024](https://arxiv.org/abs/2409.11321)*
- *[ZeRO：Rajbhandari et al. 2020，DeepSpeed](https://arxiv.org/abs/1910.02054)*
- *[AdaFactor：Shazeer & Stern 2018](https://arxiv.org/abs/1802.04210)*
- *[LLaMA-3 技术报告](https://arxiv.org/abs/2407.21783)*
- *[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)*
*更新：2026-03*
