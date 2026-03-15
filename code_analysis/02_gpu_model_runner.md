# vLLM GPU Model Runner 深度解析

> 文件：`vllm/v1/worker/gpu_model_runner.py`（4610 行）
> 定位：GPU 执行层的核心协调器，负责将调度器输出转化为一次 GPU 前向传播

---

## 一、职责定位

GPUModelRunner 是 vLLM V1 架构中最复杂的单个组件，它处于 Worker 和 Model 之间：

```
EngineCore → Scheduler → SchedulerOutput
                              ↓
                    GPUModelRunner.execute_model()
                              ↓
              _update_states → _prepare_inputs → _model_forward → 采样
                              ↓
                    ModelRunnerOutput（含采样的 token ids）
```

它需要同时处理：CUDA Graph 调度、流水线并行、投机解码、多模态输入、异步输出拷贝、前缀缓存……几乎是整个推理系统最复杂的"胶水层"。

---

## 二、初始化：预分配持久缓冲区

```python
# 在 __init__ 中，一次性分配所有推理过程中需要的缓冲区
self.input_ids  = self._make_buffer(self.max_num_tokens, dtype=torch.int32)
self.positions  = self._make_buffer(self.max_num_tokens, dtype=torch.int64)
self.seq_lens   = self._make_buffer(self.max_num_reqs,   dtype=torch.int32)
self.query_start_loc = self._make_buffer(self.max_num_reqs + 1, dtype=torch.int32)
```

### 为什么要预分配？

**CUDA Graph 的核心约束**：CUDA Graph 录制时会捕获所有的内核调用，包括其参数（指针地址）。如果每次推理步都分配新张量，指针地址就会改变，CUDA Graph 无法重复执行。

通过预分配固定地址的缓冲区，每次推理只需更新缓冲区的**内容**，而不改变**地址**，CUDA Graph 可以安全地重放。

### CpuGpuBuffer：同步的 CPU-GPU 双缓冲

```python
def _make_buffer(self, *size, dtype, numpy=True) -> CpuGpuBuffer:
    return CpuGpuBuffer(*size, dtype=dtype, device=self.device,
                        pin_memory=self.pin_memory, with_numpy=numpy)
```

`CpuGpuBuffer` 同时创建：
- **`.cpu`**：CPU 端的 pinned memory 张量（页锁定内存，DMA 直传，不经过 CPU 缓存）
- **`.gpu`**：GPU 端对应的张量（固定地址）
- **`.np`**：NumPy 数组视图（零拷贝，直接操作 CPU 内存）

**为什么要 pinned memory？** 普通 CPU 内存在 GPU-CPU 传输时需要先拷贝到 pinned memory，再 DMA 传输。直接使用 pinned memory 可以消除这次中间拷贝，H2D/D2H 传输速度提升 2~3 倍。

---

## 三、execute_model：三阶段执行

```
┌─────────────────────────────────────────────────────────────────┐
│ Preprocess：CPU 侧数据准备                                       │
│   _update_states()       → 同步请求状态变化                      │
│   _prepare_inputs()      → 构建 attention metadata、positions 等 │
│   cudagraph_dispatcher   → 决定用 CUDA Graph 还是 eager 模式     │
├─────────────────────────────────────────────────────────────────┤
│ Forward：GPU 前向传播                                            │
│   set_forward_context()  → 通过 thread-local 传递 attn_metadata  │
│   _model_forward()       → 调用 model.forward()                 │
├─────────────────────────────────────────────────────────────────┤
│ Postprocess：采样 + 输出处理                                     │
│   sampler()              → 从 logits 采样 token                  │
│   AsyncGPUModelRunnerOutput → 异步拷贝结果到 CPU                 │
└─────────────────────────────────────────────────────────────────┘
```

### @torch.inference_mode()

所有推理在 `inference_mode` 下运行，与 `no_grad()` 类似，但更彻底——不记录自动微分所需的元数据，内存占用和速度都更优。

---

## 四、_prepare_inputs：CPU 侧高速准备

```python
def _prepare_inputs(self, scheduler_output):
    # 1. 立即开始拷贝 block_table（与后续 CPU 计算重叠）
    self.input_batch.block_table.commit_block_table(num_reqs)

    # 2. 准备 num_scheduled_tokens 数组
    tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
    num_scheduled_tokens = np.array(tokens, dtype=np.int32)

    # 3. 计算 positions：每个 token 在其序列中的绝对位置
    # [2, 5, 3] 的 tokens → positions = [0,1, 0,1,2,3,4, 0,1,2]
    positions_np = self.positions.np[:total_num_scheduled_tokens]
    np.add(
        self.input_batch.num_computed_tokens_cpu[req_indices],
        arange,
        out=positions_np
    )

    # 4. 用 torch.index_select 快速散射 token ids（比 np.take 快）
    torch.index_select(
        self.input_batch.token_ids_cpu_tensor.flatten(),
        0,
        token_indices_tensor,
        out=self.input_ids.cpu[:total_num_scheduled_tokens],
    )
```

### 关键优化：overlap CPU 数据准备与 GPU 传输

`commit_block_table` 立刻触发 H2D 内存拷贝（block_table 从 CPU → GPU），之后的 CPU 计算（positions、token_ids 准备）与这次拷贝**重叠执行**，利用 CPU 和 PCIe 可以并行工作的特性。

### 为什么用 NumPy 而非 PyTorch？

`positions` 和 `num_scheduled_tokens` 等辅助数组用 NumPy 在 CPU 侧计算：
1. NumPy 对整型数组的标量操作（`np.add`, `np.repeat`）通常比 PyTorch 快（无 GPU 开销）
2. `.np` 是 zero-copy 视图，直接操作 pinned memory，无需额外分配

---

## 五、CUDA Graph：消除 CPU 调度开销

### 问题背景

每次 GPU 内核调用都需要 CPU "派遣"：CPU 发指令 → GPU 执行。对于 LLM 的 decode 阶段（batch size 小，每个内核时间短），CPU 调度开销（几十到几百 μs）可能占总延迟的 30%+。

### CUDA Graph 解决方案

```
录制阶段（启动时）：
  CUDA Graph 录制 → 捕获所有内核调用序列 → 存成 Graph

执行阶段（每步）：
  graph.replay() → GPU 直接执行整个序列 → 无 CPU 介入
```

### vLLM 的实现

```python
# 初始化时为多个 batch size 预录制 CUDA Graph
self.cudagraph_batch_sizes = list(reversed(self.compilation_config.cudagraph_capture_sizes))
# 例如：[1, 2, 4, 8, 16, 32, 64, 128, 256]

# 执行时由 CudagraphDispatcher 决定使用哪个 Graph
cudagraph_runtime_mode, batch_descriptor = (
    self.cudagraph_dispatcher.dispatch(batch_descriptor, use_cascade_attn)
)
```

**关键约束**：CUDA Graph 只适用于 decode 阶段（batch 内每个序列只生成1个 token），prefill 阶段因为 token 数量可变，通常用 eager 模式。

### 持久缓冲区与 CUDA Graph 的配合

```python
# 每步只更新缓冲区内容，不改变指针
self.input_ids.cpu[:n] = new_token_ids      # 写 CPU 端
self.input_ids.cpu_to_gpu()                  # 固定地址的 H2D 拷贝
# CUDA Graph 引用的始终是同一个 GPU 地址
```

---

## 六、异步输出拷贝：GPU→CPU 不阻塞

```python
class AsyncGPUModelRunnerOutput:
    def __init__(self, model_runner_output, sampled_token_ids, ...):
        # 在独立 stream 上发起非阻塞拷贝
        with torch.cuda.stream(async_output_copy_stream):
            async_output_copy_stream.wait_stream(default_stream)
            self.sampled_token_ids_cpu = sampled_token_ids.to("cpu", non_blocking=True)
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        # 调用时才同步，等待拷贝完成
        self.async_copy_ready_event.synchronize()
        ...
```

### 时间轴

```
主 stream:  [forward step N] → [forward step N+1] → ...
copy stream:                ↗ [D2H copy step N]  ↗ [D2H copy step N+1]
```

GPU-CPU 拷贝（sampled token ids，通常很小）与下一步的 GPU 前向传播重叠执行。在高吞吐场景下，能节省 ~5% 的端到端延迟。

---

## 七、推测解码集成

```python
# 初始化时根据配置创建不同的 drafter
if self.speculative_config.method == "ngram":
    self.drafter = NgramProposer(...)
elif self.speculative_config.use_eagle():
    self.drafter = EagleProposer(...)
    if method == "eagle3":
        self.use_aux_hidden_state_outputs = True  # 需要中间层输出
elif self.speculative_config.method == "medusa":
    self.drafter = MedusaProposer(...)

self.rejection_sampler = RejectionSampler()
```

推测解码的核心流程：
1. **drafter 提议**：生成 K 个候选 token（draft tokens）
2. **target 验证**：用大模型一次性验证所有候选 token（等同于 K+1 个 token 的 prefill）
3. **rejection sampling**：接受/拒绝候选 token，更新序列

EAGLE3 需要 `aux_hidden_states`（大模型特定层的隐藏状态），在 `LlamaModel.forward()` 中收集（见 llama.py 分析）。

---

## 八、状态管理：InputBatch 与请求跟踪

```python
# 核心状态：以 cpu tensor 形式缓存所有运行中请求的状态
self.input_batch = InputBatch(
    max_num_reqs=self.max_num_reqs,
    max_model_len=self.max_model_len,
    ...
)

# 每步更新请求状态
def _update_states(self, scheduler_output):
    # 处理新到达的请求
    for req_id in scheduler_output.scheduled_new_reqs:
        self.requests[req_id] = CachedRequestState(...)
        self.input_batch.add_request(...)

    # 处理完成/被抢占的请求
    for req_id in scheduler_output.finished_req_ids:
        del self.requests[req_id]
        self.input_batch.remove_request(...)
```

`InputBatch` 将所有请求的 token ids、positions、sampling params 等以**预分配的大张量切片**的形式存储，避免每步重建张量带来的内存分配开销。

---

## 九、set_forward_context：解耦 metadata 传递

```python
with set_forward_context(
    attn_metadata,        # 注意力 metadata（block table、seq_lens 等）
    self.vllm_config,
    num_tokens=num_input_tokens,
    cudagraph_runtime_mode=cudagraph_runtime_mode,
    ...
):
    model_output = self._model_forward(input_ids, positions, ...)
```

`set_forward_context` 通过 thread-local 存储将 `attn_metadata` 传递给模型内部的 `Attention` 层，而不需要通过函数参数逐层传递。

**好处**：
1. 模型的 `forward()` 签名干净（只有 `input_ids`, `positions`）
2. `@support_torch_compile` 时，torch.compile 看到的函数签名简单，更容易优化
3. attention metadata 复杂，逐层传递会污染每个层的接口

---

## 十、关键设计模式总结

| 技术 | 解决的问题 | 收益 |
|------|-----------|------|
| 预分配持久缓冲区 | CUDA Graph 需要固定地址 | 支持 CUDA Graph，减少内存分配 |
| CpuGpuBuffer + pinned memory | 高效 H2D 传输 | PCIe 传输加速 2~3x |
| NumPy for CPU 计算 | 快速构建辅助张量 | 低延迟 CPU 侧准备 |
| overlap：block_table 拷贝 + CPU 计算 | 隐藏 PCIe 传输延迟 | 减少等待时间 |
| CUDA Graph dispatch | CPU 调度开销太高 | decode 延迟降低 ~30% |
| 异步 D2H 拷贝 | GPU 结果传 CPU 阻塞主流 | 节省 ~5% 端到端延迟 |
| thread-local forward context | attention metadata 传递 | 干净的模型接口，便于编译 |
| InputBatch 集中状态管理 | 避免每步重建张量 | 减少内存分配和 GC 压力 |

---

*文件：`vllm/v1/worker/gpu_model_runner.py`*
*更新：2026-03*
