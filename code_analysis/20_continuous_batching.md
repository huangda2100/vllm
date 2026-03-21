# vLLM 的 Continuous Batching

> **核心问题**：推理服务同时收到 100 个请求，每个请求的 prompt 长度不同、生成长度不同，
> 如何让 GPU 始终满负荷工作，而不是等某个请求完成才能接下一个？
>
> **官方论文**：[Orca: A Distributed Serving System（Yu et al., OSDI 2022）](https://www.usenix.org/conference/osdi22/presentation/yu)
> **代码路径**：`vllm/v1/worker/gpu_model_runner.py`、`vllm/v1/core/sched/scheduler.py`

---

## 一、Static Batching 的问题

### 1.1 传统推理的工作方式

传统推理框架（非 vLLM）把所有请求分成固定的批次，每批完整处理后再接下一批：

```
Static Batching：

Batch 1（3 个请求同时进入）：
  ┌─── Prefill Phase ───┐   ┌──────────── Decode Phase ────────────┐
  │ Req A: prompt 100 tok│   │ A→1 A→2 A→3 A→4 A→5 A→6 A→done      │
  │ Req B: prompt  50 tok│   │ B→1 B→2 B→3 B→4 B→5 B→6 ... B→done  │
  │ Req C: prompt 200 tok│   │ C→1 C→2 ... (生成很长) ... C→done    │
  └─────────────────────┘   └──────────────────────────────────────┘

  问题 1：A 和 B 已经完成，但 C 还在生成
          → A、B 的 GPU 槽位被占用，新请求 D、E 无法进入
          → GPU 利用率随时间下降（C 越来越慢，batch 越来越空）

  问题 2：Prefill 阶段必须等最长的 prompt（C 的 200 tokens）
          → A 等着 C 算完才能开始 Decode

  问题 3：必须预先知道最大生成长度
          → 按最大值预留内存，大量浪费

Batch 2 只能在 Batch 1 全部完成后才开始
  → 新来的请求 D 在等待队列中一直等，TTFT（首 Token 延迟）很长
```

### 1.2 GPU 利用率随时间下降

```
GPU 利用率：

100% ████████████████████████████████
     ████████████████████████████████
 75% ████████████████████████████████
     ██████████████████████████
 50% ████████████████████████████████
     ██████████████████████
 25% ████████████████████████████████
     ██████████
  0% ─────────────────────────────────────────→ 时间
     Batch 开始       几个请求先完成   最后一个完成

  Batch 开始时 GPU 利用率 100%（所有请求在 decode）
  每当有请求完成，GPU 有效利用率下降
  最后只剩 1 个请求时，GPU 只有 1/N 的利用率
```

---

## 二、Continuous Batching 的核心思想

### 2.1 一句话描述

**不等批次凑满，也不等请求完成。每步（step）动态决定当前处理哪些请求的哪些 token，处理完就接新的。**

```
Continuous Batching（每步一次 GPU forward）：

Step 1: [Req A, prefill 100 tok]   [Req B, prefill 50 tok]
        ↑ 新请求                    ↑ 新请求

Step 2: [Req A, decode tok 1]      [Req B, decode tok 1]    [Req C, prefill 200 tok] ← C 实时插入！
        ↑ A 继续                   ↑ B 继续

Step 3: [Req A, decode tok 2]      [Req B, decode tok 2]    [Req C, decode tok 1]

Step 4: [Req A, decode tok 3]      [Req B, done → 移出！]   [Req C, decode tok 2]    [Req D, prefill] ← D 插入！

Step 5: [Req A, decode tok 4]      [Req C, decode tok 3]    [Req D, decode tok 1]

...
```

**关键特性**：
- 每步 GPU forward 可以同时处理**不同阶段**的请求（prefill + decode 混合）
- 请求完成后**立即移除**，空出位置给等待队列中的新请求
- 新请求**无需等待整批结束**即可加入

### 2.2 与 Static Batching 的对比

```
                     Static Batching    Continuous Batching
                     ───────────────    ───────────────────
GPU 利用率            随时间下降          接近恒定高位
新请求等待时间         等整批完成          等下一个 step（<100ms）
Prefill/Decode 混合  不支持              完全支持
请求数量变化          固定（每批开始时定）  动态变化（每步都可变）
内存分配              按最大长度预留       按需分配（PagedAttention）
```

---

## 三、每一步到底发生了什么

### 3.1 主循环（EngineCore.step）

```python
# vllm/v1/engine/core.py

def step(self):
    # ① 调度（CPU，< 1ms）
    scheduler_output = self.scheduler.schedule()
    #   → 决定：哪些请求本步处理，每个处理多少 token

    # ② 执行（GPU，几十到几百 ms）
    model_output = self.model_executor.execute_model(scheduler_output)
    #   → 把所有请求的 token 打包成一个张量，一次 forward

    # ③ 更新（CPU，< 1ms）
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )
    #   → 把生成的 token 写回各请求，检查是否完成，释放资源

    return engine_core_outputs
```

**时间分布**（70B 模型，batch=32，A100）：
```
schedule()          ≈ 0.5 ms    ← CPU 逻辑
execute_model()     ≈ 50 ms     ← GPU 计算（主要）
update_from_output()≈ 0.5 ms    ← CPU 逻辑
```

### 3.2 一个具体的三步示例

```
初始状态：batch 为空，等待队列有 3 个请求 [A, B, C]

─── Step 1 ─────────────────────────────────────────────────────
调度器：
  budget = 8192
  从 waiting 取出 A（prefill 10 tok）、B（prefill 8 tok）
  → scheduled_new_reqs = [A, B]
  → num_scheduled_tokens = {A: 10, B: 8}
  → budget = 8192 - 18 = 8174

Worker 构建 batch：
  input_ids   = [A_tok0, A_tok1, ..., A_tok9, B_tok0, ..., B_tok7]
                  ←── 10 个 ──→                ←── 8 个 ──→
  positions   = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7]
  形状：[18]（flat，一维）

GPU forward → 输出 logits [18, vocab_size]

采样：取每个请求的最后一个 logit
  A 的采样位置 = 9（第 10 个）→ 生成 token_A1
  B 的采样位置 = 17（第 18 个）→ 生成 token_B1

更新：
  A.num_computed_tokens = 10，A.output = [token_A1]
  B.num_computed_tokens = 8，  B.output = [token_B1]

─── Step 2 ─────────────────────────────────────────────────────
调度器：
  running = [A, B]，waiting = [C]
  A 本步 token 数 = 1（decode），B = 1（decode）
  budget = 8192 - 2 = 8190
  从 waiting 取出 C（prefill 5 tok）
  → scheduled_cached_reqs = [A, B]
  → scheduled_new_reqs = [C]
  → num_scheduled_tokens = {A: 1, B: 1, C: 5}

Worker 构建 batch：
  input_ids = [token_A1,   token_B1,   C_tok0, C_tok1, C_tok2, C_tok3, C_tok4]
               ← decode →  ← decode →  ←──────── prefill 5 个 ─────────────→
  positions = [10,          8,          0, 1, 2, 3, 4]
               ↑ A 的下一位  ↑ B 的下一位  ↑ C 从 0 开始
  形状：[7]

GPU forward → 输出 logits [7, vocab_size]

采样：
  A 的采样位置 = 0（第 1 个）→ 生成 token_A2
  B 的采样位置 = 1（第 2 个）→ 生成 token_B2
  C 的采样位置 = 6（第 7 个）→ 生成 token_C1

─── Step 3 ─────────────────────────────────────────────────────
假设 B 在 Step 2 生成了 EOS token → B 完成！

调度器：
  free(B) → 释放 B 的 KV 块，batch 空出一个位置
  running = [A, C]
  → num_scheduled_tokens = {A: 1, C: 1}
  → finished_req_ids = {B}

Worker：
  从 batch 移除 B
  batch = [A, C]
  positions = [11, 5]（各自的当前位置）
  形状：[2]
```

---

## 四、Prefill + Decode 混合的实现：张量构建细节

这是 Continuous Batching 最核心的技术问题：一次 forward 里同时有 prefill 请求（几百个 token）和 decode 请求（每个 1 个 token），GPU 如何处理？

### 4.1 Input IDs：一维展平张量

```python
# vllm/v1/worker/gpu_model_runner.py

# 每个请求在 batch 中的 token 是连续排列的
# 形状：[total_num_scheduled_tokens]（一维！）

示例（3 个请求，num_scheduled_tokens = {A:10, B:1, C:5}）：

input_ids = [
    A_tok0, A_tok1, ..., A_tok9,   # A 的 10 个 prefill token
    B_tok150,                       # B 的 1 个 decode token（第 151 个 token）
    C_tok0, C_tok1, ..., C_tok4,   # C 的 5 个 prefill token
]
形状：[16]

# 怎么构建？
# input_batch.token_ids_cpu_tensor 形状 = [max_num_reqs, max_model_len]
# 每行存一个请求的所有历史 token（包括 prompt + 已生成）

# 索引计算：
token_indices = req_indices * max_model_len + positions
# req_indices = [0,0,...,0, 1, 2,2,...,2]（每个位置对应哪个请求）
# positions   = [0,1,...,9, 150, 0,1,...,4]（在该请求历史中的位置）
# token_indices 唯一确定每个 token 的存储位置

input_ids = token_ids_cpu_tensor.flatten()[token_indices]
```

### 4.2 Positions：从 num_computed_tokens 直接推导

```python
# positions = num_computed_tokens[req_idx] + local_offset

示例：
  A: num_computed_tokens=0,   schedule 10 tok → positions = [0,1,2,...,9]
  B: num_computed_tokens=150, schedule 1 tok  → positions = [150]
  C: num_computed_tokens=0,   schedule 5 tok  → positions = [0,1,2,3,4]

最终 positions 张量（一维展平）：
  [0,1,2,3,4,5,6,7,8,9, 150, 0,1,2,3,4]
  ←────── A ──────────→  ←B→  ←── C ──→

关键洞察：positions 的语义完全统一
  prefill 的 position 从 0 开始（第一次见到这些 token）
  decode  的 position 从当前序列长度开始
  → 模型内部无需区分 prefill/decode！
    Transformer 只看 position，不关心是第几步调度的
```

### 4.3 Attention Metadata：让注意力知道边界

这是最复杂的部分。注意力计算时，每个请求的 query 范围不同、历史 KV 长度不同：

```
query_start_loc（累计和）：

  batch 中：A 有 10 个 query，B 有 1 个 query，C 有 5 个 query

  query_start_loc = [0, 10, 11, 16]
                     ↑  ↑   ↑   ↑
                     │  │   │   └─ 总共 16 个 query（batch 末尾）
                     │  │   └───── B 的 query 从位置 10 开始（A 结束后）
                     │  └───────── A 的 query 从位置 10 结束
                     └──────────── batch 从 0 开始

  → Flash Attention 用这个数组来"切割"query 张量
    A 的 query = input_ids[0:10]
    B 的 query = input_ids[10:11]
    C 的 query = input_ids[11:16]

seq_lens（每个请求的完整序列长度）：

  A: 0（已计算）+ 10（新） = 10   ← 纯 prefill，没有历史 KV
  B: 150（已计算）+ 1（新） = 151  ← decode，有 150 个历史 KV
  C: 0（已计算）+ 5（新）  = 5    ← 纯 prefill

  seq_lens = [10, 151, 5]

  → Flash Attention 用 seq_lens 决定每个请求要 attend 多少历史 token
    A: attend 全部 10 个（无缓存，纯自注意力）
    B: attend 当前 query + 150 个历史 KV（从 KV Cache 读取）
    C: attend 全部 5 个

num_computed_tokens（已缓存的 KV 数量）：

  [0, 150, 0]

  → 告诉 kernel：这些 token 的 KV 不需要重新计算，从 KV Cache 读取
```

完整的 attention metadata 结构：

```python
@dataclass
class CommonAttentionMetadata:
    query_start_loc:         torch.Tensor   # [batch_size + 1]，累计和
    seq_lens:                torch.Tensor   # [batch_size]，完整序列长度
    num_computed_tokens_cpu: torch.Tensor   # [batch_size]，已缓存的 KV 数
    slot_mapping:            torch.Tensor   # [total_tokens]，物理 KV 槽位
    block_table_tensor:      torch.Tensor   # [batch_size, max_blocks]，块表
    num_reqs:                int
    num_actual_tokens:       int            # 本步实际 token 数
    max_query_len:           int            # max(num_scheduled_tokens)
    max_seq_len:             int            # max(seq_lens)
```

### 4.4 Slot Mapping：KV Cache 的写入地址

每个新 token 需要知道"把我的 K、V 写到 GPU 显存的哪个位置"：

```
slot_mapping：一维数组，长度 = total_num_scheduled_tokens
每个位置存一个物理槽位号（block_id × block_size + offset）

示例（block_size=16）：

  A 的 prefill tokens（positions 0-9）：
    → 分配了 Block 0
    → slot_mapping[0:10] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
       （Block 0 的槽位 0-9）

  B 的 decode token（position 150）：
    → Block 9（B 的第 10 个块），槽位 = 150 % 16 = 6
    → slot_mapping[10] = 9 * 16 + 6 = 150

  C 的 prefill tokens（positions 0-4）：
    → 分配了 Block 15
    → slot_mapping[11:16] = [240, 241, 242, 243, 244]
       （Block 15 的槽位 0-4）

最终：
  slot_mapping = [0,1,2,3,4,5,6,7,8,9, 150, 240,241,242,243,244]
                  ←── A ──────────────→  ←B→  ←─────── C ─────────→

CUDA Kernel 用这个数组直接散射写入（scatter write）KV Cache：
  for i in range(total_tokens):
      kv_cache[slot_mapping[i]] = new_kv[i]
```

### 4.5 采样：只采样每个请求的最后一个 token

```python
# 采样位置 = query_start_loc[1:] - 1
# （每个请求 query 范围的最后一个位置）

query_start_loc = [0, 10, 11, 16]
logits_indices = query_start_loc[1:] - 1 = [9, 10, 15]

logits 形状：[16, vocab_size]
采样：
  A → logits[9]   （A 的第 10 个 token 的 logit）
  B → logits[10]  （B 的唯一 token 的 logit）
  C → logits[15]  （C 的第 5 个 token 的 logit）

→ 生成 3 个新 token，每个请求各 1 个（非 speculative 情况下）
```

---

## 五、Persistent Batch：跨步的请求状态管理

### 5.1 为什么需要 Persistent Batch

每步重新从头构建 batch 代价太高。vLLM 维护一个**持久化的 batch 容器**，在步间只做增量更新：

```python
# vllm/v1/worker/gpu_model_runner.py

class InputBatch:
    # 跨步持久的张量（预分配，避免重复分配）
    token_ids_cpu_tensor: torch.Tensor  # [max_num_reqs, max_model_len]
    num_computed_tokens_cpu: np.ndarray # [max_num_reqs]
    block_table: MultiGroupBlockTable   # [max_num_reqs, max_blocks_per_req]

    # 请求索引管理
    req_id_to_index: dict[str, int]     # req_id → 在 batch 中的行号
    req_ids: list[str | None]           # 每行对应哪个请求
```

### 5.2 增量更新流程

```python
# vllm/v1/worker/gpu_model_runner.py _update_states()

def _update_states(scheduler_output):

    # ① 移除已完成的请求
    for req_id in scheduler_output.finished_req_ids:
        index = self.input_batch.req_id_to_index[req_id]
        self.input_batch.req_ids[index] = None  # 标记为空槽
        del self.input_batch.req_id_to_index[req_id]

    # ② 移除本步没被调度的请求（如被 preempt 的）
    for req_id in (input_batch.req_ids - scheduled_req_ids):
        input_batch.remove_request(req_id)

    # ③ 添加新请求（第一次被调度）
    for new_req_data in scheduler_output.scheduled_new_reqs:
        req_state = CachedRequestState(
            req_id=new_req_data.req_id,
            prompt_token_ids=new_req_data.prompt_token_ids,
            block_ids=new_req_data.block_ids,
            num_computed_tokens=new_req_data.num_computed_tokens,
        )
        self.requests[req_id] = req_state
        self.input_batch.add_request(req_state)
        # → token_ids 写入 token_ids_cpu_tensor 的新行
        # → block_ids 写入 block_table 的新行

    # ④ 更新已有请求（推进状态）
    for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
        req_state = self.requests[req_id]
        req_state.num_computed_tokens = new_num_computed_tokens
        req_state.output_token_ids.extend(new_token_ids)
        # 把新生成的 token 追加到 token_ids_cpu_tensor 对应行

    # ⑤ Condense：消除 batch 中的空洞（被删除留下的 None 槽）
    input_batch.condense()
    # → 把后面的请求移到前面，使 batch 连续
```

**Condense 示意图**：

```
移除前：
  batch[0] = Req A（运行中）
  batch[1] = None（Req B 完成，留下空洞）
  batch[2] = Req C（运行中）
  batch[3] = None（Req D 被抢占，留下空洞）
  batch[4] = Req E（运行中）

condense() 后：
  batch[0] = Req A
  batch[1] = Req C
  batch[2] = Req E
  batch[3] = None（空）
  batch[4] = None（空）

→ 3 个请求紧密排列，无空洞
→ 后续构建张量时只需处理 [0:3]
```

### 5.3 CPU-GPU 数据传输最小化

```
第一次调度（NewRequestData）：
  需要传：完整 prompt token ids
  大小：4096 tokens × 4 bytes = 16 KB（每请求）
  → 发送一次后缓存在 token_ids_cpu_tensor

后续调度（CachedRequestData）：
  需要传：新生成的 token ids（1-2 个）+ 新分配的 block ids
  大小：< 100 bytes（每请求）
  → 极小的增量更新

每步 GPU 传输量（32 个 decode 请求）：
  32 × 100 bytes = 3.2 KB（接近零）
```

---

## 六、Iteration-Level 调度 vs Request-Level 调度

Continuous Batching 的根本思路来自 Orca 论文对"调度粒度"的重新定义：

```
Request-Level 调度（传统）：
  调度单元 = 一个完整请求（prefill + 全部 decode）
  → 一旦加入 batch，独占资源直到完成
  → batch 固定，新请求必须等

Iteration-Level 调度（Orca / vLLM）：
  调度单元 = 一个 iteration（一步 forward）
  → 每步重新决定谁参与
  → 完成的请求立即释放，空位立即被新请求填充
  → GPU 利用率趋近于 100%
```

```
吞吐量对比（论文数据）：

                    P50 延迟   P99 延迟   吞吐量
Static Batching      100ms      800ms     1×
Continuous Batching   80ms      200ms     2-4×
```

---

## 七、Chunked Prefill：解决 Prefill 独占 GPU 的问题

### 7.1 问题：长 Prefill 阻塞 Decode

```
没有 Chunked Prefill：

  Step N:   [Req X, prefill 4096 tokens]               ← 耗时 200ms
             [Req Y decode 1 tok] 被迫等待！

  用户 Y 的 TTFT（首 Token 延迟）= 200ms

有 Chunked Prefill（chunk_size=256）：

  Step N:   [Req X, prefill chunk 0-255]   [Req Y, decode 1 tok]
  Step N+1: [Req X, prefill chunk 256-511] [Req Y, decode 1 tok]
  ...（共 16 步完成 X 的 prefill）
  Step N+15:[Req X, prefill chunk 3840-4095][Req Y, decode 1 tok]
  Step N+16:[Req X, decode 1 tok]           [Req Y, decode 1 tok]

  用户 Y 在 Step N 就拿到了第一个 token！TTFT ≈ 5ms
```

### 7.2 代码实现

```python
# vllm/v1/core/sched/scheduler.py

# 对 RUNNING 请求（长 prefill 分块）
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold  # 截断到 chunk 大小

# 对 WAITING 请求（进入时就限制）
chunked_prefill_limit = min(token_budget, long_prefill_token_threshold)
if num_new_tokens > chunked_prefill_limit:
    num_new_tokens = chunked_prefill_limit

# 结果：一个 prefill 请求可能要跨多个 step 才完成
# req.num_computed_tokens 记录已处理进度
```

### 7.3 Chunked Prefill 的权衡

```
chunk_size 越小：
  + decode 请求延迟越低（更少被 prefill 阻塞）
  - prefill 请求完成时间越长（分成更多步）
  - 每步 GPU 利用率可能下降（prefill 的 token 少，GEMM 效率低）

chunk_size 越大：
  + prefill 完成更快（TTFT 低）
  - decode 请求可能被长时间阻塞

典型配置：
  chunk_size = 512 ~ 2048（根据模型和硬件调整）
  token_budget = 4096 ~ 16384（总预算，prefill + decode 共享）
```

---

## 八、Preemption（抢占）：KV Cache 内存保障

### 8.1 为什么需要抢占

```
场景：
  running = [A(1000 tok KV), B(500 tok KV), C(200 tok KV)]
  waiting = [D(2000 tok prefill)]

  空闲 KV 块 = 30 块（30 × 16 = 480 个 token 槽）
  D 需要 125 块（2000/16）

  → D 无法分配！但现有请求也需要继续 decode
  → 必须从 running 中腾出资源
```

### 8.2 抢占策略

```python
# FCFS 策略：抢占最晚加入的请求
preempted = self.running.pop()  # running 列表末尾

# Priority 策略：抢占优先级最低的请求
preempted = max(
    self.running,
    key=lambda r: (r.priority, r.arrival_time)
)
```

### 8.3 被抢占请求的命运

```
被抢占：
  kv_cache_manager.free(req)     → KV 块放回 LRU 队列（保留哈希！）
  req.num_computed_tokens = 0    → 重置（下次从头开始）
  waiting.prepend(req)           → 插入等待队列头部

下次被调度：
  kv_cache_manager.get_computed_blocks(req)
  → 如果 KV 块未被驱逐（哈希命中）→ 直接复用！无需重新计算
  → 如果 KV 块已被驱逐 → 重新 prefill（真正的代价）

前缀缓存的价值：
  抢占的请求如果很快被重新调度（KV 块未被其他请求覆盖）
  → 几乎无代价的恢复（O(1) 命中检查）
  → 抢占不可怕
```

---

## 九、完整数据流：一步 Forward 的全过程

```
EngineCore.step()
      │
      ▼
─── 调度阶段（CPU）────────────────────────────────────────────────────
scheduler.schedule()
  │
  ├─ 遍历 running（已运行请求）
  │    每个请求：分配本步 KV 块，决定处理多少 token
  │    KV 不足时：抢占最低优先级的请求
  │
  ├─ 遍历 waiting（等待请求）
  │    新请求：查前缀缓存，分配 KV 块，转为 RUNNING
  │
  └─ 构建 SchedulerOutput{
         scheduled_new_reqs:    [新请求完整数据]
         scheduled_cached_reqs: [旧请求增量数据]
         num_scheduled_tokens:  {req_id: token 数}
         finished_req_ids:      {完成的请求 id}
     }

─── 执行阶段（GPU）────────────────────────────────────────────────────
executor.execute_model(scheduler_output)
  │
  ├─ Worker._update_states(scheduler_output)         # CPU
  │    移除完成请求，添加新请求，更新 token ids
  │    condense() 消除 batch 空洞
  │
  ├─ Worker._prepare_inputs(scheduler_output)        # CPU
  │    构建 input_ids：[total_tokens]         一维展平
  │    构建 positions：[total_tokens]         各请求的绝对位置
  │    构建 query_start_loc：[batch+1]        各请求 query 范围
  │    构建 seq_lens：[batch]                 完整序列长度
  │    构建 slot_mapping：[total_tokens]      KV Cache 写入地址
  │    构建 block_table：[batch, max_blocks]  KV Cache 读取地址
  │    → CPU 张量 copy_to_device() → GPU
  │
  ├─ 模型 forward                                    # GPU
  │    embedding_lookup(input_ids)         → [total_tokens, hidden]
  │    for layer in layers:
  │        RMSNorm, QKV proj
  │        Flash Attention（读 KV Cache 历史，写 KV Cache 新值）
  │        O proj, FFN
  │        + 残差
  │    final_norm → lm_head → logits[total_tokens, vocab_size]
  │
  └─ 采样
       logits_indices = query_start_loc[1:] - 1   # 每个请求最后一个位置
       sampled_tokens = sampler(logits[logits_indices])
       → [batch_size] 个新 token id

─── 更新阶段（CPU）───────────────────────────────────────────────────���
scheduler.update_from_output(scheduler_output, model_output)
  │
  ├─ 对每个请求：
  │    追加新生成的 token
  │    检查停止条件（EOS / max_tokens / stop_strings）
  │
  ├─ 完成的请求：
  │    kv_cache_manager.free(req)     → KV 块归还 LRU
  │    从 running 移除
  │    生成 EngineCoreOutput（含所有 token）
  │
  └─ 返回 engine_core_outputs → 发送给用户
```

---

## 十、Flash Attention 与 PagedAttention 的配合

Continuous Batching 依赖两个底层机制配合才能高效运行：

### 10.1 Flash Attention：统一处理 Prefill 和 Decode

```
传统实现（两个独立 kernel）：
  prefill_attn(Q, K, V)    → 处理 prefill 请求
  paged_decoding_attn(Q, KV_Cache, block_table) → 处理 decode 请求
  → 需要判断请求类型，两套代码路径

vLLM 的实现（Flash Attention + PagedAttention）：
  flash_attn_with_kv_cache(
      Q,           # [total_tokens, heads, head_dim]
      K_new,       # [total_tokens, kv_heads, head_dim]
      V_new,       # [total_tokens, kv_heads, head_dim]
      KV_cache,    # [num_blocks, block_size, 2, kv_heads, head_dim]
      block_table, # [batch, max_blocks]
      seq_lens,    # [batch]（完整序列长度）
      query_start_loc,  # [batch+1]（query 切割点）
  )

  → kernel 内部自动区分：
    seq_lens[i] == num_scheduled_tokens[i]：纯 prefill，只看当前 query
    seq_lens[i] >  num_scheduled_tokens[i]：有历史 KV，读 KV Cache
```

### 10.2 PagedAttention：支持非连续 KV Cache

```
Block Table 的作用：

  batch 中的请求用不同的 KV 块，可以是不连续的物理内存：

  Req A（prefill）：blocks = [Block 0, Block 1]（刚分配，从头开始）
  Req B（decode）：blocks = [Block 5, Block 8, Block 12]（已存在的历史块）
  Req C（prefill）：blocks = [Block 3]（刚分配）

  Flash Attention kernel 通过 block_table 把逻辑 token 位置映射到物理块：
    block_table[B] = [5, 8, 12]
    B 的第 32 个 token → block_table[B][32/16] = Block 8 → slot 32%16 = 0

  → prefill 和 decode 的 KV 存在完全不同的物理块
  → 不需要连续显存，显存利用率接近 100%
```

---

## 十一、关键参数与调优

```python
# 影响 Continuous Batching 行为的关键参数

EngineArgs(
    max_num_seqs=256,              # 最大并发请求数（batch size 上限）
    max_num_batched_tokens=8192,   # 单步最大 token 数（token budget）
    max_model_len=32768,           # 单请求最大序列长度

    # Chunked Prefill
    enable_chunked_prefill=True,   # 是否开启（v1 默认开启）
    max_num_batched_tokens=2048,   # 等于 chunk_size 时，控制每步 prefill 量

    # 内存
    gpu_memory_utilization=0.9,    # 90% 显存用于 KV Cache
    block_size=16,                 # 每个 KV 块的 token 数

    # 推测解码（与 CB 兼容）
    speculative_model="draft_model",
    num_speculative_tokens=5,
)
```

**常见调优场景**：

| 目标 | 调整参数 | 效果 |
|------|---------|------|
| 降低延迟（TTFT） | 减小 `max_num_batched_tokens` | prefill 更快完成，但吞吐可能下降 |
| 提升吞吐 | 增大 `max_num_seqs` | 更多并发请求，GPU 利用率更高 |
| 长上下文场景 | 增大 `max_model_len` | 支持更长序列，KV 块数减少 |
| 减少 decode 延迟 | 增大 chunk 限制 | 允许更多 decode 请求进入 batch |

---

## 十二、总结

```
Continuous Batching 的本质是把调度粒度从"请求"变成"iteration"：

传统：  [    Req 1 完整处理    ] [    Req 2 完整处理    ] [    Req 3    ]
vLLM：  [Step1][Step2][Step3][Step4][Step5][Step6][Step7][Step8]...
         ↑      ↑      ↑      ↑ 每步动态选择参与的请求和 token 数
         Req1   Req1   Req2   Req1                        Req3
         (prefill) (decode) (加入) (decode+decode+prefill) ...

实现的关键技术：
  1. 展平的一维 token 张量：统一处理 prefill/decode，无需分支
  2. Position 由 num_computed_tokens 推导：自动区分序列位置
  3. query_start_loc 累计和：告诉 attention kernel 每个请求的边界
  4. Slot Mapping：精确控制每个 token 的 KV Cache 写入地址
  5. Persistent Batch：跨步维护状态，增量更新避免重复初始化
  6. PagedAttention：非连续 KV 块使请求可以随时加入/离开

效果：
  GPU 利用率：30-60% → 90%+（Static Batching vs Continuous）
  吞吐量：2-4× 提升
  P99 延迟：大幅降低（新请求不需要等待整批完成）
```

| 技术组件 | 解决的问题 | 关键文件 |
|---------|-----------|---------|
| Iteration-Level Scheduling | GPU 空闲等待 | `scheduler.py:schedule()` |
| 一维 Token 张量 | Prefill/Decode 统一 | `gpu_model_runner.py:_prepare_inputs()` |
| query_start_loc | Attention 边界感知 | `attention/backends/utils.py` |
| Slot Mapping | KV Cache 精确写入 | `block_table.py:compute_slot_mapping()` |
| Persistent InputBatch | 减少初始化开销 | `gpu_model_runner.py:_update_states()` |
| Chunked Prefill | 防长 prefill 阻塞 | `scheduler.py:long_prefill_threshold` |
| LRU Preemption | KV 内存溢出保障 | `scheduler.py:_preempt()` |

---

*参考资料：*
- *[Orca: A Distributed Serving System（Yu et al., OSDI 2022）](https://www.usenix.org/conference/osdi22/presentation/yu)*
- *[vLLM 原始论文：PagedAttention（Kwon et al. 2023）](https://arxiv.org/abs/2309.06180)*
- *[Sarathi-Serve：Chunked Prefill（Agrawal et al. 2024）](https://arxiv.org/abs/2403.02310)*
- *[vLLM 源码：gpu_model_runner.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py)*
- *[FlashAttention-2（Dao et al. 2023）](https://arxiv.org/abs/2307.08691)*
*更新：2026-03*
