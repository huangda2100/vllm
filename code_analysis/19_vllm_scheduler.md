# vLLM 调度器：Continuous Batching 的核心

> **核心问题**：推理服务同时收到大量请求，如何高效决定每一步处理哪些请求、
> 处理多少 token，同时保证 GPU 显存不溢出、延迟不失控？
>
> **代码路径**：`vllm/v1/core/sched/scheduler.py`（1300+ 行）

---

## 一、调度器在系统中的位置

```
用户请求
  │
  ▼
vllm/entrypoints/         ← HTTP Server / 同步接口
  │
  ▼
vllm/v1/engine/core.py    ← EngineCore（主循环）
  │
  ├──► scheduler.schedule()   ← ★ 调度器：决定本步处理什么
  │
  ├──► executor.execute_model(scheduler_output)  ← GPU 执行
  │
  └──► scheduler.update_from_output()            ← 更新请求状态
         │
         ▼
     EngineCoreOutput → 返回给用户
```

调度器是 CPU 侧的"大脑"，每一步（step）都要回答：
1. 哪些请求参与本次计算？
2. 每个请求处理多少 token？
3. 如何在 GPU 显存不够时处理冲突？

---

## 二、请求状态机

```python
# vllm/v1/request.py
class RequestStatus(enum.IntEnum):
    WAITING               # 在等待队列，尚未被调度
    WAITING_FOR_FSM       # 等待结构化输出的 FSM 编译完成
    WAITING_FOR_REMOTE_KVS  # 等待远程 KV 缓存传输（P/D 分离）
    RUNNING               # 本步被调度，正在执行
    PREEMPTED             # 被抢占，资源被释放，退回等待队列

    # 终止状态（以下均 > PREEMPTED）
    FINISHED_STOPPED        # 生成了 EOS token，正常停止
    FINISHED_LENGTH_CAPPED  # 达到 max_tokens 上限
    FINISHED_ABORTED        # 用户主动取消
    FINISHED_IGNORED        # Prompt 超过 max_model_len，被忽略
```

**状态转移图**：

```
                ┌──────────────────────────────────────────┐
                │                                          │
                ▼                 KV 缓存不足               │
  add_request() → WAITING ──── schedule() ────► RUNNING ──┤ PREEMPTION
                    ▲                               │      └─►PREEMPTED
                    │                               │
                    │ prepend（头部插入）             │ update_from_output()
                    └─── PREEMPTED ◄────────────────┘
                                               │
                                               ▼
                                         FINISHED_*
                                    （释放所有 KV 块）
```

**关键细节**：
- 被抢占的请求 `num_computed_tokens = 0`（重置，需重新计算）
- 被抢占的请求插入等待队列**头部**（下步优先调度）
- 前缀缓存可能命中，减少重新计算的开销

---

## 三、Continuous Batching 与传统 Static Batching 的对比

```
传统 Static Batching（旧方式）：
  ┌─────────────────────────────────────────────────────────────┐
  │ Prefill Phase（一次处理）      │ Decode Phase（逐步生成）      │
  │ Req A: [prompt 100 tokens]   │ A: → → → → → → done (20 tok)│
  │ Req B: [prompt 200 tokens]   │ B: → → → → → → → → → → done │
  │ Req C: [prompt 50 tokens]    │ C: → → → → done (5 tok)      │
  │ Batch 固定，完成后才接新请求  │ C 完成后 GPU 资源浪费！        │
  └─────────────────────────────────────────────────────────────┘

Continuous Batching（vLLM 方式）：
  Step 1: [Req A prefill 100tok] [Req B prefill 100tok]
  Step 2: [Req A decode 1tok]    [Req B prefill 100tok] [Req C prefill 50tok] ← C 实时插入！
  Step 3: [Req A decode 1tok]    [Req B decode 1tok]    [Req C decode 1tok]
  Step 4: [Req A decode 1tok]    [Req B decode 1tok]    [Req C done → Req D 插入！]
  ...

优势：
  ① GPU 永远满负荷（不会等某个请求完成才接新的）
  ② 不同请求可以混合不同阶段（Prefill + Decode）
  ③ 吞吐量显著提升（2-4×）
```

---

## 四、核心数据结构

### 4.1 Scheduler 成员变量

```python
class Scheduler:
    # 请求管理
    requests:    dict[str, Request]   # req_id → Request，全局映射
    running:     list[Request]        # 当前 step 运行的请求（有序）
    waiting:     RequestQueue         # 等待队列（FCFS 或 Priority）

    # 资源管理
    kv_cache_manager:    KVCacheManager       # KV 块分配/释放/前缀缓存
    encoder_cache_manager: EncoderCacheManager  # 多模态编码器缓存

    # 配置限制
    max_num_running_reqs:     int  # 最大并行请求数（max_num_seqs）
    max_num_scheduled_tokens: int  # 单步最大 token 数（max_num_batched_tokens）

    # 状态跟踪
    finished_req_ids:     set[str]  # 上一步到本步之间完成的请求
    num_lookahead_tokens: int       # 推测解码预留 token 数
    policy: SchedulingPolicy        # FCFS 或 PRIORITY
```

### 4.2 SchedulerOutput（调度结果）

调度器输出，传给 GPU Worker 执行：

```python
@dataclass
class SchedulerOutput:
    # 本步新加入的请求（第一次调度，需完整数据）
    scheduled_new_reqs: list[NewRequestData]

    # 本步继续的旧请求（增量更新，只传差异）
    scheduled_cached_reqs: CachedRequestData

    # 每个请求本步处理的 token 数
    num_scheduled_tokens: dict[str, int]  # req_id → token 数
    total_num_scheduled_tokens: int

    # 已完成的请求（Worker 需要清理）
    finished_req_ids: set[str]

    # 通用前缀块数（Cascade Attention 优化用）
    num_common_prefix_blocks: list[int]

    # 推测解码
    scheduled_spec_decode_tokens: dict[str, list[int]]

    # 结构化输出
    grammar_bitmask: np.ndarray  # [num_reqs, vocab_size]

    # KV 传输（P/D 分离）
    kv_connector_metadata: KVConnectorMetadata | None
```

**NewRequestData 与 CachedRequestData 的区别**：

```
NewRequestData（第一次调度）：       CachedRequestData（后续调度）：
  req_id                              req_ids（列表）
  prompt_token_ids（完整 prompt）     new_token_ids（只有新生成的）
  sampling_params                     new_block_ids（只有新分配的块）
  block_ids（完整块列表）             num_computed_tokens（更新后的值）
  num_computed_tokens                 resumed_from_preemption（是否从抢占恢复）

Worker 缓存了 NewRequestData → 后续只需增量更新，减少 CPU→GPU 传输
```

---

## 五、schedule() 函数：三阶段调度算法

```python
def schedule(self) -> SchedulerOutput:
    """
    每个 step 调用一次，决定本 step 处理哪些请求、多少 token
    """
```

### 5.1 第一阶段：调度 RUNNING 请求

已经在运行的请求，优先保证它们继续执行：

```
for request in self.running:

  ① 计算本步需要处理的 token 数：
       num_new_tokens = (num_tokens + num_lookahead - num_computed_tokens)

  ② 应用 long_prefill 限制：
       if num_new_tokens > long_prefill_threshold:
           num_new_tokens = long_prefill_threshold   ← 分块，避免长 prefill 独占 GPU

  ③ 检查 token_budget：
       if token_budget <= 0: break                   ← 预算耗尽，停止调度

  ④ 尝试为本步新 token 分配 KV 缓存块：
       new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)

       if new_blocks is not None:
           scheduled_running_reqs.append(request)   ← 成功，加入本步
           token_budget -= num_new_tokens

       else:
           PREEMPTION！（见第五章）
```

### 5.2 第二阶段：调度 WAITING 请求

RUNNING 请求调度完毕后，再尝试从等待队列填充剩余算力：

```
if no preemption happened:    ← 有抢占时不调度新请求（避免进一步加剧资源压力）

  for request in waiting_queue:

    ① 检查并发上限：
         if len(running) >= max_num_running_reqs: break

    ② 检查 FSM 编译状态（结构化输出）：
         if request.status == WAITING_FOR_FSM: skip

    ③ 检查 KV 传输状态（P/D 分离）：
         if request.status == WAITING_FOR_REMOTE_KVS: skip

    ④ 前缀缓存查询：
         computed_blocks, num_cached_tokens = kv_cache_manager.get_computed_blocks(request)
         num_new_tokens = request.num_tokens - num_cached_tokens

    ⑤ Chunked Prefill：
         if num_new_tokens > chunked_prefill_limit:
             num_new_tokens = chunked_prefill_limit  ← 拆分 prefill

    ⑥ 分配 KV 缓存：
         new_blocks = kv_cache_manager.allocate_slots(request, num_new_tokens)

         if new_blocks is not None:
             request.status = RUNNING
             token_budget -= num_new_tokens
         else:
             break   ← 等待队列后面的请求也分配不了，停止
```

### 5.3 第三阶段：构建 SchedulerOutput

```
① 计算通用前缀块（所有运行请求共享的前缀）
② 构建 NewRequestData（第一次被调度的请求）
③ 构建 CachedRequestData（继续执行的请求，增量）
④ 收集结构化输出的 grammar bitmask
⑤ 构建 KV Connector 元数据（P/D 分离）
⑥ 调用 _update_after_schedule()：推进 num_computed_tokens
⑦ return SchedulerOutput
```

### 5.4 完整调度流程图

```
schedule() 被调用
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│ 第一阶段：处理 RUNNING 请求                                        │
│                                                                   │
│  for req in self.running:                                         │
│    计算 num_new_tokens（含 lookahead）                             │
│    应用 long_prefill_threshold 分块                               │
│    token_budget 够用？                                            │
│    ├─ 是 → allocate_slots() 成功？                                │
│    │        ├─ 是 → 加入 scheduled，budget -= tokens              │
│    │        └─ 否 → PREEMPTION（选抢占对象，释放 KV，回等待队列）    │
│    └─ 否 → break                                                  │
└────────────────────────────────────────────────────┬────────────┘
                                                     │
                                                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 第二阶段：处理 WAITING 请求（无抢占时）                             │
│                                                                   │
│  while budget > 0 and len(running) < max_running:                │
│    req = waiting_queue.pop()                                      │
│    检查 FSM / KV 传输状态                                         │
│    get_computed_blocks() → 前缀缓存命中数                          │
│    chunked prefill 限制                                           │
│    allocate_slots() 成功？                                        │
│    ├─ 是 → RUNNING，加入 scheduled                               │
│    └─ 否 → break                                                  │
└────────────────────────────────────────────────────┬────────────┘
                                                     │
                                                     ▼
                               构建 SchedulerOutput，返回给 Executor
```

---

## 六、Preemption（抢占）机制

### 6.1 为什么需要抢占？

```
场景：
  步 1：Req A（prefill 100 tokens），Req B（prefill 100 tokens）
        → 分别分配了 10 个 KV 块
  步 2：Req C 到来（prefill 500 tokens）
        → A、B 继续各需 1 个 decode 块
        → C 需要 40 个块
        → GPU 总空余块 = 25，只够 A+B 或 C 其中之一

  如果没有抢占：C 必须等待（但 A、B 的 decode 产生不了新请求）
              → GPU 利用率下降

  有抢占：选择优先级最低的请求驱逐，释放其 KV 块
```

### 6.2 抢占算法

```python
# FCFS 策略：抢占 running 列表中最后一个（最晚加入的）
preempted_req = self.running.pop()

# PRIORITY 策略：抢占优先级最低的（priority 值最大）
preempted_req = max(
    self.running,
    key=lambda r: (r.priority, r.arrival_time)
    #              ↑ 优先级值越大，实际优先级越低
    #                              ↑ 同优先级时，后到的先被抢占
)
```

### 6.3 抢占处理步骤

```python
# 1. 释放所有 KV 缓存块
kv_cache_manager.free(preempted_req)
encoder_cache_manager.free(preempted_req)

# 2. 更新状态
preempted_req.status = RequestStatus.PREEMPTED
preempted_req.num_computed_tokens = 0   # ← 重置！下次从头计算
preempted_req.num_preemptions += 1

# 3. 插入等待队列头部（下步优先重新调度）
self.waiting.prepend_request(preempted_req)
```

### 6.4 前缀缓存缓解抢占代价

```
Req A 被抢占（已处理 200 tokens，占 12 个 KV 块）：
  kv_cache_manager.free(req_a)
  → 12 个块放回 LRU 队列（但哈希保留！）

下步 Req A 被重新调度：
  kv_cache_manager.get_computed_blocks(req_a)
  → 如果 12 个块都没被驱逐 → 全部命中！
  → num_computed_tokens 恢复到 200，无需重新计算 Prefill
  → 只需 decode 阶段继续

最坏情况（KV 块被其他请求覆盖）：
  → 从 0 重新 prefill（真正的"代价"）
```

---

## 七、Token Budget 机制

调度器用 token budget 控制每步的计算量，防止单步计算时间过长：

```
max_num_scheduled_tokens = 8192   （假设配置）
max_num_running_reqs     = 256    （假设配置）

调度开始：token_budget = 8192

Step 调度各请求后 budget 变化：
  Req A（decode，1 tok）  → budget = 8191
  Req B（decode，1 tok）  → budget = 8190
  Req C（prefill，512 tok）→ budget = 7678
  Req D（prefill，512 tok）→ budget = 7166
  Req E（prefill，5000 tok，但 threshold=2048）→ budget = 5118
  ...
  budget ≤ 0 → 停止调度本步等待队列中剩余的请求
```

**关键限制参数**：

| 参数 | 来源 | 作用 |
|------|------|------|
| `max_num_batched_tokens` | SchedulerConfig | 单步最大 token 数 |
| `max_num_seqs` | SchedulerConfig | 最大并发请求数 |
| `long_prefill_token_threshold` | SchedulerConfig | 单个 prefill 最大 token 数（分块） |
| `max_model_len` | ModelConfig | 单请求最大序列长度 |

---

## 八、Chunked Prefill

### 8.1 问题：长 Prefill 阻塞 Decode

```
没有 Chunked Prefill：
  Step 1: [Req A, prefill 8000 tokens] → GPU 忙 200ms
          [Req B, decode 1 token]      → 等待 200ms 才能开始！

  用户感知延迟（TTFT）很长 ← Req B 被 Req A 阻塞

有 Chunked Prefill（threshold=256）：
  Step 1: [Req A, prefill chunk 0-255]  [Req B, decode 1 tok]
  Step 2: [Req A, prefill chunk 256-511] [Req B, decode 1 tok]
  ...
  Step 31: [Req A, prefill chunk 7936-7999] [Req B, decode 1 tok]
  Step 32: [Req A, decode 1 tok]            [Req B, decode 1 tok]

  优势：Req B 的延迟从 200ms → 5ms（第一步就开始 decode）
```

### 8.2 代码实现

```python
# vllm/v1/core/sched/scheduler.py

# 对 RUNNING 请求的长 prefill 分块
if 0 < long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = long_prefill_token_threshold

# 对 WAITING 请求的 chunked prefill
chunked_prefill_limit = min(
    token_budget,
    long_prefill_token_threshold or token_budget,
)
if num_new_tokens > chunked_prefill_limit:
    num_new_tokens = chunked_prefill_limit
```

**结果**：一个 prefill 请求可能需要多步才能完成，每步只处理 `threshold` 个 token。

---

## 九、优先级队列

### 9.1 FCFS（先来先服务）

```python
# 基于 deque 实现
class FIFORequestQueue(RequestQueue):
    def add_request(self, request):
        self._queue.append(request)           # 尾部插入

    def pop_request(self):
        return self._queue.popleft()          # 头部取出（先来先得）

    def prepend_request(self, request):
        self._queue.appendleft(request)       # 被抢占的请求插入头部（优先恢复）
```

### 9.2 Priority Queue（优先级队列）

```python
class PriorityRequestQueue(RequestQueue):
    def __init__(self):
        self._heap: list[tuple[int, float, Request]] = []
                         # ↑ priority  ↑ arrival_time

    def add_request(self, request):
        heapq.heappush(self._heap,
                       (request.priority, request.arrival_time, request))
        # Python heapq 是最小堆：priority=0 的请求先弹出（优先级最高）

    def pop_request(self):
        _, _, request = heapq.heappop(self._heap)
        return request
```

**优先级语义**：

```
priority = 0  → 最高优先级（最先被调度）
priority = 10 → 普通优先级
priority = 100 → 最低优先级（最后被调度，最先被抢占）

应用场景：
  交互式请求（用户正在等待）   → priority = 0
  批处理任务（后台处理）        → priority = 50
  低优先级任务（可被抢占）      → priority = 100
```

---

## 十、与 KVCacheManager 的接口

调度器通过 KVCacheManager 管理所有 KV 块：

```
调度器调用                       KVCacheManager 内部
─────────────────────────────────────────────────────────────────
get_computed_blocks(req)    →    查找前缀缓存（哈希匹配）
                                 返回：命中的块列表 + 命中 token 数

allocate_slots(req,          →   计算需要块数
               num_new_tok)      从 free_block_queue 分配
                                 返回：新分配的块，或 None（OOM）

free(req)                    →   ref_cnt--，归还到 LRU 队列
                                 （哈希保留，等待前缀复用）

cache_blocks(req, n)         →   对满块计算哈希，注册到缓存表
                                 （下次同前缀请求可命中）
```

**调度器对 KVCacheManager 的调用时序**：

```
┌─────────────────────────────────────────────────────┐
│ 新请求第一次调度                                       │
│   get_computed_blocks()  → 检查是否有可复���的前缀块    │
│   allocate_slots()       → 分配新块（含 lookahead）    │
│   → 成功：request.status = RUNNING                    │
│   → 失败：触发 PREEMPTION                             │
└───────────────────────────────────────────┬─────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────┐
│ 已运行请求（每步）                                     │
│   allocate_slots()       → 只分配本步新 token 的块    │
│   _update_after_schedule() → 推进 num_computed_tokens │
│   cache_blocks()           → ���存本步完成的满块        │
└───────────────────────────────────────────┬─────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────┐
│ 请求完成（或被抢占）                                   │
│   free(req)              → 释放所有块（LRU 保留哈希）  │
│   del requests[req_id]   → 清理元数据                 │
└─────────────────────────────────────────────────────┘
```

---

## 十一、update_from_output()：处理 GPU 输出

GPU 执行完毕后，调度器处理输出，推进请求状态：

```python
def update_from_output(
    self,
    scheduler_output: SchedulerOutput,
    model_runner_output: ModelRunnerOutput,
) -> dict[int, EngineCoreOutputs]:
```

**处理流程**：

```
for req_id, num_scheduled_tokens in num_scheduled_tokens.items():
  request = self.requests[req_id]

  ① 提取生成的 token：
       generated_token_ids = model_runner_output.sampled_token_ids[req_index]

  ② 处理推测解码的接受/拒绝：
       if request.spec_token_ids:
           num_accepted = len(generated_token_ids) - 1
           num_rejected = num_draft_tokens - num_accepted
           request.num_computed_tokens -= num_rejected  ← 回退被拒绝的 token

  ③ 推进已计算 token 数：
       request.num_computed_tokens += num_scheduled_tokens

  ④ 追加输出 token：
       request.append_output_token_ids(generated_token_ids)

  ⑤ 检查停���条件：
       stopped = check_stop(request, self.max_model_len)
       # 检查：EOS token | max_tokens | stop_strings | stop_token_ids

  ⑥ 如果完成：
       self._free_request(request)   ← 释放 KV 缓存，从 running 移除

  ⑦ 构建 EngineCoreOutput 返回给用户
```

---

## 十二、EngineCore 主循环

调度器如何被调用：

```python
# vllm/v1/engine/core.py
def step(self) -> tuple[dict[int, EngineCoreOutputs], bool]:

    if not self.scheduler.has_requests():
        return {}, False           ← 无请求，不执行

    # ① 调度（CPU 侧，<1ms）
    scheduler_output = self.scheduler.schedule()

    # ② 执行（GPU 侧，几十到几百 ms）
    model_output = self.model_executor.execute_model(scheduler_output)

    # ③ 更新（CPU 侧，<1ms）
    engine_core_outputs = self.scheduler.update_from_output(
        scheduler_output, model_output
    )

    return engine_core_outputs, scheduler_output.total_num_scheduled_tokens > 0
```

**时间比例**：

```
单步时间（70B 模型，batch=32，A100）：
  scheduler.schedule()           ≈ 0.5 ms  （CPU，极快）
  executor.execute_model()       ≈ 50 ms   （GPU，主要耗时）
  scheduler.update_from_output() ≈ 0.5 ms  （CPU，极快）

  调度器开销 < 2%，可以忽略
```

---

## 十三、P/D 分离（Prefill-Decode Disaggregation）集成

vLLM 支持将 Prefill 计算和 Decode 计算放在不同的机器组上：

```
Prefill 节点（算力密集）：
  专门处理 Prompt → 计算 KV Cache → 通过网络传输到 Decode 节点

Decode 节点（显存密集）：
  接收 KV Cache → 执行 Decode 生成 token

调度器中的相关状态：
  WAITING_FOR_REMOTE_KVS：Prefill 已完成，正在等待 KV Cache 传输

  load_kv_async = True：
    num_new_tokens = 0     ← 本步不处理 token，只等待 KV 传输
    status = WAITING_FOR_REMOTE_KVS
    → 后续步骤检查传输是否完成

  KVConnectorBase_V1：
    get_num_new_matched_tokens()   → 查询外部缓存命中数
    update_state_after_alloc()     → 更新连接器状态
    build_connector_meta()         → 构建传输元数据，传给 Worker
```

---

## 十四、推测解码（Speculative Decoding）集成

```
普通 Decode（每步生成 1 token）：
  Step N:   [token_N] → GPU → new_token_{N+1}

推测解码（每步尝试生成 K token）：
  Draft Model 先生成 K 个候选 token
  Step N:   [token_N, draft_{N+1}, ..., draft_{N+K}]  → GPU（一次验证 K 个）
  → 接受 k 个（0 ≤ k ≤ K），实际生成 1+k 个 token

调度器中的处理：

  num_scheduled_tokens = num_tokens + num_lookahead_tokens
  ↑ 为 draft token 预留 KV 空间（虽然可能被拒绝）

  update_from_output() 中处理接受/拒绝：
    num_accepted = len(generated_tokens) - 1
    num_rejected = K - num_accepted
    request.num_computed_tokens -= num_rejected  ← 回退拒绝的 draft token
```

---

## 十五、多模态（编码器）调度

图像/音频等多模态输入需要先经过编码器，生成的特征嵌入到 token 序列中：

```
请求：[文本: 50 tok] [图像] [文本: 100 tok]
      │               │       │
      │    编码器     │       │
      ▼   处理图像    ▼       ▼
      50 tok  +  图像嵌入(256 tok)  +  100 tok = 406 decoder tokens

调度器决策：
  ① 如果当前 decode 范围不包含图像位置 → 跳过编码器，节省算力
  ② 如果即将处理图像位置 → 调用编码器，计算图像特征
  ③ 编码器有独立的 budget（max_num_encoder_input_tokens）
  ④ 编码结果缓存（encoder_cache_manager），避免重复编码相同图像

关键代码：
  _try_schedule_encoder_inputs(request, num_computed_tokens, num_new_tokens)
  → 返回：需要调用的编码器输入列表 + 调整后的 decoder token 数
```

---

## 十六、性能优化细节

### 16.1 通用前缀块（Common Prefix Blocks）

```python
num_common_prefix_blocks = kv_cache_manager.get_num_common_prefix_blocks(req_id)
```

所有运行请求共享的前缀块（如：相同 system prompt）可以用 Cascade Attention 优化：
- 前缀的 KV 只计算一次
- Attention 分两阶段：prefix attention + request-specific attention
- 适合"同一 system prompt + 不同用户问题"的场景

### 16.2 增量更新减少 CPU-GPU 传输

```
NewRequestData（第一次调度，~KB 级别）：
  需要传：完整 prompt token ids（可能 4096 tokens × 4 bytes = 16KB）
  仅发送一次！

CachedRequestData（后续调度，<100 bytes）：
  只传：新生成的 1-2 个 token ids + 新分配的块 ID
  每步 ~几十字节，接近零开销
```

### 16.3 请求状态批量传输

```python
scheduled_cached_reqs: CachedRequestData  # 注意：是一个对象，包含所有请求的列表

# 而非：
# scheduled_cached_reqs: list[CachedRequestData]  # 逐请求对象

批量设计避免了 Python 对象创建的 GC 压力
```

---

## 十七、代码架构总览

```
vllm/v1/core/sched/
│
├── scheduler.py             ← 主调度器（1300+ 行）
│   class Scheduler:
│     - schedule()           主调度函数（三阶段）
│     - update_from_output() 处理 GPU 输出
│     - add_request()        新请求加入
│     - finish_requests()    结束请求
│     - _preempt()           抢占逻辑（内部）
│
├── output.py                ← 调度结果数据结构
│   class SchedulerOutput    传给 Executor 的完整调度信息
│   class NewRequestData     新请求数据
│   class CachedRequestData  增量更新数据
│
├── request_queue.py         ← 等待队列实现
│   class FIFORequestQueue   先来先服务（deque）
│   class PriorityRequestQueue  优先级队列（heapq）
│
└── output.py（check_stop 等辅助函数）
```

---

## 十八、总结

```
vLLM 调度器的核心设计原则：

1. Continuous Batching（持续批处理）：
   每步动态决定请求集合，不等批次凑满
   → 消除传统 Static Batching 的 GPU 空闲问题

2. Token Budget 限制（防过载）：
   max_num_scheduled_tokens 控制单步计算量
   → 保证每步延迟可预测（不因单个长 prefill 阻塞）

3. Chunked Prefill（分块预填充）：
   长 prefill 分多步完成
   → decode 请求不被长 prefill 阻塞，TTFT 降低

4. LRU 抢占（资源保障）：
   KV 内存不足时驱逐低优先级请求
   → 前缀缓存减轻抢占代价

5. 前缀缓存集成（复用计算）：
   调度时自动检查命中，减少 prefill 计算量
   → System Prompt 场景命中率 90%+

6. 增量更新（减少传输）：
   只向 Worker 传变化的部分（新 token、新块 ID）
   → CPU-GPU 通信开销接近零
```

| 机制 | 解决的问题 | 效果 |
|------|-----------|------|
| Continuous Batching | GPU 利用率低 | 吞吐 2-4× 提升 |
| Token Budget | 单步延迟不可控 | P99 延迟稳定 |
| Chunked Prefill | 长 prefill 阻塞 decode | TTFT 大幅降低 |
| LRU 抢占 | KV 内存溢出崩溃 | 系统稳定性保障 |
| 前缀缓存 | 重复计算 system prompt | TTF降低 50%+ |
| 增量更新 | CPU-GPU 数据传输开销 | 传输量降低 99% |

---

*参考资料：*
- *[vLLM 源码：scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)*
- *[Continuous Batching 论文：Orca（Yu et al. 2022）](https://www.usenix.org/conference/osdi22/presentation/yu)*
- *[vLLM 原始论文（Kwon et al. 2023）](https://arxiv.org/abs/2309.06180)*
- *[Sarathi-Serve（Chunked Prefill）：Agrawal et al. 2024](https://arxiv.org/abs/2403.02310)*
*更新：2026-03*
