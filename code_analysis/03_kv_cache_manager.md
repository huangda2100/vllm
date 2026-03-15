# vLLM KV Cache Manager 深度解析

> 文件：`vllm/v1/core/kv_cache_manager.py`（426 行）
> 定位：调度器层的 KV Cache 内存管理，实现 PagedAttention 的内存分配策略

---

## 一、为什么需要 KV Cache Manager？

### KV Cache 是什么？

LLM 在自回归解码时，每生成一个新 token，都需要把它的 Key 和 Value 向量存下来（避免下次重算），这就是 **KV Cache**。

```
Prefill 阶段（prompt = "Hello world"）：
  计算并缓存 K/V: [K_0, K_1, K_2, K_3] 和 [V_0, V_1, V_2, V_3]

Decode 阶段（生成第5个 token）：
  新的 Q_4 需要与历史所有 K/V 做注意力：
  Attention(Q_4, [K_0..K_4], [V_0..V_4])
  → 只需计算 K_4/V_4，直接读取历史缓存
```

### 传统方案的问题

传统方案按最大序列长度预分配连续内存：

```
请求A（实际100 tokens，预分配4096）：浪费 3996 个 KV slots
请求B（实际200 tokens，预分配4096）：浪费 3896 个 KV slots
```

**结果**：GPU 内存利用率极低（<40%），大量显存被"预留"但实际未使用。

### PagedAttention 的解决方案

借鉴操作系统的**虚拟内存分页**思想：
- 将 KV Cache 切成固定大小的 **Block**（如 16 tokens/block）
- 按需分配，用完再要更多
- 不同请求的 Block 可以在物理上不连续
- 多请求可以**共享**前缀相同的 Block（前缀缓存）

---

## 二、核心数据结构

### KVCacheBlock

```
Block = {
    block_id: int       # 物理块编号（GPU 显存中的位置）
    block_hash: Hash    # 内容哈希（用于前缀缓存匹配）
    ref_cnt: int        # 引用计数（多请求共享时>1）
    is_full: bool       # 块是否已满（满了才能缓存）
}
```

### KVCacheBlocks（调度器与Manager的接口）

```python
@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]
    # blocks[kv_cache_group_id][block_index]
    # 外层：不同的 KV Cache 组（如标准注意力、滑动窗口注意力）
    # 内层：该请求的 Block 列表
```

**设计意图**：通过 `KVCacheBlocks` 这个数据类，对调度器**隐藏** KV Cache 内部实现细节。调度器只需知道"有多少个 Block"，不需要知道 Block 的物理布局、哈希逻辑等。

`blocks` 是二维的原因：vLLM 支持同一模型有多种 KV Cache 类型（例如 Mamba 层和 Attention 层使用不同的 KV Cache 规格），需要分组管理。

---

## 三、get_computed_blocks：前缀缓存命中检测

```python
def get_computed_blocks(self, request: Request) -> tuple[KVCacheBlocks, int]:
    # 禁用缓存 or 需要 prompt logprobs → 直接返回空
    if not self.enable_caching or request.sampling_params.prompt_logprobs:
        return self.empty_kv_cache_blocks, 0

    # 必须至少留1个 token 重计算（获取 logits）
    max_cache_hit_length = request.num_tokens - 1

    # 查找最长的前缀缓存命中
    computed_blocks, num_new_computed_tokens = (
        self.coordinator.find_longest_cache_hit(
            request.block_hashes, max_cache_hit_length
        )
    )
    return self.create_kv_cache_blocks(computed_blocks), num_new_computed_tokens
```

### 前缀缓存的工作原理

每个 Block 按其包含的 token 内容计算哈希值：

```
Block 0: hash([system_prompt_tokens_0..15])  →  0xABCD...
Block 1: hash([system_prompt_tokens_16..31]) →  0xEF01...
```

当新请求到来时，按顺序匹配哈希：

```
新请求前缀: [token_0 ... token_47]
  Block 0 哈希 0xABCD 命中 ✓  → 复用物理块 #42
  Block 1 哈希 0xEF01 命中 ✓  → 复用物理块 #17
  Block 2 哈希 0x9999 未命中 → 需要新分配
```

哈希匹配：O(k)（k = 命中的块数），而重新计算注意力：O(n²)（n = 序列长度）。对 System Prompt 很长（如 4096 tokens）的场景，前缀缓存命中率极高，吞吐提升显著。

### 为什么最后1个 token 必须重计算？

```python
max_cache_hit_length = request.num_tokens - 1  # 而不是 num_tokens
```

调度器需要**最后一个 token 的 logits** 来采样下一个 token。如果所有 token 都命中缓存，没有任何计算，就拿不到 logits。因此至少要强制重算最后一个 token（或者最后一个 block，因为 block 必须对齐）。

---

## 四、allocate_slots：Block 分配逻辑

### 内存布局

```
---------------------------------------------------------------------------
| < computed > | < new computed > |    < new >    | < pre-allocated >    |
---------------------------------------------------------------------------
|                  < required >                   |
--------------------------------------------------
|                    < full >                  |
------------------------------------------------
                              | <new full> |
                              --------------
```

- **computed**：已计算好且有 KV Cache 的 tokens（来自前几步）
- **new computed**：刚刚命中前缀缓存的 tokens（本次调度新发现的命中）
- **new**：真正需要本步计算的 tokens（需要分配新 Block）
- **pre-allocated**：为推测解码提前分配的 slots

```python
def allocate_slots(self, request, num_new_tokens, num_new_computed_tokens=0,
                   new_computed_blocks=None, num_lookahead_tokens=0, ...):

    # 1. 释放滑动窗口外的 blocks（减少驱逐压力）
    self.coordinator.remove_skipped_blocks(request.request_id, ...)

    # 2. 计算需要覆盖的总 token 数
    num_computed_tokens = request.num_computed_tokens + num_new_computed_tokens
    num_tokens_need_slot = min(
        num_computed_tokens + num_new_tokens + num_lookahead_tokens,
        self.max_model_len
    )

    # 3. 计算需要新分配的 block 数
    num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(...)

    # 4. 检查是否有足够的空闲 block
    if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
        return None  # 无法调度，OOM 保护

    # 5. touch：防止命中的 blocks 被驱逐
    if self.enable_caching:
        self.block_pool.touch(new_computed_block_list)

    # 6. 保存前缀缓存命中的 blocks 到请求状态
    self.coordinator.save_new_computed_blocks(request.request_id, ...)

    # 7. 分配新 blocks
    new_blocks = self.coordinator.allocate_new_blocks(...)

    # 8. 缓存已完成的 blocks（标记为可被后续请求复用）
    num_tokens_to_cache = min(
        num_computed_tokens + num_new_tokens, request.num_tokens
    )
    self.coordinator.cache_blocks(request, num_tokens_to_cache)

    return self.create_kv_cache_blocks(new_blocks)
```

### touch 操作：防止 LRU 驱逐

```python
self.block_pool.touch(new_computed_block_list)
```

KV Cache 采用 LRU（最近最少使用）驱逐策略。当命中前缀缓存后，那些命中的 Blocks 被"摸了一下"（touch），更新其 LRU 时间戳，确保在本次前向传播期间不会被驱逐。

**时序问题**：必须在 `allocate_new_blocks` **之前** touch，因为分配新 block 可能触发驱逐，而被驱逐的 block 不能是刚命中的前缀缓存 block。

### 为什么只缓存到 num_tokens（而不是 num_tokens_need_slot）？

```python
num_tokens_to_cache = min(
    num_computed_tokens + num_new_tokens,
    request.num_tokens   # ← 只到真实 token 数
)
```

`num_tokens_need_slot` 包含了 **lookahead tokens**（推测解码的草稿 token）。这些 token 可能被 rejection sampling 拒绝，若被拒绝则不应该缓存其 KV。`request.num_tokens` 是当前确定的 token 数，只有这些才是"已确定"可以安全缓存的。

---

## 五、free：逆序释放

```python
def free(self, request: Request) -> None:
    """
    We free the blocks in reverse order so that the tail blocks are
    evicted first when caching is enabled.
    """
    self.coordinator.free(request.request_id)
```

注释明确说明：**逆序释放**，尾部 block 先被驱逐。

**原因**：前缀缓存命中时，前面的 block 更可能被其他请求复用（因为前缀往往是相同的系统提示）。逆序释放意味着：
- 尾部 block（序列末尾，复用价值低）：引用计数降为0，立即可被驱逐
- 头部 block（前缀，复用价值高）：可能仍有其他请求引用（ref_cnt > 1），不会立即驱逐

这是一种**基于复用价值的隐式优先级排序**。

---

## 六、empty_kv_cache_blocks：避免 GC 压力

```python
# 预构造的空 KVCacheBlocks，复用而不是每次创建新对象
self.empty_kv_cache_blocks = KVCacheBlocks(
    tuple(() for _ in range(self.num_kv_cache_groups))
)

def create_kv_cache_blocks(self, blocks):
    # 只有非空时才创建新对象
    return KVCacheBlocks(blocks) if any(blocks) else self.empty_kv_cache_blocks
```

在高 QPS 场景下，调度器每毫秒可能处理数百个请求，每个请求的 prefill 阶段都会调用 `get_computed_blocks`（无论命中与否）。若每次都创建新的 `KVCacheBlocks` 对象，Python GC 压力巨大。

通过复用 `empty_kv_cache_blocks` 单例（注意它是不可变的 nested tuple），避免了大量小对象的创建和销毁。

---

## 七、prefix_cache_stats：可观测性

```python
if self.log_stats:
    if request.num_preemptions > 0:
        self.prefix_cache_stats.preempted_requests += 1
        self.prefix_cache_stats.preempted_hits += num_new_computed_tokens
    else:
        self.prefix_cache_stats.requests += 1
        self.prefix_cache_stats.hits += num_new_computed_tokens
```

统计区分了**新请求**和**被抢占后重新调度的请求**（preemption），因为被抢占的请求命中缓存不代表真正的"前缀共享"收益，而是因为该请求自身之前的计算还在缓存中。分开统计便于分析实际的前缀共享效果。

---

## 八、与 Coordinator 的分层设计

`KVCacheManager` 本身是一个**外观层**（Facade），真正的 block 分配逻辑委托给 `KVCacheCoordinator`：

```python
self.coordinator = get_kv_cache_coordinator(kv_cache_config, ...)
```

这种分层的好处：
- `KVCacheManager` 处理调度器侧的业务逻辑（统计、错误检查、接口转换）
- `KVCacheCoordinator` 处理具体的 block 分配算法（不同模型架构的 block 大小可能不同）
- 两者可以独立测试和替换

---

## 九、关键设计模式总结

| 设计 | 原因 | 收益 |
|------|------|------|
| 分页 Block 分配 | 避免连续内存浪费 | 内存利用率从 <40% 提升到 >90% |
| 内容哈希 + 前缀匹配 | 复用相同前缀的 KV | System Prompt 场景首 token 延迟降低 |
| touch 机制 | 防止刚命中的 block 被驱逐 | 前缀缓存逻辑正确性 |
| 逆序释放 | 优先保留高复用价值的头部 block | 提高前缀缓存命中率 |
| 不缓存 lookahead tokens | 推测 token 可能被拒绝 | 避免错误的 KV 污染缓存 |
| 复用 empty_kv_cache_blocks | 高频调用路径 | 减少 Python GC 压力 |
| KVCacheBlocks 接口隔离 | 解耦调度器与内存管理实现 | 易于替换和测试 |
| 禁用缓存时跳过 prompt_logprobs | 前缀缓存跳过部分前缀 | logprobs 准确性要求完整计算 |

---

*文件：`vllm/v1/core/kv_cache_manager.py`*
*更新：2026-03*
