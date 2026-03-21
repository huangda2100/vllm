# vLLM 的 KV Cache 管理机制

> **核心问题**：LLM 推理时，KV Cache 是显存的最大消耗者之一。
> vLLM 如何让有限的 GPU 显存服务尽可能多的并发请求？
>
> **官方文献**：[PagedAttention 论文（Kwon et al. 2023）](https://arxiv.org/abs/2309.06180)
> **代码路径**：`vllm/v1/core/`（调度侧）、`vllm/v1/worker/`（执行侧）

---

## 一、KV Cache 是什么，为什么需要管理它

### 1.1 KV Cache 的来源

Transformer 的自注意力计算中，每个 token 在每一层都会产生 Key 和 Value 向量：

```
第 t 步 Decode（生成第 t 个 token）：

  输入：token t 的 hidden state h_t: [1, hidden_size]
                      ↓ K/V 投影
  K_t: [1, num_kv_heads, head_size]
  V_t: [1, num_kv_heads, head_size]

  注意力：Q_t 需要与所有历史的 K, V 做运算：
    A = softmax(Q_t × [K_0, K_1, ..., K_t]^T / √d)
    out = A × [V_0, V_1, ..., V_t]

  如果不缓存，每步都要重新计算 K_0~K_{t-1} → 计算量 O(t²)
  如果缓存，只需读取已存的 K/V → 计算量 O(t)，时间换空间

KV Cache = 所有历史 token 的 K 和 V 向量，存储在 GPU 显存中
```

### 1.2 KV Cache 的显存占用

以 LLaMA-3-8B（32 层，8 KV 头，head_size=128，BF16）为例：

```
单个 token 的 KV Cache 大小：
  每层 K：num_kv_heads × head_size × 2 bytes = 8 × 128 × 2 = 2 KB
  每层 V：同上 = 2 KB
  单层合计：4 KB/token
  32 层合计：4 KB × 32 = 128 KB/token

单个请求（context=4096 tokens）：
  128 KB × 4096 = 512 MB

同时服务 32 个请求：
  512 MB × 32 = 16 GB
  （A100-80G 的 KV Cache 预算约 40-60 GB）

问题：
  ① 不同请求的序列长度不同 → 显存利用率低（内部碎片）
  ② 请求随时到达和结束 → 显存碎片化（外部碎片）
  ③ 部分请求共享相同前缀 → 重复计算浪费
```

---

## 二、PagedAttention：核心创新

### 2.1 类比操作系统的虚拟内存

传统 LLM 推理框架为每个请求**连续分配**一大块显存（类似 OS 的段式内存）：

```
传统方式（连续分配）：
  请求 A (1024 tokens)：[                1024 tokens 的 KV Cache                    ]
  请求 B (512 tokens)：[     512 tokens 的 KV Cache    ][          空闲             ]
  请求 C (2048 tokens)：[                        2048 tokens 的 KV Cache            ]

  问题：
  ① 必须预先知道最大序列长度，按最大值分配 → 浪费
  ② 请求 B 结束后留下碎片，但 C 需要连续空间 → 外部碎片
  ③ 中途无法扩展（旁边可能被占用）
```

**PagedAttention 的解法**：把 KV Cache 分成固定大小的**块（Block）**，类似 OS 的页式内存：

```
PagedAttention（分块分配）：

物理内存（GPU 显存，统一管理的块池）：
  Block 0  | Block 1  | Block 2  | Block 3  | Block 4  | Block 5  | ...
  [tokens  | [tokens  | [tokens  | [tokens  | [FREE    | [tokens  | ...
   0-15 A] |  16-31 A]|  0-15 B] |  32-47 A]|          |  0-15 C] |

  请求 A 的块表（Block Table）：Block 0 → Block 1 → Block 3（不连续！）
  请求 B 的块表：Block 2
  请求 C 的块表：Block 5

优势：
  ① 显存利用率接近 100%（没有内部碎片，只有最后一块的部分浪费）
  ② 不同请求的块可以交错，无外部碎片
  ③ 支持前缀共享（不同请求可以指向同一物理��）
```

### 2.2 块的核心参数

```python
# 典型配置（LLaMA-3-8B）
block_size = 16           # 每块存 16 个 token 的 K/V
num_kv_heads = 8          # GQA
head_size = 128
dtype = torch.bfloat16    # 2 bytes/元素

# 单块大小：
# 2（K+V）× block_size × num_kv_heads × head_size × dtype_bytes
# = 2 × 16 × 8 × 128 × 2 = 65,536 bytes = 64 KB（每层）

# 全模型（32 层）单块大小：
# 64 KB × 32 = 2 MB/block

# A100-80G 可用于 KV Cache 的显存约 40 GB：
# 可用块数 = 40 GB / 2 MB = 20,000 块 = 320,000 个 token 槽位
```

---

## 三、GPU 物理内存布局

### 3.1 KV Cache 张量的存储格式

vLLM 在 GPU 上预分配一大块连续显存，所有块都在这块空间内：

```python
# 每层的 KV Cache 张量形状（Flash Attention NHD 布局）：
# [num_blocks, block_size, 2, num_kv_heads, head_size]
#      ↑           ↑       ↑       ↑            ↑
#   块数量     每块token数  K和V  KV头数量     头维度

# 例：LLaMA-3-8B，1000 块
kv_cache_layer_0 = torch.empty(
    1000,      # num_blocks
    16,        # block_size
    2,         # K(0) 和 V(1)
    8,         # num_kv_heads
    128,       # head_size
    dtype=torch.bfloat16,
    device='cuda'
)
# 总大小：1000 × 16 × 2 × 8 × 128 × 2 bytes = 64 MB（单层）
# 32 层：64 MB × 32 = 2 GB（与之前估算一致）
```

**块内的存储方式（放大一个块）**：

```
Block 0（block_id=0，存 token 0-15 的 KV）：

  kv_cache[0, 0, 0, :, :]  ← token 0 的 K，形状 [num_kv_heads, head_size]
  kv_cache[0, 0, 1, :, :]  ← token 0 的 V，形状 [num_kv_heads, head_size]
  kv_cache[0, 1, 0, :, :]  ← token 1 的 K
  kv_cache[0, 1, 1, :, :]  ← token 1 的 V
  ...
  kv_cache[0, 15, 0, :, :] ← token 15 的 K
  kv_cache[0, 15, 1, :, :] ← token 15 的 V
```

### 3.2 Block Table 和 Slot Mapping

调度器给每个请求维护一个**块表（Block Table）**，把逻辑 token 位置映射到物理块：

```
请求 A（已生成 38 个 token，block_size=16）：
  块表：[Block 5, Block 3, Block 12, null, null, ...]
  逻辑位置 → 物理位置映射：

  token 0-15   → Block 5（block_id=5）的 slot 0-15
  token 16-31  → Block 3（block_id=3）的 slot 0-15
  token 32-37  → Block 12（block_id=12）的 slot 0-5（还未满）
                  ↑ 最后一块是"部分满"的

Slot Mapping（槽位映射，用于 CUDA 核函数写入新 KV）：
  每个新 token 需要知道"我的 KV 应该写到哪个物理位置"：
  slot = block_id × block_size + position_within_block

  token 38 应写入：Block 12 的第 6 个槽位
  slot_mapping[38] = 12 × 16 + 6 = 198
```

**Block Table 在 GPU 上的表示**：

```python
# 形状：[max_requests, max_blocks_per_request]
block_table = torch.zeros(
    max_requests,        # 如 256
    max_seq_len // block_size,  # 如 4096/16 = 256
    dtype=torch.int32,
    device='cuda'
)
# block_table[req_idx, block_idx] = physical_block_id
# CUDA 注意力核函数通过 block_table 访问正确的 KV 数据
```

---

## 四、核心数据结构

### 4.1 KVCacheBlock（元数据，在 CPU 上）

```python
# vllm/v1/core/kv_cache_utils.py
@dataclass
class KVCacheBlock:
    # 物理块 ID（0 到 num_gpu_blocks-1）
    block_id: int

    # 引用计数：有多少请求正在使用这个块
    # ref_cnt == 0：空闲，可以被驱逐
    # ref_cnt > 0：正在使用，不可驱逐
    ref_cnt: int = 0

    # 前缀缓存哈希（仅块已满且已缓存时有值）
    _block_hash: BlockHashWithGroupId | None = None

    # LRU 双向链表指针
    prev_free_block: KVCacheBlock | None = None
    next_free_block: KVCacheBlock | None = None

    # 是否是占位符块（Sliding Window 的空洞用）
    is_null: bool = False
```

**关键：KVCacheBlock 只是元数据（几十字节），不存储真实 KV 数据！**
真实的 K/V 向量在 GPU 显存的 `kv_cache_tensors` 中，通过 `block_id` 索引。

### 4.2 BlockPool（块池，统一管理所有物理块）

```
BlockPool（所有块的管理中心）：

  [blocks: list[KVCacheBlock]]    ← 所有块的元数据数组（CPU 上）
       ↓
  [free_block_queue]              ← 空闲块的 LRU 双向链表
       ↓
  [null_block]                    ← 特殊占位符块（永不被驱逐）
       ↓
  [cached_block_hash_to_block]    ← 哈希 → 块的映射（前缀缓存查找）

  GPU 显存（真实 KV 数据）：
  [kv_cache_tensors]              ← 物理块数组，通过 block_id 索引
```

### 4.3 FreeKVCacheBlockQueue（LRU 空闲队列）

```
双向链表实现的 LRU 队列：

HEAD ←→ [Block 7] ←→ [Block 2] ←→ [Block 15] ←→ [Block 0] ←→ TAIL
  最旧（最先被驱逐）                              最新（最近释放）

操作复杂度：
  popleft()：O(1)，从头部取块（驱逐最旧的）
  append()：O(1)，从尾部插入（请求释放的块）
  remove()：O(1)，从中间删除（前缀命中时，"激活"缓存块）

驱逐机制（LRU）：
  popleft 时：取出队列头部的块（最长时间没被使用的块）
  如果该块有 prefix cache 哈希：清除哈希（内容作废）
  块 ID 不变，内容会在下次写入时被覆盖
```

---

## 五、前缀缓存（Prefix Caching）

### 5.1 核心思想

如果多个请求有相同的前缀（如：系统提示词 + 不同的用户问题），KV Cache 可以复用：

```
请求 1："你是一个有帮助的助手。用户问题：今天天气怎么样？"
请求 2："你是一个有帮助的助手。用户问题：帮我写一首诗。"

前 N 个 token（系统提示词）完全相同 → 对应的 KV Cache 也相同！

没有前缀缓存：
  请求 1：重新计算所有 token 的 KV
  请求 2：重新计算所有 token 的 KV（前缀部分重复计算！）

有前缀缓存：
  请求 1：计算所有 KV，缓存系统提示词的块
  请求 2：命中前缀缓存 → 直接复用请求 1 的块！
  → 节省系统提示词部分的计算时间
```

### 5.2 哈希机制（Prefix-Aware Chained Hashing）

每个满块（包含 block_size 个 token）都有一个唯一的哈希值，由**内容 + 父块哈希**决定：

```python
def hash_block_tokens(
    parent_block_hash,     # 前驱块的哈希（确保位置感知）
    curr_block_token_ids,  # 本块的 token IDs
    extra_keys,            # LoRA ID、多模态哈希等
) -> BlockHash:
    return hash((parent_block_hash, tuple(curr_block_token_ids), extra_keys))
```

**链式哈希的必要性**：

```
示例（block_size=4）：

两个请求：
  请求 A：[系统提示=100,101,102,103] [用户问题=200,201,202,203]
  请求 B：[系统提示=100,101,102,103] [用户问题=300,301,302,303]

Block 0（两者相同）：tokens=[100,101,102,103]
  Hash_0 = hash(None, [100,101,102,103]) = 0xABCD

请求 A 的 Block 1：tokens=[200,201,202,203]
  Hash_A1 = hash(0xABCD, [200,201,202,203]) = 0x1234

请求 B 的 Block 1：tokens=[300,301,302,303]
  Hash_B1 = hash(0xABCD, [300,301,302,303]) = 0x5678

关键：Hash_A1 ≠ Hash_B1（不同内容，不同哈希）
     但两者的 Block 0 哈希相同 → 可以共享！

为什么需要 parent_block_hash？
  假设请求 C：[完全不同的前缀][tokens=200,201,202,203]
  如果不含 parent hash：Hash_C1 = hash([200,201,202,203]) = 0x1234（与 A1 相同！）
  会错误地复用请求 A 的 Block 1（内容相同但位置含义不同）
  → parent hash 确保哈希与位置绑定，不会跨上下文复用
```

### 5.3 前缀缓存命中检测流程

```python
# vllm/v1/core/single_type_kv_cache_manager.py

def find_longest_cache_hit(
    block_hashes: list[BlockHash],  # 请求预计算的所有块哈希
    max_length: int,
) -> list[KVCacheBlock]:

    computed_blocks = []

    # 从左到右（从序列开始）逐块检查
    for block_hash in block_hashes[:max_blocks]:

        # 在哈希表中查找
        cached_block = block_pool.get_cached_block(block_hash)

        if cached_block is not None:
            computed_blocks.append(cached_block)
        else:
            break  # ← 关键：遇到第一个 miss 就停止！

    return computed_blocks
```

**为什么遇到 miss 就必须停止？**

```
Token 序列：[A B C D | E F G H | I J K L]
               Block 0    Block 1    Block 2

如果 Block 0 命中，Block 1 miss，Block 2 命中：
  不能跳过 Block 1 使用 Block 2！

原因：Block 2 的 KV 值是在 Block 0 + Block 1 都已处理的上下文下计算的
     如果重新计算时跳过 Block 1，attention 的上下文不完整
     → Block 2 缓存的 K/V 对这个请求没有意义
```

**最后一块不缓存（必须留一个 token 重计算）**：

```python
# get_computed_blocks 中：
max_cache_hit_length = request.num_tokens - 1  # 最后一个 token 不算！
```

原因：最后一个 token 的 KV 值需要在"它自己之前"的全部上下文下计算，
前缀缓存只能命中"完整的"历史块。

---

## 六、内存分配流程

### 6.1 整体分配流程图

```
请求到达 → Scheduler 调用 KVCacheManager
                │
                ▼
    get_computed_blocks(request)
         │            │
         ▼            ▼
   计算请求的      在 BlockHashToBlockMap
   所有块哈希      中查找命中的块
         │            │
         └─────────── ▼
                命中块列表（可能为空）
                      │
                      ▼
         allocate_slots(request, num_new_tokens,
                       new_computed_blocks)
                      │
          ┌───────────┼────────────────┐
          ▼           ▼                ▼
    touch 命中块   计算需要新      从 free_block_queue
   （防止被驱逐）  分配的块数      popleft 分配新块
                      │
                      ▼
               新块足够？
               ├── 是 → 分配成功，返回 KVCacheBlocks
               └── 否 → 返回 None（OOM，请求等待）
                         （隐式触发 LRU 驱逐）
```

### 6.2 allocate_slots 详解

```python
def allocate_slots(
    self,
    request: Request,
    num_new_tokens: int,           # 本步骤的新 token 数
    num_new_computed_tokens: int,  # 前缀缓存命中的 token 数
    new_computed_blocks: KVCacheBlocks | None,  # 命中的物理块
) -> KVCacheBlocks | None:

    # ① 释放 Sliding Window 不再需要的旧块
    self.coordinator.remove_skipped_blocks(
        request.request_id,
        request.num_computed_tokens
    )

    # ② 计算总共需要多少块
    num_tokens_need_slot = (
        request.num_computed_tokens     # 已处理的 token
        + num_new_computed_tokens       # 前缀命中的 token
        + num_new_tokens                # 本步骤的新 token
        + num_lookahead_tokens          # 投机解码预留
    )
    num_required_blocks = ceil(num_tokens_need_slot / block_size)

    # ③ 计算需要新分配多少块
    num_new_blocks = (
        num_required_blocks
        - len(new_computed_blocks)            # 前缀命中的块（复用）
        - len(request.already_allocated_blocks) # 之前已分配的块
    )

    # ④ 检查空闲块是否足够
    if num_new_blocks > free_block_count:
        return None  # OOM，返回 None

    # ⑤ "激活"前缀命中的块（从 LRU 队列移除，防止被驱逐）
    if enable_caching:
        block_pool.touch(new_computed_blocks)

    # ⑥ 从空闲队列分配新块（自动 LRU 驱逐）
    new_physical_blocks = block_pool.get_new_blocks(num_new_blocks)

    # ⑦ 缓存本步骤完成的满块（计算并存储哈希）
    if enable_caching:
        self.coordinator.cache_blocks(request, num_tokens_to_cache)

    return KVCacheBlocks(new_physical_blocks)
```

### 6.3 LRU 驱逐机制

```python
# vllm/v1/core/block_pool.py

def get_new_blocks(self, num_blocks: int) -> list[KVCacheBlock]:
    """从空闲队列取块（自动 LRU 驱逐）"""

    ret = self.free_block_queue.popleft_n(num_blocks)  # 取最旧的空闲块

    for block in ret:
        # 如果这个块有前缀缓存哈希，清除它（驱逐）
        if block.block_hash is not None:
            self.cached_block_hash_to_block.pop(block.block_hash)
            block.reset_hash()

        # 标记为已分配
        block.ref_cnt = 1

    return ret
```

**LRU 队列的动态变化**：

```
初始状态：所有块空闲，按 block_id 排列

  HEAD ←→ [0] ←→ [1] ←→ [2] ←→ [3] ←→ ... ←→ [N] ←→ TAIL
  （最旧）                                        （最新）

步骤 1：请求 A 到来，分配 Block 0, 1, 2（popleft_n(3)）
  HEAD ←→ [3] ←→ [4] ←→ ... ←→ [N] ←→ TAIL
  Block 0, 1, 2 的 ref_cnt = 1（不在队列中）

步骤 2：请求 A 完成，释放 Block 0, 1, 2（append 到尾部）
  HEAD ←→ [3] ←→ ... ←→ [N] ←→ [0] ←→ [1] ←→ [2] ←→ TAIL
  Block 0, 1, 2 的 ref_cnt = 0（回到队列，在尾部）
  Block 0, 1, 2 仍有哈希（前缀缓存有效，等待被命中或驱逐）

步骤 3：新请求到来，需要分配 K 个块
  → 从 HEAD（Block 3...）开始取，Block 0, 1, 2 暂时安全
  → 如果需要更多块，才会轮到 Block 0（最新释放，最后被驱逐）

步骤 4：请求 B 命中 Block 0（前缀缓存命中）
  → touch(Block 0)：从队列中间 remove(Block 0)，然后不再放回队列
  → Block 0 的 ref_cnt 变为 1（在使用中，不可驱逐）
  HEAD ←→ [3] ←→ ... ←→ [N] ←→ [1] ←→ [2] ←→ TAIL
  （Block 0 被"激活"，不在空闲队列中）
```

---

## 七、请求释放块的流程

### 7.1 请求完成时

```python
def free(self, request: Request) -> None:
    """请求完成，释放所有块"""

    # 获取该请求的所有物理块
    req_blocks = self.req_to_blocks[request.request_id]

    # 反向遍历（从最后一块开始）
    for block in reversed(req_blocks):
        block.ref_cnt -= 1

        if block.ref_cnt == 0:
            # 没有其他请求引用这个块
            if block.block_hash is not None:
                # 有前缀缓存哈希 → 不清除，保留在缓存中
                # 把块加入 LRU 队列尾部（等待被命中或被新分配驱逐）
                self.free_block_queue.append(block)
            else:
                # 没有哈希（末尾的不完整块或 caching 未启用）
                # 直接加入 LRU 队列，内容不保留
                self.free_block_queue.append(block)

    del self.req_to_blocks[request.request_id]
```

**关键设计**：块被释放时，**不立即清除 GPU 上的 KV 数据**，只是把块元数据放回 LRU 队列。如果后来有请求命中，可以立即复用这些 K/V 值（Prompt 的 KV 依然有效）。

### 7.2 前缀共享的引用计数

```
情景：请求 A 和 B 共享 Block 5（系统提示词的前缀块）

  Block 5.ref_cnt = 2（A 和 B 各持有一份引用）

请求 A 完成：
  Block 5.ref_cnt -= 1 → ref_cnt = 1（B 还在用，不放回队列）

请求 B 完成：
  Block 5.ref_cnt -= 1 → ref_cnt = 0（没有请求使用了）
  → append(Block 5) 到 LRU 队列尾部
  → Block 5 的哈希保留（等待后续请求复用）
```

---

## 八、完整架构层次

### 8.1 代码结构与职责

```
vllm/v1/core/
│
├── kv_cache_manager.py          ← 对外 API（Scheduler 调用）
│   class KVCacheManager:
│     - get_computed_blocks()    获取前缀命中
│     - allocate_slots()         分配新块
│     - free()                   释放块
│     - empty_kv_cache_blocks    空结果单例（GC 优化）
│
├── kv_cache_coordinator.py      ← 多类型注意力的路由器
│   class KVCacheCoordinator:
│     - 协调 FullAttention + SlidingWindow 的混合管理
│
├── single_type_kv_cache_manager.py  ← 单种注意力类型的管理逻辑
│   class FullAttentionManager:
│     - find_longest_cache_hit()  从左到右贪心匹配
│   class SlidingWindowManager:
│     - find_longest_cache_hit()  从右到左搜索有效窗口
│
├── block_pool.py                ← 物理块的统一管理
│   class BlockPool:
│     - get_new_blocks()         分配（触发 LRU 驱逐）
│     - touch()                  激活前缀块（移出 LRU 队列）
│     - cache_full_blocks()      计算哈希，加入前缀缓存
│
└── kv_cache_utils.py            ← 数据结构和哈希工具
    class KVCacheBlock            物理块元数据
    class FreeKVCacheBlockQueue   LRU 双向链表
    class BlockHashToBlockMap     哈希 ��� 块的索引
    hash_block_tokens()           链式哈希计算
```

### 8.2 调度器到 GPU 的完整数据流

```
① 请求到达（Scheduler 侧，CPU）：
   Request → kv_cache_manager.get_computed_blocks()
           → 返回：命中的 KVCacheBlock 列表 + 命中 token 数

② 分配（Scheduler 侧，CPU）：
   → kv_cache_manager.allocate_slots()
   → 返回：新分配的 KVCacheBlock 列表（包含 block_id）

③ 构建 GPU 元数据（ModelRunner 侧，CPU→GPU）：
   block_ids = [block.block_id for block in all_blocks]
   block_table[req_idx] = block_ids  ← 写入 GPU 上的 block_table 张量
   slot_mapping[token_pos] = block_id * block_size + within_block_pos

④ 前向传播（GPU 执行）：
   for layer in model.layers:
       # 注意力层通过 block_table 和 slot_mapping 访问 KV Cache
       new_kv → write to kv_cache_tensor[slot_mapping]  ← 写入新 KV
       attn_output = flash_attn(Q, kv_cache_tensor, block_table) ← 读历史 KV

⑤ 完成请求（Scheduler 侧，CPU）：
   → kv_cache_manager.free(request)
   → 释放的块加入 LRU 队列，保留哈希（等待前缀复用）
```

---

## 九、Sliding Window Attention 的特殊处理

```
Sliding Window Attention（如 Mistral-7B，window=4096 tokens）：
  每个 token 只能 attend 到最近 4096 个 token
  超出窗口的旧块不再需要 → 可以提前释放！

  例：block_size=16，window=4096（256 块）

  请求生成 300 个块后：
  Block 0（token 0-15）：已在窗口外 → 可以释放
  Block 1（token 16-31）：已在窗口外 → 可以释放
  ...
  Block 44（token 704-719）：仍在最新 256 个块之内 → 保留

SlidingWindowManager 的特���逻辑：
  - 每步调用 remove_skipped_blocks() 释放窗口外的块
  - 前缀缓存命中时从右到左搜索（找最近的有效块组）
  - 使用 null_block 作为窗口外位置的占位符

内存优势：
  Dense 模型（context=32K）：每请求最多 2000 块
  Sliding Window（window=4K）：每请求最多 256 块（持续释放旧块！）
  → 相同显存可以服务 8× 更多并发请求
```

---

## 十、GPU 显存利用率优化

### 10.1 统计实际效果

```
传统方式（连续分配，无前缀缓存）：
  显存利用率：~30-60%（大量内部碎片 + 外部碎片）
  KV Cache 命中率：0%（每次重新计算）

vLLM PagedAttention：
  显存利用率：~95%+（只有每个请求最后一个不完整块有部分浪费）
  KV Cache 命中率：视工作负载，系统提示词命中率可达 90%+

量化对比（LLaMA-2-13B，A100-80G）：
  传统：最大并发 ~20 请求（分配 4096 tokens/请求 × 20 = 80GB）
  vLLM：最大并发 ~200 请求（大多数请求 <1000 token 使用量）
  → 10× 并发提升
```

### 10.2 显存预算分配

```
A100-80G 的显存预算（LLaMA-3-8B 推理）：

  模型权重（BF16）：~16 GB
  激活值（运行时）：~2 GB
  KV Cache 预留：  ~58 GB（其余全给 KV Cache！）

  58 GB / 2 MB（每块，32 层，BF16）≈ 29,000 块
  29,000 块 × 16 tokens/块 = 464,000 个 token 槽位

  实际上 vLLM 在启动时通过"试运行"确定 num_gpu_blocks：
  1. 加载模型权重
  2. 运行一次最大 batch 的前向传播（测量峰值显存）
  3. 剩余显存 / 每块大小 = KV Cache 块数
  4. 预分配这些块（一次性分配，避免碎片）
```

### 10.3 empty_kv_cache_blocks 单例优化

```python
# vllm/v1/core/kv_cache_manager.py

# 避免每次前缀缓存未命中都创建新对象
self.empty_kv_cache_blocks = KVCacheBlocks(
    blocks=tuple([] for _ in range(len(kv_cache_groups)))
)

# 使用时直接返回单例，不创建新对象
def get_computed_blocks(self, request):
    if not enable_caching:
        return self.empty_kv_cache_blocks, 0   # ← 单例复用
    ...
```

这个细节避免了在高并发场景下每次未命中都产生 GC 压力。

---

## 十一、KV Cache 的未来优化方向

### vLLM 已支持的扩展

```
1. MLA（DeepSeek 的 Multi-head Latent Attention）：
   KVCacheSpec: MLAAttentionSpec
   压缩 KV 到更低维度（每 token KV 从 128 KB → 几 KB）
   → KV Cache 大小缩小 5-10×

2. 分层 KV Cache（Prefix KV 卸载到 CPU/NVMe）：
   热块：GPU 显存（毫秒访问）
   温块：CPU 内存（几十毫秒访问）
   冷块：NVMe SSD（百毫秒访问）
   → 扩展有效 KV Cache 容量数十倍

3. 分布式 KV Cache（多机共享）：
   通过 kv_cache_events（BlockStored/BlockRemoved 事件）
   → 不同节点的 KV Cache 可以通过 P2P 传输共享
   → 适合跨节点的 Prefill-Decode 分离部署

4. KV Cache 量化（FP8 / INT8）：
   BF16 → FP8：KV Cache 显存减半
   vllm/v1/kv_cache_interface.py 中 MLAAttentionSpec 支持 cache_dtype_str
```

---

## 十二、总结

```
vLLM KV Cache 管理的核心设计原则：

1. PagedAttention（分页）：
   消除碎片，显存利用率从 ~50% 提升到 ~95%

2. LRU 双向链表（O(1) 驱逐）：
   高效管理空闲块，无需堆/排序，O(1) 所有操作

3. 链式哈希（前缀缓存）：
   位���感知的内容哈希，安全地跨请求共享 KV Cache

4. 引用计数（共享块安全）：
   多个请求可以指向同一物理块，ref_cnt=0 才允许回收

5. 隐式驱逐（分配即驱逐）：
   不需要显式的驱逐操作，popleft 时自动清除最旧的缓存

6. CPU-GPU 分离：
   块元数据（KVCacheBlock）在 CPU，真实 KV 数据在 GPU
   调度逻辑高效（CPU），计算高效（GPU）
```

| 机制 | 解决的问题 | 效果 |
|------|----------|------|
| PagedAttention | 显存碎片 | 利用率 50% → 95% |
| LRU 双向链表 | 驱逐复杂度 | O(1) 分配/驱逐 |
| 前缀缓存 | 重复计算前缀 | 命中率 50-90%，延迟降低 50%+ |
| 引用计数 | 多请求共享安全 | 无拷贝的前缀共享 |
| Sliding Window | 长文本显存浪费 | 8× 并发提升（window 模型）|
| MLA 支持 | KV Cache 体积 | 每 token KV 缩小 5-10× |

---

*参考资料：*
- *[PagedAttention 论文：Kwon et al. 2023](https://arxiv.org/abs/2309.06180)*
- *[vLLM 源码：kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)*
- *[vLLM 源码：block_pool.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/block_pool.py)*
- *[vLLM 源码：kv_cache_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_utils.py)*
- *[DeepSeek MLA：DeepSeek-V2 技术报告](https://arxiv.org/abs/2405.04434)*
*更新：2026-03*
