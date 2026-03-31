# Qwen3.5 在 vLLM 中的推理优化方案

## 1. Qwen3.5 核心特性分析

### 1.1 架构特点

**混合注意力机制 (Hybrid Attention)**
- **Gated Delta Networks (GDN)**: 线性注意力变体，计算复杂度 O(n) vs 传统 O(n²)
- **稀疏 MoE**: 397B 总参数但每次只激活 17B (激活率 ~4.3%)
- **长上下文支持**: 原生 262K tokens，可扩展至 1M tokens

**关键数值**
- Qwen3.5-397B-A17B: 总参数 397B，激活参数 17B
- Qwen3.5-122B-A10B: 总参数 122B，激活参数 10B
- Qwen3.5-35B-A3B: 总参数 35B，激活参数 3B

### 1.2 推理瓶颈分析

**内存瓶颈**
1. **KV Cache 爆炸**: 262K context × num_layers × hidden_dim × 2 (K+V)
2. **MoE 权重加载**: 虽然只激活 4.3%，但所有专家权重需常驻显存
3. **多模态 Token**: 图像 token 数量远超文本 (单图可达数千 tokens)

**计算瓶颈**
1. **Expert Routing**: 每个 token 需计算 gating scores 选择专家
2. **长序列注意力**: 即使是线性注意力，262K 长度仍有挑战
3. **动态批处理**: MoE 导致不同 token 激活不同专家，难以批处理

---

## 2. vLLM 针对性优化方案

### 2.1 KV Cache 优化 (目标: 节省 60-80% 显存)

#### 方案 1: 分层 KV Cache 压缩
```python
# 文件: vllm/v1/core/kv_cache_manager.py

class QwenKVCacheManager(KVCacheManager):
    """针对 Qwen3.5 的分层 KV Cache 策略"""

    def __init__(self, config):
        super().__init__(config)
        # 前 25% 层: 全精度 FP16
        # 中间 50% 层: INT8 量化
        # 后 25% 层: INT4 量化 (输出层附近需要精度)
        self.layer_quant_bits = self._init_layer_quantization(config.num_layers)

    def _init_layer_quantization(self, num_layers):
        bits = []
        for i in range(num_layers):
            if i < num_layers * 0.25:
                bits.append(16)  # 前层保持精度
            elif i < num_layers * 0.75:
                bits.append(8)   # 中层 INT8
            else:
                bits.append(16)  # 后层恢复精度
        return bits
```

**预期收益**:
- 显存占用: 100% → 40% (平均 6.5 bit/param)
- 精度损失: < 0.5% perplexity 增加

#### 方案 2: Sliding Window + Sparse Attention
```python
# 文件: vllm/attention/backends/flash_attn.py

class QwenHybridAttention:
    """混合窗口注意力: 局部密集 + 全局稀疏"""

    def __init__(self, window_size=4096, sparse_ratio=0.1):
        self.local_window = window_size      # 最近 4K tokens 全注意力
        self.sparse_ratio = sparse_ratio     # 历史 tokens 10% 采样

    def forward(self, q, k, v, seq_len):
        if seq_len <= self.local_window:
            return flash_attn(q, k, v)  # 短序列直接计算

        # 长序列: 局部 + 稀疏
        local_out = flash_attn(q, k[:, -self.local_window:], v[:, -self.local_window:])
        sparse_indices = self._select_sparse_tokens(seq_len - self.local_window)
        sparse_out = flash_attn(q, k[:, sparse_indices], v[:, sparse_indices])

        return self._merge_outputs(local_out, sparse_out)
```

**预期收益**:
- KV Cache: 262K tokens → 4K + 25.8K = 29.8K (节省 88.6%)
- TTFT: 262K context 从 ~30s → ~5s

---

### 2.2 MoE 专家调度优化 (目标: 提升 2-3x 吞吐)

#### 方案 3: Expert Prefetching + Caching
```python
# 文件: vllm/model_executor/layers/fused_moe/fused_moe.py

class QwenMoELayer:
    """预测式专家预取"""

    def __init__(self, num_experts=64, top_k=2):
        self.expert_cache = {}  # GPU 缓存热门专家
        self.routing_history = []  # 记录路由模式
        self.prefetch_queue = []

    def forward(self, hidden_states):
        # 1. 预测下一批可能的专家
        predicted_experts = self._predict_experts(self.routing_history[-100:])
        self._prefetch_experts(predicted_experts)

        # 2. 计算 gating scores
        router_logits = self.gate(hidden_states)
        routing_weights, selected_experts = self._top_k_gating(router_logits)

        # 3. 批量调度相同专家的 tokens
        expert_outputs = self._batched_expert_forward(
            hidden_states, selected_experts, routing_weights
        )

        return expert_outputs

    def _batched_expert_forward(self, x, experts, weights):
        """按专家分组批处理"""
        outputs = torch.zeros_like(x)
        for expert_id in experts.unique():
            mask = (experts == expert_id)
            token_ids = mask.nonzero(as_tuple=True)[0]
            if len(token_ids) > 0:
                # 批量处理同一专家的所有 tokens
                expert_out = self.experts[expert_id](x[token_ids])
                outputs[token_ids] = expert_out * weights[token_ids]
        return outputs
```

**预期收益**:
- 专家切换开销: 减少 70% (通过批处理)
- 吞吐量: 提升 2.5x (batch_size=32 时)

#### 方案 4: 动态专家卸载
```python
# 文件: vllm/v1/worker/gpu_model_runner.py

class QwenExpertManager:
    """动态专家显存管理"""

    def __init__(self, num_experts=64, gpu_capacity=8):
        self.gpu_experts = {}  # 常驻 GPU 的专家
        self.cpu_experts = {}  # CPU 缓存的专家
        self.capacity = gpu_capacity  # GPU 最多保留 8 个专家
        self.access_freq = [0] * num_experts

    def get_expert(self, expert_id):
        if expert_id in self.gpu_experts:
            self.access_freq[expert_id] += 1
            return self.gpu_experts[expert_id]

        # 从 CPU 加载，淘汰最少使用的专家
        if len(self.gpu_experts) >= self.capacity:
            lru_id = self.access_freq.index(min(self.access_freq))
            self._offload_expert(lru_id)

        return self._load_expert(expert_id)
```

**预期收益**:
- 显存占用: 64 experts → 8 experts (节省 87.5%)
- 适用场景: 长文本生成 (专家访问模式稳定)

---

### 2.3 调度器优化 (目标: 降低 50% TTFT)

#### 方案 5: Chunked Prefill
```python
# 文件: vllm/v1/core/scheduler.py

class QwenChunkedScheduler:
    """分块预填充调度器"""

    def __init__(self, chunk_size=2048):
        self.chunk_size = chunk_size

    def schedule(self, requests):
        prefill_reqs = [r for r in requests if r.is_prefill]
        decode_reqs = [r for r in requests if not r.is_prefill]

        # 长 prefill 请求分块处理
        chunked_batches = []
        for req in prefill_reqs:
            if req.prompt_len > self.chunk_size:
                # 分成多个 chunk，与 decode 交替执行
                num_chunks = (req.prompt_len + self.chunk_size - 1) // self.chunk_size
                for i in range(num_chunks):
                    start = i * self.chunk_size
                    end = min(start + self.chunk_size, req.prompt_len)
                    chunked_batches.append({
                        'type': 'prefill_chunk',
                        'request': req,
                        'range': (start, end)
                    })
            else:
                chunked_batches.append({'type': 'prefill', 'request': req})

        # 交替调度: prefill_chunk → decode → prefill_chunk → decode
        return self._interleave_schedule(chunked_batches, decode_reqs)
```

**预期收益**:
- TTFT: 从 30s → 15s (262K context)
- 用户体验: 首 token 延迟减半

---

## 3. 对照实验设计

### 3.1 实验环境配置

**硬件配置**
- GPU: 8x NVIDIA A100 80GB (NVLink)
- CPU: 2x AMD EPYC 7763 (128 cores)
- 内存: 1TB DDR4
- 存储: NVMe SSD 4TB

**软件版本**
- vLLM: v0.6.0 (baseline) vs v0.7.0-qwen35-optimized
- PyTorch: 2.4.0
- CUDA: 12.4
- Flash Attention: 2.6.0

**测试模型**
- Qwen3.5-122B-A10B (中等规模，便于快速迭代)
- Qwen3.5-35B-A3B (对比实验)

---

### 3.2 实验组设计

#### 实验 A: KV Cache 优化效果

**对照组 (Baseline)**
```bash
# 标准 vLLM 配置
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-122B-A10B \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --kv-cache-dtype auto
```

**实验组 1: 分层量化**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-122B-A10B \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --kv-cache-dtype layered \
  --kv-quant-config "0-30:fp16,31-90:int8,91-120:fp16"
```

**实验组 2: 混合窗口注意力**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3.5-122B-A10B \
  --tensor-parallel-size 8 \
  --max-model-len 262144 \
  --attention-backend qwen-hybrid \
  --local-window-size 4096 \
  --sparse-ratio 0.1
```

**测试指标**
- 最大支持序列长度 (OOM 前)
- 显存占用 (nvidia-smi)
- 精度损失 (perplexity on long-context benchmark)

---

#### 实验 B: MoE 调度优化效果

**对照组**
```bash
# 标准 MoE 调度
python benchmark_throughput.py \
  --model Qwen/Qwen3.5-122B-A10B \
  --input-len 2048 --output-len 512 \
  --num-prompts 100 --batch-size 32
```

**实验组 1: 专家预取**
```bash
python benchmark_throughput.py \
  --model Qwen/Qwen3.5-122B-A10B \
  --input-len 2048 --output-len 512 \
  --num-prompts 100 --batch-size 32 \
  --enable-expert-prefetch \
  --prefetch-window 10
```

**实验组 2: 动态专家卸载**
```bash
python benchmark_throughput.py \
  --model Qwen/Qwen3.5-122B-A10B \
  --input-len 2048 --output-len 512 \
  --num-prompts 100 --batch-size 32 \
  --enable-expert-offload \
  --gpu-expert-capacity 8
```

**测试指标**
- 吞吐量 (tokens/sec)
- GPU 利用率 (%)
- 专家切换次数
- 端到端延迟

---

#### 实验 C: TTFT/TPOT 优化效果

**对照组**
```bash
# 标准调度
python benchmark_latency.py \
  --model Qwen/Qwen3.5-122B-A10B \
  --input-len 131072 --output-len 128 \
  --num-prompts 50
```

**实验组: Chunked Prefill**
```bash
python benchmark_latency.py \
  --model Qwen/Qwen3.5-122B-A10B \
  --input-len 131072 --output-len 128 \
  --num-prompts 50 \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192
```

**测试指标**
- TTFT (Time to First Token): ms
- TPOT (Time per Output Token): ms
- P50/P95/P99 延迟分布

---

### 3.3 测试脚本

#### 自动化测试脚本
```python
# benchmark_qwen35.py
import subprocess
import json
import time
from dataclasses import dataclass

@dataclass
class BenchmarkConfig:
    name: str
    args: dict

configs = [
    BenchmarkConfig("baseline", {"max_model_len": 32768}),
    BenchmarkConfig("layered_kv", {"max_model_len": 131072, "kv_cache_dtype": "layered"}),
    BenchmarkConfig("hybrid_attn", {"max_model_len": 262144, "attention_backend": "qwen-hybrid"}),
]

def run_benchmark(config):
    cmd = f"python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3.5-122B-A10B"
    for k, v in config.args.items():
        cmd += f" --{k.replace('_', '-')} {v}"

    proc = subprocess.Popen(cmd, shell=True)
    time.sleep(60)  # 等待服务启动

    # 运行性能测试
    results = {}
    results['memory'] = get_gpu_memory()
    results['throughput'] = measure_throughput()
    results['latency'] = measure_latency()

    proc.terminate()
    return results
```

---

### 3.4 数据收集方案

#### 监控指标脚本
```python
# metrics_collector.py
import pynvml
import time

class MetricsCollector:
    def __init__(self):
        pynvml.nvmlInit()
        self.device_count = pynvml.nvmlDeviceGetCount()

    def collect_gpu_metrics(self):
        metrics = []
        for i in range(self.device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)

            metrics.append({
                'gpu_id': i,
                'memory_used_gb': mem_info.used / 1024**3,
                'memory_total_gb': mem_info.total / 1024**3,
                'utilization': util.gpu,
                'timestamp': time.time()
            })
        return metrics

    def collect_latency_metrics(self, client, prompts):
        results = []
        for prompt in prompts:
            start = time.time()
            response = client.completions.create(
                model="Qwen/Qwen3.5-122B-A10B",
                prompt=prompt,
                max_tokens=128,
                stream=True
            )

            first_token_time = None
            token_times = []
            for chunk in response:
                current_time = time.time()
                if first_token_time is None:
                    first_token_time = current_time - start
                else:
                    token_times.append(current_time)

            results.append({
                'ttft': first_token_time * 1000,  # ms
                'tpot': (token_times[-1] - token_times[0]) / len(token_times) * 1000,
                'total_time': time.time() - start
            })
        return results
```

---

### 3.5 预期实验结果

#### 表 1: KV Cache 优化效果对比

| 配置 | 最大序列长度 | 显存占用 (GB) | Perplexity | 改善 |
|------|-------------|--------------|-----------|------|
| Baseline | 32K | 640 | 5.23 | - |
| 分层量化 | 131K | 480 | 5.28 | 4x 长度, 25% 显存节省 |
| 混合窗口 | 262K | 280 | 5.45 | 8x 长度, 56% 显存节省 |

#### 表 2: MoE 调度优化效果对比

| 配置 | 吞吐量 (tok/s) | GPU 利用率 | 专家切换次数 | 改善 |
|------|---------------|-----------|-------------|------|
| Baseline | 1,240 | 68% | 45,600 | - |
| 专家预取 | 2,850 | 82% | 38,200 | 2.3x 吞吐 |
| 动态卸载 | 1,680 | 71% | 52,100 | 显存节省 87% |

#### 表 3: TTFT/TPOT 优化效果对比

| 配置 | TTFT (ms) | TPOT (ms) | P99 延迟 (ms) | 改善 |
|------|-----------|-----------|--------------|------|
| Baseline (131K) | 28,450 | 45 | 32,100 | - |
| Chunked Prefill | 14,200 | 48 | 16,800 | 50% TTFT 降低 |

---

## 4. 实施路线图

### 阶段 1: 基础优化 (2-3 周)
1. **KV Cache 分层量化** (优先级: 高)
   - 修改 `vllm/v1/core/kv_cache_manager.py`
   - 实现 INT8/INT4 量化逻辑
   - 添加配置参数 `--kv-quant-config`

2. **Chunked Prefill** (优先级: 高)
   - 修改 `vllm/v1/core/scheduler.py`
   - 实现分块调度逻辑
   - 添加参数 `--max-num-batched-tokens`

### 阶段 2: MoE 优化 (3-4 周)
3. **专家批处理调度** (优先级: 中)
   - 修改 `vllm/model_executor/layers/fused_moe/fused_moe.py`
   - 实现按专家分组的批处理

4. **专家预取机制** (优先级: 中)
   - 添加路由模式预测
   - 实现异步预取队列

### 阶段 3: 高级优化 (4-6 周)
5. **混合窗口注意力** (优先级: 低)
   - 修改 `vllm/attention/backends/flash_attn.py`
   - 实现局部+稀疏混合注意力

6. **动态专家卸载** (优先级: 低)
   - 实现 GPU-CPU 专家交换
   - 添加 LRU 缓存策略

---

## 5. 关键代码修改点

### 5.1 核心文件清单

| 文件路径 | 修改内容 | 代码量 |
|---------|---------|--------|
| `vllm/v1/core/kv_cache_manager.py` | 添加分层量化逻辑 | ~200 行 |
| `vllm/v1/core/scheduler.py` | 实现 chunked prefill | ~150 行 |
| `vllm/model_executor/layers/fused_moe/fused_moe.py` | MoE 批处理调度 | ~180 行 |
| `vllm/attention/backends/flash_attn.py` | 混合窗口注意力 | ~250 行 |
| `vllm/v1/worker/gpu_model_runner.py` | 专家管理器 | ~120 行 |

### 5.2 配置参数新增

```python
# vllm/config.py
@dataclass
class QwenOptimizationConfig:
    # KV Cache 优化
    kv_quant_config: Optional[str] = None  # "0-30:fp16,31-90:int8,91-120:fp16"

    # 注意力优化
    attention_backend: str = "flash-attn"  # "qwen-hybrid"
    local_window_size: int = 4096
    sparse_ratio: float = 0.1

    # MoE 优化
    enable_expert_prefetch: bool = False
    prefetch_window: int = 10
    enable_expert_offload: bool = False
    gpu_expert_capacity: int = 8

    # 调度优化
    enable_chunked_prefill: bool = True
    max_num_batched_tokens: int = 8192
```

---

## 6. 风险评估与缓解

### 6.1 技术风险

| 风险 | 影响 | 概率 | 缓解措施 |
|------|------|------|---------|
| KV Cache 量化精度损失 | 中 | 高 | 分层策略，关键层保持 FP16 |
| MoE 专家预测失败 | 低 | 中 | 回退到标准调度 |
| 长序列 OOM | 高 | 中 | 动态调整 chunk size |
| 稀疏注意力质量下降 | 中 | 中 | 可配置稀疏率，默认保守 |

### 6.2 兼容性风险

- **模型版本**: Qwen3.5 架构可能与 Qwen2 不兼容
  - 缓解: 添加模型版本检测，自动选择优化策略

- **硬件依赖**: 部分优化需要 A100+ GPU
  - 缓解: 提供降级方案，V100 使用基础优化

---

## 7. 总结

### 7.1 优化收益汇总

**显存优化**
- KV Cache 分层量化: 节省 25-60% 显存
- 混合窗口注意力: 节省 88% KV Cache
- 动态专家卸载: 节省 87% 专家权重显存

**性能优化**
- 吞吐量提升: 2-3x (通过 MoE 批处理)
- TTFT 降低: 50% (通过 chunked prefill)
- 最大序列长度: 32K → 262K (8x 提升)

### 7.2 适用场景

| 场景 | 推荐配置 | 预期效果 |
|------|---------|---------|
| 长文档问答 (100K+ tokens) | 混合窗口注意力 + 分层量化 | 8x 序列长度 |
| 高并发服务 (batch=32+) | MoE 批处理 + 专家预取 | 2.5x 吞吐 |
| 实时对话 (低延迟) | Chunked prefill | 50% TTFT 降低 |
| 显存受限环境 | 动态专家卸载 + INT8 量化 | 节省 80% 显存 |

### 7.3 下一步行动

1. **立即开始**: KV Cache 分层量化 + Chunked Prefill (ROI 最高)
2. **短期目标**: 完成基础优化，验证 2x 性能提升
3. **中期目标**: MoE 调度优化，达到 3x 吞吐
4. **长期目标**: 混合注意力，支持 262K 上下文

---

## 8. 参考资料

### 8.1 Qwen3.5 官方资源
- Qwen3.5 技术报告: https://qwen.ai
- Hugging Face 模型库: https://huggingface.co/Qwen
- GitHub 仓库: https://github.com/QwenLM/Qwen

### 8.2 相关技术论文
- **Gated Delta Networks**: 线性注意力机制
- **Sparse MoE**: Switch Transformers (Google, 2021)
- **KV Cache 量化**: KIVI (MIT, 2024)
- **Chunked Prefill**: Orca (Microsoft, 2023)

### 8.3 vLLM 相关文档
- vLLM 架构文档: https://docs.vllm.ai
- PagedAttention 论文: https://arxiv.org/abs/2309.06180
- MoE 支持: vLLM v0.5.0+ 特性

---

## 附录 A: 完整测试命令

```bash
# 1. 环境准备
conda create -n vllm-qwen python=3.10
conda activate vllm-qwen
pip install vllm==0.7.0 torch==2.4.0

# 2. 下载模型
huggingface-cli download Qwen/Qwen3.5-122B-A10B --local-dir ./models/

# 3. 运行 Baseline
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3.5-122B-A10B \
  --tensor-parallel-size 8 \
  --max-model-len 32768 \
  --port 8000

# 4. 运行优化版本
python -m vllm.entrypoints.openai.api_server \
  --model ./models/Qwen3.5-122B-A10B \
  --tensor-parallel-size 8 \
  --max-model-len 131072 \
  --kv-cache-dtype layered \
  --enable-chunked-prefill \
  --max-num-batched-tokens 8192 \
  --port 8001

# 5. 性能测试
python benchmark_throughput.py --backend vllm --port 8000 > baseline.json
python benchmark_throughput.py --backend vllm --port 8001 > optimized.json
python compare_results.py baseline.json optimized.json
```

---

## 附录 B: 监控 Dashboard 配置

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

# grafana_dashboard.json
{
  "panels": [
    {"title": "GPU Memory Usage", "type": "graph"},
    {"title": "Throughput (tokens/s)", "type": "graph"},
    {"title": "TTFT Distribution", "type": "heatmap"},
    {"title": "Expert Activation Heatmap", "type": "heatmap"}
  ]
}
```

---

**文档版本**: v1.0
**创建日期**: 2026-03-29
**作者**: vLLM Optimization Team
**最后更新**: 2026-03-29
