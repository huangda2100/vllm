# vLLM + DeepSeek-R1 在 Kubernetes 上的完整部署指南

## 目录
1. [准备工作与硬件要求](#1-准备工作与硬件要求)
2. [模型下载](#2-模型下载)
3. [Docker 镜像制作](#3-docker-镜像制作)
4. [Kubernetes 集群部署](#4-kubernetes-集群部署)
5. [部署验证](#5-部署验证)
6. [社区流行部署方案](#6-社区流行部署方案)
7. [vLLM GPU 性能优化](#7-vllm-gpu-性能优化)
8. [生产运维建议](#8-生产运维建议)

---

## 1. 准备工作与硬件要求

### 1.1 DeepSeek-R1 模型系列

| 模型 | 参数量 | 激活参数 | 最低 GPU 需求 | 推荐配置 |
|------|--------|---------|-------------|---------|
| DeepSeek-R1-Distill-Qwen-1.5B | 1.5B | 1.5B | 1×A10G (24GB) | 1×A100 |
| DeepSeek-R1-Distill-Qwen-7B | 7B | 7B | 1×A100 (40GB) | 1×A100-80G |
| DeepSeek-R1-Distill-Llama-8B | 8B | 8B | 1×A100 (40GB) | 1×A100-80G |
| DeepSeek-R1-Distill-Qwen-14B | 14B | 14B | 1×A100-80G | 2×A100-80G |
| DeepSeek-R1-Distill-Llama-70B | 70B | 70B | 4×A100-80G | 8×A100-80G |
| DeepSeek-R1（完整版，MoE） | 671B | ~37B | 8×H100 × 8节点 | 16×H100-80G |

> **注意**：DeepSeek-R1 完整版是 671B MoE 模型（同 DeepSeek-V3 架构），需要多机多卡。
> 生产环境建议优先使用 Distill 版本（蒸馏模型，Dense 架构，部署简单）。

### 1.2 k8s 集群前置条件

```bash
# 检查 NVIDIA GPU 驱动（每个节点）
nvidia-smi

# 安装 NVIDIA Device Plugin（如未安装）
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.15.0/nvidia-device-plugin.yml

# 验证 GPU 资源
kubectl describe nodes | grep nvidia.com/gpu

# 安装 NVIDIA Container Toolkit（每个节点）
# Ubuntu：
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=containerd
sudo systemctl restart containerd
```

### 1.3 存储规划

```bash
# 模型文件大小参考（fp16）
# DeepSeek-R1-Distill-7B:  ~15GB
# DeepSeek-R1-Distill-70B: ~140GB
# DeepSeek-R1（完整版）:   ~1.3TB

# 推荐使用共享存储（NFS / Ceph / 云存储）挂载到所有 GPU 节点
# 避免每个节点重复下载
```

---

## 2. 模型下载

### 2.1 方案一：使用 huggingface-cli（推荐）

```bash
# 安装 huggingface_hub
pip install -U huggingface_hub hf_transfer

# 启用高速传输（多线程下载）
export HF_HUB_ENABLE_HF_TRANSFER=1

# 下载 Distill-7B（示例）
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --local-dir /models/deepseek-r1-distill-7b \
  --local-dir-use-symlinks False

# 下载完整版 R1（671B），建议指定 token 并后台运行
export HF_TOKEN=<your_hf_token>
nohup huggingface-cli download deepseek-ai/DeepSeek-R1 \
  --local-dir /models/deepseek-r1 \
  --local-dir-use-symlinks False \
  > /tmp/download.log 2>&1 &
```

### 2.2 方案二：使用 ModelScope（国内镜像，推荐国内用户）

```bash
pip install modelscope

# 下载模型
python -c "
from modelscope import snapshot_download
snapshot_download(
    'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B',
    cache_dir='/models',
    local_dir='/models/deepseek-r1-distill-7b'
)
"
```

### 2.3 在 k8s 中用 Job 批量下载

```yaml
# download-model-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: download-deepseek-r1
  namespace: vllm
spec:
  template:
    spec:
      restartPolicy: OnFailure
      containers:
      - name: downloader
        image: python:3.11-slim
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
        - name: HF_HUB_ENABLE_HF_TRANSFER
          value: "1"
        command:
        - /bin/sh
        - -c
        - |
          pip install -q huggingface_hub hf_transfer && \
          huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
            --local-dir /models/deepseek-r1-distill-7b \
            --local-dir-use-symlinks False
        volumeMounts:
        - name: model-storage
          mountPath: /models
        resources:
          requests:
            memory: 4Gi
            cpu: 2
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
# 创建 HF Token Secret
kubectl create secret generic hf-token-secret \
  --from-literal=token=<your_hf_token> \
  -n vllm
```

### 2.4 验证下载完整性

```bash
# 检查文件列表
ls -lh /models/deepseek-r1-distill-7b/
# 应包含：config.json, tokenizer.json, *.safetensors, generation_config.json

# 快速检查 config
python3 -c "
import json
with open('/models/deepseek-r1-distill-7b/config.json') as f:
    cfg = json.load(f)
print(f'Model: {cfg.get(\"model_type\")}, Layers: {cfg.get(\"num_hidden_layers\")}')
"
```

---

## 3. Docker 镜像制作

### 3.1 方案一：基于官方镜像（推荐）

```dockerfile
# Dockerfile.vllm-deepseek
# 直接使用 vLLM 官方镜像，仅添加自定义配置
FROM vllm/vllm-openai:latest

# 安装额外依赖（可选）
RUN pip install --no-cache-dir \
    prometheus-client \
    opentelemetry-sdk \
    opentelemetry-exporter-otlp

# 健康检查脚本
COPY scripts/healthcheck.sh /usr/local/bin/healthcheck.sh
RUN chmod +x /usr/local/bin/healthcheck.sh

# 启动脚本
COPY scripts/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
```

```bash
# scripts/entrypoint.sh
#!/bin/bash
set -e

MODEL_PATH=${MODEL_PATH:-/models/deepseek-r1-distill-7b}
TP_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEM=${GPU_MEMORY_UTILIZATION:-0.9}
MAX_LEN=${MAX_MODEL_LEN:-16384}
MAX_SEQS=${MAX_NUM_SEQS:-64}

exec vllm serve "${MODEL_PATH}" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size "${TP_SIZE}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --max-model-len "${MAX_LEN}" \
  --max-num-seqs "${MAX_SEQS}" \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code \
  --served-model-name "deepseek-r1" \
  --dtype auto
```

```bash
# 构建镜像
docker build -t my-registry.example.com/vllm-deepseek:v1.0 -f Dockerfile.vllm-deepseek .

# 推送到私有仓库
docker push my-registry.example.com/vllm-deepseek:v1.0
```

### 3.2 方案二：从源码构建（需要自定义内核时）

```dockerfile
# Dockerfile.custom（基于 vLLM 官方 Dockerfile）
ARG CUDA_VERSION=12.8.1
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build

ARG PYTHON_VERSION=3.12
ARG TORCH_VERSION=2.6.0
ARG VLLM_VERSION=v0.8.0

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    git wget cmake ninja-build \
    python${PYTHON_VERSION} python${PYTHON_VERSION}-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装 PyTorch
RUN pip install torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cu124

# 克隆并构建 vLLM
RUN git clone --depth 1 --branch ${VLLM_VERSION} https://github.com/vllm-project/vllm.git /workspace/vllm
WORKDIR /workspace/vllm

ENV MAX_JOBS=4
ENV NVCC_THREADS=4
RUN pip install -e ".[all]" --no-build-isolation

# ── 运行阶段 ──
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

COPY --from=build /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=build /usr/local/bin/vllm /usr/local/bin/vllm

EXPOSE 8000
```

### 3.3 镜像版本管理建议

```bash
# 推荐的镜像标签策略
my-registry.example.com/vllm-deepseek:0.8.0-cuda12.8-py3.12      # 精确版本
my-registry.example.com/vllm-deepseek:latest                       # 最新稳定版
my-registry.example.com/vllm-deepseek:0.8.0-cuda12.8-py3.12-fp8   # FP8量化版本

# 验证镜像可用性
docker run --rm --gpus all my-registry.example.com/vllm-deepseek:v1.0 \
  vllm --version
```

---

## 4. Kubernetes 集群部署

### 4.1 命名空间与 RBAC

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: vllm
  labels:
    app: vllm
```

### 4.2 持久化存储（PVC）

```yaml
# pvc-models.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: model-pvc
  namespace: vllm
spec:
  accessModes:
    - ReadWriteMany        # 多节点共享读取
  storageClassName: nfs-client  # 根据集群存储类修改
  resources:
    requests:
      storage: 200Gi       # 根据模型大小调整
---
# 如使用 hostPath（单节点测试）
apiVersion: v1
kind: PersistentVolume
metadata:
  name: model-pv-local
spec:
  capacity:
    storage: 500Gi
  accessModes:
    - ReadWriteMany
  hostPath:
    path: /data/models
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - gpu-node-01
```

### 4.3 ConfigMap（服务配置）

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: vllm-config
  namespace: vllm
data:
  # DeepSeek-R1-Distill-7B（单 GPU 配置）
  MODEL_PATH: "/models/deepseek-r1-distill-7b"
  TENSOR_PARALLEL_SIZE: "1"
  GPU_MEMORY_UTILIZATION: "0.90"
  MAX_MODEL_LEN: "32768"
  MAX_NUM_SEQS: "64"
  MAX_NUM_BATCHED_TOKENS: "32768"
  DTYPE: "bfloat16"
  SERVED_MODEL_NAME: "deepseek-r1"
  BLOCK_SIZE: "16"
```

### 4.4 主 Deployment（Distill-7B，单 GPU）

```yaml
# deployment-7b.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deepseek-r1
  namespace: vllm
  labels:
    app: vllm-deepseek-r1
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-deepseek-r1
  template:
    metadata:
      labels:
        app: vllm-deepseek-r1
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      # 调度到有 GPU 的节点
      nodeSelector:
        nvidia.com/gpu.present: "true"

      # GPU 节点容忍（如 GPU 节点有 taint）
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"

      containers:
      - name: vllm-server
        image: my-registry.example.com/vllm-deepseek:v1.0
        imagePullPolicy: IfNotPresent

        ports:
        - containerPort: 8000
          name: http
          protocol: TCP

        # 从 ConfigMap 注入环境变量
        envFrom:
        - configMapRef:
            name: vllm-config

        # 额外环境变量
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: NCCL_DEBUG
          value: "WARN"
        - name: VLLM_LOGGING_LEVEL
          value: "INFO"
        # Prometheus 指标
        - name: VLLM_PROMETHEUS_PORT
          value: "8000"

        # 启动命令
        command:
        - vllm
        - serve
        - "$(MODEL_PATH)"
        - --host
        - "0.0.0.0"
        - --port
        - "8000"
        - --tensor-parallel-size
        - "$(TENSOR_PARALLEL_SIZE)"
        - --gpu-memory-utilization
        - "$(GPU_MEMORY_UTILIZATION)"
        - --max-model-len
        - "$(MAX_MODEL_LEN)"
        - --max-num-seqs
        - "$(MAX_NUM_SEQS)"
        - --max-num-batched-tokens
        - "$(MAX_NUM_BATCHED_TOKENS)"
        - --dtype
        - "$(DTYPE)"
        - --served-model-name
        - "$(SERVED_MODEL_NAME)"
        - --block-size
        - "$(BLOCK_SIZE)"
        - --enable-prefix-caching
        - --enable-chunked-prefill
        - --trust-remote-code

        # GPU 资源
        resources:
          requests:
            cpu: "8"
            memory: "32Gi"
            nvidia.com/gpu: "1"
          limits:
            cpu: "16"
            memory: "64Gi"
            nvidia.com/gpu: "1"

        # 模型存储挂载
        volumeMounts:
        - name: model-storage
          mountPath: /models
          readOnly: true
        - name: shm
          mountPath: /dev/shm    # 共享内存（NCCL 通信必须）

        # 健康检查
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120   # 模型加载需要时间
          periodSeconds: 10
          failureThreshold: 6
          successThreshold: 1

        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 180
          periodSeconds: 30
          failureThreshold: 3

        startupProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 15
          failureThreshold: 40     # 最多等待 60 + 40×15 = 660s 启动

      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "16Gi"       # 共享内存大小

      # 优雅终止
      terminationGracePeriodSeconds: 60
```

### 4.5 多 GPU 部署（Distill-70B，TP=4）

```yaml
# deployment-70b-tp4.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-deepseek-r1-70b
  namespace: vllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm-deepseek-r1-70b
  template:
    metadata:
      labels:
        app: vllm-deepseek-r1-70b
    spec:
      # 亲和性：确保所有 GPU 在同一节点（TP 需要 NVLink）
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.count
                operator: Gt
                values: ["3"]   # 至少 4 块 GPU

      containers:
      - name: vllm-server
        image: my-registry.example.com/vllm-deepseek:v1.0

        command:
        - vllm
        - serve
        - /models/deepseek-r1-distill-70b
        - --tensor-parallel-size
        - "4"
        - --gpu-memory-utilization
        - "0.90"
        - --max-model-len
        - "32768"
        - --max-num-seqs
        - "32"
        - --dtype
        - "bfloat16"
        - --enable-prefix-caching
        - --enable-chunked-prefill
        - --trust-remote-code
        - --host
        - "0.0.0.0"
        - --port
        - "8000"

        env:
        - name: NCCL_DEBUG
          value: "WARN"
        - name: NCCL_SOCKET_IFNAME
          value: "eth0"   # 根据网卡名调整

        resources:
          requests:
            cpu: "32"
            memory: "128Gi"
            nvidia.com/gpu: "4"   # 申请 4 块 GPU
          limits:
            cpu: "64"
            memory: "256Gi"
            nvidia.com/gpu: "4"

        volumeMounts:
        - name: model-storage
          mountPath: /models
        - name: shm
          mountPath: /dev/shm

        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 300   # 70B 加载时间更长
          periodSeconds: 15
          failureThreshold: 20

      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "32Gi"
```

### 4.6 Service（服务暴露）

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-deepseek-r1
  namespace: vllm
  labels:
    app: vllm-deepseek-r1
spec:
  selector:
    app: vllm-deepseek-r1
  ports:
  - name: http
    port: 80
    targetPort: 8000
    protocol: TCP
  type: ClusterIP   # 集群内访问；生产环境用 LoadBalancer 或 Ingress
---
# Ingress（对外暴露，可选）
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-ingress
  namespace: vllm
  annotations:
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
spec:
  rules:
  - host: vllm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-deepseek-r1
            port:
              number: 80
```

### 4.7 HPA（自动扩缩）

```yaml
# hpa.yaml
# 注意：vLLM 实例通常不像无状态服务那样线性扩展（每个副本都要加载完整模型）
# HPA 适合 Distill 小模型多副本横向扩展（数据并行）
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
  namespace: vllm
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm-deepseek-r1
  minReplicas: 1
  maxReplicas: 4
  metrics:
  - type: Pods
    pods:
      metric:
        name: vllm:request_queue_size     # 自定义指标（需 Prometheus Adapter）
      target:
        type: AverageValue
        averageValue: "10"               # 每个 Pod 队列超 10 请求时扩容
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300    # 缩容前等待 5 分钟（避免频繁缩容）
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 30
      policies:
      - type: Pods
        value: 1
        periodSeconds: 60
```

### 4.8 使用 Helm 部署（vLLM 官方 Helm Chart）

```bash
# 官方 Helm Chart 位于 examples/online_serving/chart-helm/
# 克隆 vLLM 仓库或单独获取 chart

# 自定义 values.yaml
cat > my-values.yaml << 'EOF'
image:
  repository: "my-registry.example.com/vllm-deepseek"
  tag: "v1.0"
  command:
    - vllm
    - serve
    - /models/deepseek-r1-distill-7b
    - --served-model-name
    - deepseek-r1
    - --tensor-parallel-size
    - "1"
    - --gpu-memory-utilization
    - "0.90"
    - --max-model-len
    - "32768"
    - --dtype
    - bfloat16
    - --enable-prefix-caching
    - --enable-chunked-prefill
    - --trust-remote-code
    - --host
    - "0.0.0.0"
    - --port
    - "8000"

replicaCount: 1
containerPort: 8000
servicePort: 80

resources:
  requests:
    cpu: 8
    memory: 32Gi
    nvidia.com/gpu: 1
  limits:
    cpu: 16
    memory: 64Gi
    nvidia.com/gpu: 1

readinessProbe:
  initialDelaySeconds: 120
  periodSeconds: 10
  failureThreshold: 30
  path: /health
  port: 8000

livenessProbe:
  initialDelaySeconds: 180
  periodSeconds: 30
  failureThreshold: 3
  path: /health
  port: 8000

extraVolumes:
  - name: model-storage
    persistentVolumeClaim:
      claimName: model-pvc
  - name: shm
    emptyDir:
      medium: Memory
      sizeLimit: "16Gi"

extraVolumeMounts:
  - name: model-storage
    mountPath: /models
  - name: shm
    mountPath: /dev/shm
EOF

# 安装/升级
helm upgrade --install vllm-deepseek ./chart-helm \
  --namespace vllm \
  --create-namespace \
  -f my-values.yaml

# 查看状态
helm status vllm-deepseek -n vllm
```

---

## 5. 部署验证

### 5.1 检查 Pod 状态

```bash
# 查看 Pod 状态
kubectl get pods -n vllm -w

# 查看启动日志（重点关注模型加载进度）
kubectl logs -n vllm deployment/vllm-deepseek-r1 -f

# 正常启动日志应包含：
# INFO: Loading model...
# INFO: Model loaded in X.Xs
# INFO: Started server process
# INFO: Uvicorn running on http://0.0.0.0:8000

# 检查 GPU 使用情况
kubectl exec -n vllm deployment/vllm-deepseek-r1 -- nvidia-smi
```

### 5.2 健康检查

```bash
# 端口转发到本地（测试用）
kubectl port-forward -n vllm svc/vllm-deepseek-r1 8080:80 &

# 基础健康检查
curl -s http://localhost:8080/health
# 正常返回：{"status":"ok"}

# 查看模型信息
curl -s http://localhost:8080/v1/models | python3 -m json.tool
# 应看到 "deepseek-r1" 模型

# 查看服务指标
curl -s http://localhost:8080/metrics | grep vllm_
```

### 5.3 功能验证

```bash
# 基础推理测试（非流式）
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [
      {"role": "user", "content": "请用一句话介绍你自己"}
    ],
    "max_tokens": 200,
    "temperature": 0.6
  }' | python3 -m json.tool

# 流式输出测试
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [{"role": "user", "content": "1+1等于几？"}],
    "max_tokens": 100,
    "stream": true
  }'

# 推理链测试（R1 特性）
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-r1",
    "messages": [
      {"role": "user", "content": "一个数学问题：斐波那契数列第10项是多少？请展示推理过程"}
    ],
    "max_tokens": 1000,
    "temperature": 0.6
  }' | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[\"choices\"][0][\"message\"][\"content\"])"
```

### 5.4 性能基准测试

```bash
# 使用 vLLM 自带的基准测试工具
kubectl exec -n vllm deployment/vllm-deepseek-r1 -- \
  python3 -m vllm.benchmarks.benchmark_serving \
    --backend openai \
    --base-url http://localhost:8000 \
    --model deepseek-r1 \
    --dataset-name random \
    --num-prompts 100 \
    --request-rate 10 \
    --max-concurrency 20

# 关注指标：
# - Throughput (tokens/s)
# - Time to First Token (TTFT)
# - Time Per Output Token (TPOT)
# - P50/P99 latency
```

### 5.5 使用 Python 客户端验证

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="not-needed"  # vLLM 默认不需要 API key
)

# 测试 chat completion
response = client.chat.completions.create(
    model="deepseek-r1",
    messages=[
        {"role": "system", "content": "你是一个有帮助的助手。"},
        {"role": "user", "content": "写一段关于深度学习的简介，100字以内"}
    ],
    max_tokens=300,
    temperature=0.7,
    stream=False
)

print(f"模型: {response.model}")
print(f"输入 tokens: {response.usage.prompt_tokens}")
print(f"输出 tokens: {response.usage.completion_tokens}")
print(f"回复: {response.choices[0].message.content}")
```

---

## 6. 社区流行部署方案

### 6.1 方案对比总览

| 方案 | 适用场景 | 特点 | 难度 |
|------|---------|------|------|
| 原生 vLLM + k8s Helm | 生产环境，自主管控 | 最灵活，官方支持 | 中 |
| **BentoML + BentoCloud** | 快速上线，托管平台 | 简单，内置扩缩容 | 低 |
| **KServe + vLLM Runtime** | 企业级 MLOps | 标准化，支持金丝雀发布 | 中高 |
| **Ray Serve** | 多模型编排，大规模 | 灵活，支持 PP+TP | 中 |
| **LMDeploy** | 国内社区，低资源 | TurboMind 引擎，快 | 低 |
| **Ollama** | 本地开发，快速验证 | 极简，不适合生产 | 极低 |
| **LLM Operator（k8s）** | k8s 原生 | CRD 管理，自动化 | 中高 |

---

### 6.2 方案一：Ray Serve（多节点大规模）

适合 DeepSeek-R1 完整版（671B，多机多卡）

```python
# ray_serve_deepseek.py（参考 vLLM examples/online_serving/ray_serve_deepseek.py）
import ray
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import build_async_engine_client

@serve.deployment(
    ray_actor_options={
        "num_gpus": 8,      # 每个 replica 使用 8 块 GPU
        "num_cpus": 32,
    },
    num_replicas=2,         # 2 个副本���数据并行）
    max_ongoing_requests=50,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(self):
        self.engine_args = AsyncEngineArgs(
            model="/models/deepseek-r1",
            tensor_parallel_size=8,
            pipeline_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.92,
            max_model_len=32768,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            # MoE 专家并行
            enable_expert_parallel=True,
            enable_eplb=True,
            trust_remote_code=True,
        )

# 启动 Ray Serve
ray.init(address="auto")   # 连接已有 Ray 集群
serve.run(VLLMDeployment.bind(), host="0.0.0.0", port=8000)
```

```bash
# 在 k8s 上使用 KubeRay 部署 Ray 集群
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/master/ray-operator/config/crd/bases/ray.io_rayclusters.yaml
kubectl apply -f https://raw.githubusercontent.com/ray-project/kuberay/master/ray-operator/config/crd/bases/ray.io_rayjobs.yaml

# Ray Cluster 配置
cat > raycluster-deepseek.yaml << 'EOF'
apiVersion: ray.io/v1
kind: RayCluster
metadata:
  name: deepseek-cluster
  namespace: vllm
spec:
  headGroupSpec:
    rayStartParams:
      dashboard-host: "0.0.0.0"
    template:
      spec:
        containers:
        - name: ray-head
          image: my-registry.example.com/vllm-deepseek:v1.0
          resources:
            limits:
              cpu: "16"
              memory: "64Gi"
              nvidia.com/gpu: "8"
  workerGroupSpecs:
  - groupName: gpu-workers
    replicas: 1
    minReplicas: 1
    maxReplicas: 4
    rayStartParams: {}
    template:
      spec:
        containers:
        - name: ray-worker
          image: my-registry.example.com/vllm-deepseek:v1.0
          resources:
            limits:
              cpu: "16"
              memory: "64Gi"
              nvidia.com/gpu: "8"
EOF
kubectl apply -f raycluster-deepseek.yaml
```

---

### 6.3 方案二：KServe + vLLM Runtime（企业 MLOps）

```yaml
# inferenceservice-deepseek.yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: deepseek-r1
  namespace: kserve-test
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    model:
      modelFormat:
        name: vllm
      storageUri: "pvc://model-pvc/deepseek-r1-distill-7b"
      resources:
        requests:
          cpu: "8"
          memory: "32Gi"
          nvidia.com/gpu: "1"
        limits:
          nvidia.com/gpu: "1"
      args:
      - --tensor-parallel-size=1
      - --gpu-memory-utilization=0.90
      - --max-model-len=32768
      - --enable-prefix-caching
      - --dtype=bfloat16
      - --trust-remote-code
```

```bash
kubectl apply -f inferenceservice-deepseek.yaml

# 金丝雀发布（逐步切流量）
kubectl patch inferenceservice deepseek-r1 -n kserve-test \
  --type merge \
  -p '{"spec":{"predictor":{"canaryTrafficPercent": 20}}}'
```

---

### 6.4 方案三：LMDeploy（轻量高效，国内社区流行）

```bash
# 使用 LMDeploy 的 TurboMind 引擎（比 vLLM 在某些场景更快）
docker run --gpus all \
  -v /models:/models \
  -p 23333:23333 \
  openmmlab/lmdeploy:latest \
  lmdeploy serve api_server \
    /models/deepseek-r1-distill-7b \
    --backend turbomind \
    --tp 1 \
    --max-batch-size 64 \
    --cache-max-entry-count 0.8 \
    --model-name deepseek-r1
```

---

### 6.5 方案四：NVIDIA NIM（企业级，开箱即用）

```bash
# NVIDIA NIM 提供 DeepSeek-R1 的优化容器（需要 NVIDIA NGC 账号）
docker login nvcr.io

docker run --gpus all \
  -e NGC_API_KEY=$NGC_API_KEY \
  -v /models:/opt/nim/.cache \
  -p 8000:8000 \
  nvcr.io/nim/deepseek-ai/deepseek-r1:1.0.0
```

---

### 6.6 方案五：Sealos/FastGPT 一站式平台

国内社区方案，提供 Web UI + API 管理：

```bash
# Sealos 内置 AI 应用商店，一键部署 DeepSeek
# https://cloud.sealos.io → 应用商店 → AI → DeepSeek-R1

# 或通过 FastGPT 的 OneAPI 中间层管理多个模型
```

---

## 7. vLLM GPU 性能优化

### 7.1 关键参数速查表

| 参数 | 类型 | 默认值 | 影响 | DeepSeek-R1 推荐 |
|------|------|--------|------|-----------------|
| `--tensor-parallel-size` | 并行 | 1 | 多 GPU 切分权重 | 与 GPU 数相同 |
| `--pipeline-parallel-size` | ���行 | 1 | 多节点层间分割 | 节点数 |
| `--gpu-memory-utilization` | 内存 | 0.90 | KV Cache 分配比例 | 0.90~0.95 |
| `--max-num-seqs` | 调度 | 128 | 最大并发请求数 | 32~64（大模型） |
| `--max-num-batched-tokens` | 调度 | auto | 每次迭代最大 token 数 | 32768 |
| `--enable-prefix-caching` | KV | - | 相同前缀复用 KV | ✅ 必开 |
| `--enable-chunked-prefill` | 调度 | - | 长 prompt 分块处理 | ✅ 必开 |
| `--block-size` | KV | 16 | KV Cache 块大小 | 16（H100 用 32） |
| `--kv-cache-dtype` | 量化 | auto | KV Cache 精度 | fp8（A100/H100） |
| `--dtype` | 精度 | auto | 权重精度 | bfloat16 或 fp8 |
| `--quantization` | 量化 | - | 权重量化方案 | fp8（H100）或 awq |
| `--enforce-eager` | 计算 | False | 禁用 CUDA Graph | False（Dense模型）|
| `--enable-expert-parallel` | MoE | False | 专家并行 | True（完整R1） |
| `--enable-eplb` | MoE | False | 专家负载均衡 | True（完整R1） |
| `--max-model-len` | 内存 | 模型默认 | 最大序列长度 | 32768 或 131072 |

---

### 7.2 并行策略优化

#### Tensor Parallel（TP）——单机多卡首选

```bash
# 单机 4×A100-80G，Distill-70B
vllm serve /models/deepseek-r1-distill-70b \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92
```

**TP 原则**：
- TP 在同一节点 NVLink 互联，通信延迟 ~1μs（PCIe 跨卡 ~10μs）
- TP size 应为 GPU 数量（不要跨节点做 TP，改用 PP）
- 对 Attention Head 数有整除要求（`num_heads % tp_size == 0`）

#### Pipeline Parallel（PP）——多节点必用

```bash
# 2 节点 × 8 GPU，完整 R1（671B）
vllm serve /models/deepseek-r1 \
  --tensor-parallel-size 8 \      # 节点内 TP=8
  --pipeline-parallel-size 2 \    # 跨节点 PP=2
  --distributed-executor-backend ray
```

#### Expert Parallel（EP）——MoE 完整版专用

```bash
# DeepSeek-R1 完整版（671B MoE）推荐配置
export VLLM_ALL2ALL_BACKEND="deepep_high_throughput"
export VLLM_USE_DEEP_GEMM=1   # 启用 DeepGEMM FP8 内核

vllm serve /models/deepseek-r1 \
  --tensor-parallel-size 8 \
  --enable-expert-parallel \    # 专家并行
  --enable-eplb \               # 负载均衡
  --num-redundant-experts 4 \   # 热门专家冗余副本
  --dtype bfloat16 \
  --trust-remote-code
```

---

### 7.3 内存优化

#### KV Cache 量化（节省 50% KV 内存）

```bash
# FP8 KV Cache（A100/H100 支持）
vllm serve /models/deepseek-r1-distill-7b \
  --kv-cache-dtype fp8 \          # KV 缓存用 FP8
  --calculate-kv-scales           # 自动计算量化缩放系数（推荐）
```

#### 权重量化（减小模型内存占用）

```bash
# FP8 权重量化（H100 推荐，几乎无精度损失）
vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --quantization fp8 \
  --dtype auto

# AWQ INT4（旧 GPU 或内存受限）
# 需先用 AWQ 量化工具转换模型
vllm serve /models/deepseek-r1-awq \
  --quantization awq \
  --dtype half
```

#### 调整 GPU 内存分配

```bash
# 默认 90% GPU 内存用于 KV Cache
# 如果遇到 OOM，降低到 0.85
--gpu-memory-utilization 0.85

# CPU 卸载（显存不足时，KV Cache 溢出到 CPU）
--cpu-offload-gb 20    # 20GB CPU 内存作为 KV Cache 溢出空间
--swap-space 8         # 8GB swap

# 限制最大序列长度（减少单请求最大 KV 占用）
--max-model-len 16384  # 从 128K 降到 16K，节省大量 KV 空间
```

---

### 7.4 吞吐量优化

#### Continuous Batching 调优

```bash
# 核心参数：每次迭代处理的 token 预算
--max-num-batched-tokens 32768    # 大值 → 高吞吐，高延迟
--max-num-seqs 64                 # 最大并发序列

# Chunked Prefill：长 prompt 不阻塞 decode
--enable-chunked-prefill
--max-num-partial-prefills 2      # 同时 prefill 的分块数
```

**权衡**：
```
max-num-batched-tokens 大 → 每步处理更多 token → 吞吐高，但 TTFT 增大
max-num-batched-tokens 小 → decode 响应快 → 低延迟，但吞吐低
```

#### Prefix Caching（相同系统 prompt 场景必开）

```bash
--enable-prefix-caching
--prefix-caching-hash-algo sha256_cbor   # 更快的 hash 算法

# 效果：相同系统 prompt 的请求，首 token 延迟从 500ms → 5ms
# 适合：RAG 检索结果复用、多轮对话（共享历史）、固定 system prompt
```

#### CUDA Graph（Dense 模型 decode 加速，MoE 慎用）

```bash
# 默认启用 CUDA Graph（--enforce-eager=False）
# CUDA Graph 捕获 decode 阶段（batch=1 到 N 的所有 size）
# 对 batch_size 较小的 decode 阶段效果显著

# MoE 模型（完整 R1）因动态路由，CUDA Graph 收益小，可禁用
--enforce-eager    # 禁用 CUDA Graph（仅 MoE ��考虑）
```

---

### 7.5 延迟优化

#### 推测解码（Speculative Decoding）

```bash
# Eagle3 方案（推荐，vLLM 原生支持）
# 需要额外的 Draft 模型（通常是同架构的小模型）
vllm serve /models/deepseek-r1-distill-7b \
  --speculative-model /models/deepseek-r1-distill-1.5b \  # 草稿模型
  --num-speculative-tokens 4 \    # 每步预测 4 个 token
  --speculative-draft-tensor-parallel-size 1

# MTP（Multi-Token Prediction，DeepSeek-R1 原生支持）
# DeepSeek-R1 训练时已支持 MTP，可直接启用
vllm serve /models/deepseek-r1-distill-7b \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'

# ngram prompt lookup（无需草稿模型，适合长重复序列）
vllm serve /models/deepseek-r1-distill-7b \
  --speculative-config '{"method": "ngram", "prompt_lookup_max": 4}'
```

**推测解码效果**：
- 输入重复性高（代码、翻译）→ 接受率高 → 速度提升 2~3x
- 数学推理链（R1 典型场景）→ 接受率中 → 速度提升 1.3~1.8x

#### 降低首 token 延迟（TTFT）

```bash
# 减小 max-num-batched-tokens（减少 prefill 占用 decode 的时间）
--max-num-batched-tokens 8192    # 而非默认 32768

# 或启用 Chunked Prefill（自动将 prefill 分块，与 decode 交错）
--enable-chunked-prefill
--long-prefill-token-threshold 2048   # 超过 2048 token 的 prefill 才分块
```

---

### 7.6 DeepSeek-R1 特定优化

```bash
# ── DeepSeek-R1-Distill-7B（单 A100-80G，均衡配置）──
vllm serve /models/deepseek-r1-distill-7b \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8 \         # 节省约 50% KV 内存
  --trust-remote-code \
  --served-model-name deepseek-r1

# ── DeepSeek-R1-Distill-70B（4×A100-80G，高吞吐）──
vllm serve /models/deepseek-r1-distill-70b \
  --tensor-parallel-size 4 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 32 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --kv-cache-dtype fp8

# ── DeepSeek-R1（完整版 671B，8×H100-80G，单节点 FP8）──
export VLLM_ALL2ALL_BACKEND=deepep_high_throughput
export VLLM_USE_DEEP_GEMM=1

vllm serve /models/deepseek-r1 \
  --tensor-parallel-size 8 \
  --dtype bfloat16 \
  --quantization fp8 \           # FP8 权重（节省 50% 内存）
  --gpu-memory-utilization 0.92 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --enable-expert-parallel \
  --enable-eplb \
  --num-redundant-experts 8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --trust-remote-code

# R1 思维链特殊处理：
# R1 的 <think>...</think> 可能非常长（数千 token）
# 建议设置合理的 max_tokens 上限以避免单请求占用过多 KV 空间
```

---

### 7.7 监控与调优循环

```bash
# vLLM 内置 Prometheus 指标（/metrics 端点）
# 关键指标：

# 1. 吞吐量
vllm:request_success_total           # 成功请求数
vllm:prompt_tokens_total             # 处理的 prompt token 总量
vllm:generation_tokens_total         # 生成的 token 总量

# 2. 延迟
vllm:time_to_first_token_seconds     # 首 token 延迟
vllm:time_per_output_token_seconds   # 每输出 token 时间
vllm:e2e_request_latency_seconds     # 端到端延迟

# 3. 队列状态
vllm:num_requests_waiting            # 等待队列长度
vllm:num_requests_running            # 正在运行的请求数
vllm:num_requests_swapped            # 被 swap 的请求数（应为 0）

# 4. 内存
vllm:gpu_cache_usage_perc            # GPU KV Cache 使用率（建议 < 90%）
vllm:cpu_cache_usage_perc            # CPU KV Cache 使用率

# 性能调优判断依据：
# num_requests_waiting > 10  → 吞吐瓶颈，考虑增加 max-num-batched-tokens 或副本
# gpu_cache_usage_perc > 95% → KV Cache 满，增加 gpu-memory-utilization 或减少 max-model-len
# time_to_first_token > 2s   → prefill 慢，启用 chunked-prefill
# time_per_output_token > 50ms → decode 慢，启用推测解码
```

---

## 8. 生产运维建议

### 8.1 滚动更新

```bash
# 更新镜像时使用滚动更新
kubectl set image deployment/vllm-deepseek-r1 \
  vllm-server=my-registry.example.com/vllm-deepseek:v1.1 \
  -n vllm

# 设置更新策略（避免中断服务）
kubectl patch deployment vllm-deepseek-r1 -n vllm \
  -p '{"spec":{"strategy":{"type":"RollingUpdate","rollingUpdate":{"maxUnavailable":0,"maxSurge":1}}}}'
```

### 8.2 资源配额

```yaml
# 限制命名空间的 GPU 用量
apiVersion: v1
kind: ResourceQuota
metadata:
  name: gpu-quota
  namespace: vllm
spec:
  hard:
    requests.nvidia.com/gpu: "8"
    limits.nvidia.com/gpu: "8"
    requests.memory: "512Gi"
```

### 8.3 优雅关闭

```bash
# vLLM 支持 SIGTERM 优雅关闭（等待当前请求完成）
# terminationGracePeriodSeconds: 60 确保有足够时间处理中
# 配置 preStop hook 防止流量在关闭前涌入
```

```yaml
lifecycle:
  preStop:
    exec:
      command: ["/bin/sh", "-c", "sleep 10"]  # 等待负载均衡摘除此 Pod
```

### 8.4 日志与告警

```bash
# 关键告警阈值（Prometheus AlertManager）
# 1. GPU 内存使用 > 95%（KV Cache 即将耗尽）
# 2. 请求队列 > 50（服务过载）
# 3. 首 token 延迟 P99 > 5s（用户体验差）
# 4. Pod 重启次数 > 2（服务不稳定）
# 5. GPU 利用率 < 10%（资源浪费）
```

---

## 总结

**部署流程**：下载模型（HF/ModelScope）→ 制作/拉取 vLLM 镜像 → 创建 PVC 挂载模型 → Helm/kubectl 部署 Deployment → Service/Ingress 暴露 → 健康检查验证 → Prometheus 监控。

**核心优化原则**：
1. **必开**：`--enable-prefix-caching` + `--enable-chunked-prefill`（几乎零成本，收益大）
2. **内存**：`--kv-cache-dtype fp8` 节省 50% KV 内存，换取更大 batch 或更长上下文
3. **并行**：单机 TP，多机 TP+PP；MoE 完整版加 EP
4. **延迟**：推测解码（MTP/Eagle）对 R1 思维链场景可显著降低 TPOT
5. **监控**：盯住 `gpu_cache_usage_perc`、`num_requests_waiting`、`time_to_first_token`

**模型选择建议**：实际业务中 DeepSeek-R1-Distill-7B 或 14B 通常是性价比最优选择，节省 80%+ 资源，推理质量对大多数任务足够。
