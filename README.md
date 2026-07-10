# KVCache-Serve：面向 KV Cache 观测的大模型推理服务平台

KVCache-Serve 是一个面向大模型推理服务的个人工程项目，重点关注 KV Cache 的运行时观测、推理延迟指标、异步任务队列、Benchmark 测试以及 Prometheus / Grafana 可观测性。

这个项目不是单独的算法小 demo，而是一个完整的服务系统。当前已经包含 FastAPI API 服务、本地 Transformers 推理后端、Redis Queue、Inference Worker、Prometheus 指标采集、Grafana Dashboard、Benchmark Runner 和 HTML 报告生成。

## 一、项目目标

本项目主要用于学习和实践 LLM Serving / AI Infra / SRE 相关能力。

核心目标包括：

- 搭建一个可运行的大模型推理 API 服务
- 将推理过程拆分为 Prefill 和 Decode 阶段
- 统计 TTFT、ITL、吞吐量、请求延迟等推理指标
- 估算 KV Cache tokens 和 KV Cache memory
- 使用 Redis Queue 实现异步推理任务队列
- 使用 Worker 独立消费任务并执行模型推理
- 使用 Prometheus 采集服务指标
- 使用 Grafana 展示推理服务和 KV Cache 指标
- 使用 Benchmark 脚本生成测试数据和 HTML 报告
- 后续扩展到 Kubernetes 部署和 KV Cache 策略实验

## 二、当前功能

### 1. API 服务

当前支持以下接口：

- GET /health：健康检查
- POST /chat：同步推理接口
- GET /metrics：Prometheus 指标接口
- GET /runtime/status：KV Cache Runtime 状态接口
- POST /queue/chat：异步推理任务提交
- GET /queue/status/{job_id}：查询任务状态
- GET /queue/result/{job_id}：查询任务结果
- GET /queue/health：Redis Queue 健康检查

### 2. 推理后端

当前支持：

- MockBackend：用于快速测试服务链路
- TransformersBackend：基于 Hugging Face Transformers 的本地推理后端

当前默认模型：

- sshleifer/tiny-gpt2

说明：tiny-gpt2 生成质量较差，经常重复 stairs / factors，这属于正常现象。当前模型主要用于验证推理链路、KV Cache 指标和服务架构。

### 3. Prefill / Decode 指标

项目手动拆分了 Transformer 推理过程：

- Prefill：处理完整 prompt，生成 past_key_values
- Decode：逐 token 生成，并复用 past_key_values

当前统计指标：

- latency_ms：总延迟
- prefill_ms：Prefill 阶段延迟
- decode_ms：Decode 阶段延迟
- ttft_ms：Time To First Token
- avg_itl_ms：平均 Inter-Token Latency
- tokens_per_second：生成吞吐
- kv_cache_tokens：KV Cache 中缓存的 token 数
- kv_cache_memory_bytes：KV Cache 估算内存
- kv_cache_memory_mb：KV Cache 估算内存 MB

### 4. Redis Queue + Worker

项目已经支持异步推理架构：

Client
  |
  v
FastAPI API Server
  |
  v
Redis Queue
  |
  v
Inference Worker
  |
  v
Transformers Backend
  |
  v
Result Store in Redis

API 层只负责接收请求和写入队列，Worker 独立消费任务并执行模型推理。这样可以将 API 服务和推理执行解耦，后续可以扩展多个 Worker、限流、队列监控和 Kubernetes 部署。

### 5. Prometheus + Grafana

项目支持 Docker Compose 一键启动可观测性组件：

- Prometheus：采集 API 服务暴露的 /metrics
- Grafana：展示 KVCache-Serve Dashboard

Grafana 当前端口：

- http://localhost:3001

Prometheus 当前端口：

- http://localhost:9090

API 当前端口：

- http://localhost:18000

Grafana Dashboard 包含：

- LLM 请求数
- 请求速率
- 平均请求延迟
- TTFT
- ITL
- Generated Tokens
- KV Cache Tokens
- KV Cache Memory

## 三、项目架构

整体架构如下：

Client / curl / Benchmark
  |
  v
FastAPI API Server
  |
  +--> Sync Chat API
  |
  +--> Redis Queue API
             |
             v
        Redis Queue
             |
             v
      Inference Worker
             |
             v
     Transformers Backend
             |
             v
      Prefill / Decode
             |
             v
      KV Cache Estimator
             |
             v
      KV Cache Runtime
             |
             v
Prometheus Metrics + Grafana Dashboard

## 四、Docker Compose 部署

项目支持 Docker Compose 一键启动。

包含组件：

- kvcache-api
- kvcache-worker
- kvcache-redis
- kvcache-prometheus
- kvcache-grafana

启动：

    docker compose up -d --build

查看容器：

    docker compose ps

停止：

    docker compose down

服务地址：

- API: http://localhost:18000
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3001

## 五、快速测试

健康检查：

    curl http://localhost:18000/health

同步推理：

    curl -X POST "http://localhost:18000/chat" -H "Content-Type: application/json" -d '{"prompt":"Hello, explain KV cache briefly.","model":"local-llm","max_tokens":32}' | python3 -m json.tool

队列健康检查：

    curl http://localhost:18000/queue/health

提交异步任务：

    curl -X POST "http://localhost:18000/queue/chat" -H "Content-Type: application/json" -d '{"prompt":"Hello, explain KV cache briefly.","model":"local-llm","max_tokens":32}'

查询任务状态：

    curl http://localhost:18000/queue/status/{job_id}

查询任务结果：

    curl http://localhost:18000/queue/result/{job_id}

查看 Prometheus 指标：

    curl http://localhost:18000/metrics

Prometheus 查询：

    http://localhost:9090

Grafana Dashboard：

    http://localhost:3001

## 六、Benchmark

运行短 prompt 测试：

    python benchmark/run_benchmark.py --workload benchmark/workloads/short_prompt.json --repeat 2 --concurrency 1 --max-tokens 32

运行中等 prompt 测试：

    python benchmark/run_benchmark.py --workload benchmark/workloads/medium_prompt.json --repeat 2 --concurrency 1 --max-tokens 64

运行长 prompt 测试：

    python benchmark/run_benchmark.py --workload benchmark/workloads/long_prompt.json --repeat 2 --concurrency 1 --max-tokens 64

生成 HTML 报告：

    python benchmark/report_generator.py

打开报告：

    explorer.exe "$(wslpath -w benchmark/results/benchmark_report.html)"

## 七、当前进度

- V0.1 FastAPI 服务骨架：已完成
- V0.2 本地 Transformers 推理：已完成
- V0.3 Prefill / Decode 指标统计：已完成
- V0.4 Inference Backend + KV Cache Runtime：已完成
- V0.5 Benchmark Runner：已完成
- V0.6 HTML Benchmark Report：已完成
- V0.7 Redis Queue + Worker：已完成
- V0.8 Docker Compose 部署：已完成
- V0.9 Prometheus + Grafana：已完成
- V1.0 Grafana Dashboard 自动导入：已完成
- 后续：Kubernetes 部署、Worker 指标暴露、Queue 指标、KV Cache 策略实验

## 八、项目价值

这个项目覆盖了 AI Infra / LLM Serving / SRE 中常见的多个工程点：

- 大模型推理服务化
- KV Cache 运行时观测
- Prefill / Decode 性能分析
- Redis 异步任务队列
- Worker 解耦架构
- Prometheus 指标采集
- Grafana Dashboard 可视化
- Docker Compose 一键部署
- Benchmark 自动化评估

相比单纯复现 KV Cache 算法，这个项目更偏向真实服务系统，适合用于 SRE、DevOps、云原生、AI Infra、LLM Serving 方向的简历和面试展示。

## 九、Kubernetes 部署

项目已经支持在本地 K3s / Kubernetes 环境中部署核心推理链路。

Kubernetes 部署组件包括：

- kvcache-redis：Redis 队列与结果存储
- kvcache-api：FastAPI API 服务
- kvcache-worker：推理 Worker

部署命令：

    kubectl apply -f deploy/k8s/namespace.yaml
    kubectl apply -f deploy/k8s/redis.yaml
    kubectl apply -f deploy/k8s/api.yaml
    kubectl apply -f deploy/k8s/worker.yaml

查看 Pod 状态：

    kubectl get pods -n kvcache-serve

查看 Service：

    kubectl get svc -n kvcache-serve

本地转发 API 服务：

    kubectl port-forward -n kvcache-serve svc/kvcache-api 18001:18000

测试健康检查：

    curl http://localhost:18001/health

测试异步推理链路：

    curl -X POST "http://localhost:18001/queue/chat" -H "Content-Type: application/json" -d '{"prompt":"Hello, explain KV cache briefly.","model":"local-llm","max_tokens":32}'

说明：

如果使用 K3s，本地 Docker 镜像不会自动被 K3s 看到，需要先将镜像导入 K3s containerd：

    docker save kvcache-serve-api:latest -o /tmp/kvcache-serve-api.tar
    docker save kvcache-serve-worker:latest -o /tmp/kvcache-serve-worker.tar
    sudo k3s ctr images import /tmp/kvcache-serve-api.tar
    sudo k3s ctr images import /tmp/kvcache-serve-worker.tar

当前 Kubernetes 部署已经验证通过，API、Redis、Worker 均可正常运行，异步推理任务可以成功完成并返回 KV Cache 指标。
