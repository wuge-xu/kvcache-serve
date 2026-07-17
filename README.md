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

## 十、队列功能验收记录

本项目已经完成 Redis Queue + Worker 异步推理链路验证。

### 验收目标

本次验收目标是确认完整链路：

用户提交任务
  |
  v
API Server 写入 Redis Queue
  |
  v
Worker 从 Redis Queue 消费任务
  |
  v
Transformers Backend 执行模型推理
  |
  v
结果写回 Redis
  |
  v
用户通过 job_id 查询结果

### 本次测试环境

- 部署方式：Kubernetes / K3s
- API 访问方式：kubectl port-forward
- 本地访问端口：http://localhost:18001
- 模型：sshleifer/tiny-gpt2
- 后端：TransformersBackend
- 队列：Redis Queue
- Worker：kvcache-worker

### 健康检查结果

API 健康检查成功：

    curl http://localhost:18001/health

返回：

    {"status":"ok","service":"kvcache-serve","version":"0.4.0"}

队列健康检查成功：

    curl http://localhost:18001/queue/health

返回：

    {"redis":"ok","queue_size":0}

### 本次任务 ID

    d7e14308-5ae0-46e6-80c8-153f86074798

### 状态查询结果

任务提交后，第一次查询状态为：

    {
        "status": "running",
        "job_id": "d7e14308-5ae0-46e6-80c8-153f86074798",
        "queue_size": 0
    }

2 秒后状态变为：

    {
        "status": "finished",
        "job_id": "d7e14308-5ae0-46e6-80c8-153f86074798",
        "queue_size": 0
    }

说明：

- running 表示 Worker 已经从 Redis Queue 中取走任务并开始推理
- finished 表示任务已完成，结果已经写回 Redis
- queue_size 为 0 表示队列中没有积压任务

当前实现中的状态命名为：

- queued：任务已入队
- running：任务处理中，对应 processing
- finished：任务完成，对应 completed

由于 Worker 消费速度较快，queued 状态可能一闪而过，实际测试中观察到了 running 到 finished 的完整状态变化。

### 结果查询结果

最终结果可以正常返回：

    {
        "job_id": "d7e14308-5ae0-46e6-80c8-153f86074798",
        "status": "finished",
        "result": {
            "request_id": "d7e14308-5ae0-46e6-80c8-153f86074798",
            "model": "sshleifer/tiny-gpt2",
            "backend": "transformers",
            "device": "cpu",
            "prompt_tokens": 13,
            "completion_tokens": 32,
            "total_tokens": 45,
            "latency_ms": 75.9,
            "prefill_ms": 47.29,
            "decode_ms": 28.0,
            "ttft_ms": 47.58,
            "avg_itl_ms": 0.89,
            "tokens_per_second": 421.61,
            "kv_cache_tokens": 44,
            "kv_cache_memory_bytes": 1408,
            "kv_cache_memory_mb": 0.001343
        }
    }

说明模型推理结果、token 统计、延迟指标和 KV Cache 指标均正常返回。

### 日志验证

API 日志显示请求链路正常：

    GET /health 200 OK
    GET /queue/health 200 OK
    POST /queue/chat 200 OK
    GET /queue/status/{job_id} 200 OK
    GET /queue/result/{job_id} 200 OK

Worker 日志显示任务被正常消费并完成：

    [Worker] KVCache-Serve inference worker started.
    [Worker] Waiting for tasks from Redis...
    [Worker] Received job: d7e14308-5ae0-46e6-80c8-153f86074798, model=local-llm
    [ModelLoader] Loading model: sshleifer/tiny-gpt2
    [ModelLoader] Device: cpu
    [ModelLoader] Model loaded successfully.
    [Worker] Finished job: d7e14308-5ae0-46e6-80c8-153f86074798

Redis 日志显示 Redis 服务正常启动并接受连接：

    Ready to accept connections tcp

### 验收结论

本次验证确认 Redis Queue + Worker 异步推理链路已经完整跑通：

用户提交任务 -> API 写入 Redis Queue -> Worker 消费任务 -> Transformers Backend 执行模型推理 -> 结果写回 Redis -> 用户通过 job_id 查询结果。


## 十二、任务失败状态验收

队列任务状态已统一为：

- queued：任务已进入 Redis 队列
- processing：Worker 已取出任务并开始推理
- completed：模型推理成功，结果已写回 Redis
- failed：模型推理失败，错误原因已写入 Redis

失败测试使用专门的测试模型名称：

    fail-test

提交失败测试任务：

    BASE_URL=http://localhost:18000

    FAIL_JOB=$(curl -s -X POST "$BASE_URL/queue/chat" -H "Content-Type: application/json" -d '{"prompt":"simulate worker failure","model":"fail-test","max_tokens":8}' | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")

查询状态：

    curl -s "$BASE_URL/queue/status/$FAIL_JOB" | python3 -m json.tool

查询失败结果：

    curl -s "$BASE_URL/queue/result/$FAIL_JOB" | python3 -m json.tool

最终返回内容包含：

    {
        "status": "failed",
        "error": "simulated inference failure",
        "attempts": 2,
        "max_retries": 2
    }

Worker 日志可以看到任务被接收、重新入队并最终失败。由此确认 Worker 能捕获推理异常，Redis 能保存简短错误信息，查询接口能够返回失败状态与错误原因。

## 十三、队列运行统计

项目提供队列运行统计接口：

    GET /queue/stats

调用方式：

    curl -s http://localhost:18000/queue/stats | python3 -m json.tool

返回字段：

- queue_size：当前 Redis 队列中等待消费的任务数量
- jobs_submitted_total：累计提交任务数
- processing_attempts_total：累计推理执行次数，包含重试执行
- jobs_completed_total：累计成功任务数
- jobs_failed_total：累计最终失败任务数
- retries_total：累计重试次数

示例：

    {
        "queue_size": 0,
        "jobs_submitted_total": 2,
        "processing_attempts_total": 4,
        "jobs_completed_total": 1,
        "jobs_failed_total": 1,
        "retries_total": 2
    }

统计数据存储在当前 Redis 实例的 `kvcache:stats` Hash 中，用于观察异步任务提交、执行、成功、失败及重试情况。

## 十三、可靠消费机制第一阶段

早期实现使用 Redis `BLPOP` 从待处理队列领取任务。任务一旦被 Worker 取出，就会立即离开待处理队列。

如果 Worker 在完成推理前发生进程崩溃、容器退出或节点故障，任务可能既不在待处理队列中，也没有生成最终结果，从而造成任务丢失。

当前实现增加了两个 Redis 队列：

- `kvcache:tasks`：待处理队列
- `kvcache:processing`：处理中队列

Worker 使用 Redis `BLMOVE` 命令领取任务：

    BLMOVE kvcache:tasks kvcache:processing LEFT RIGHT

该命令会原子地完成两项操作：

1. 从待处理队列取出任务；
2. 将同一任务放入处理中队列。

因此，任务被 Worker 领取后不会立即从 Redis 中完全消失。

任务处理流程为：

    待处理队列
        ↓ 原子领取
    处理中队列
        ↓
    模型推理
        ↓
    completed 或 failed
        ↓
    确认并从处理中队列删除

正常推理成功后，Worker 保存结果、将状态更新为 `completed`，并从处理中队列删除任务。

任务达到最终失败状态后，Worker 保存简短错误信息、将状态更新为 `failed`，并从处理中队列删除任务。

任务需要重试时，会在一个 Redis 事务中重新放回待处理队列，并删除处理中队列中的旧任务记录。

验证命令：

    docker compose exec -T redis redis-cli LLEN kvcache:tasks
    docker compose exec -T redis redis-cli LLEN kvcache:processing

正常任务测试过程：

- Worker 停止时提交任务：pending=1，processing=0；
- Worker 领取任务后：pending=0，processing=1；
- 推理完成并确认后：pending=0，processing=0；
- 查询接口返回状态 `completed` 和完整推理结果。

本阶段解决的是任务领取后立即从 Redis 完全消失的问题。

当前阶段尚未实现处理中任务的超时扫描和自动恢复。如果 Worker 在推理过程中直接崩溃，任务会保留在 `kvcache:processing` 中，后续可以通过超时回收机制重新放回待处理队列。


## 可靠消费机制：Processing 队列与 ACK

早期版本使用 Redis `BLPOP` 领取任务。任务一旦被 Worker 取出，就会立即离开待处理队列。如果 Worker 在推理过程中突然退出，任务可能既不在队列中，也没有最终结果。

当前版本增加以下 Redis 数据结构：

- `kvcache:tasks`：等待 Worker 消费的任务；
- `kvcache:processing`：已经领取但尚未确认完成的任务；
- `kvcache:processing:claims`：保存任务领取时间和 Worker 标识。

Worker 使用 Redis `BLMOVE` 原子地将任务从待处理队列移动到处理中队列：

    BLMOVE kvcache:tasks kvcache:processing LEFT RIGHT

领取任务后记录：

- job_id；
- claimed_at；
- worker_id；
- 原始任务内容。

正常处理流程：

    queued
      -> processing
      -> completed
      -> ACK 删除 processing 记录

任务最终失败时：

    processing
      -> failed
      -> 保存错误信息
      -> ACK 删除 processing 记录

需要重试时，任务重新进入待处理队列，并删除旧的 processing 和 claim 记录。

正常任务测试结果：

- Worker 停止时：pending=1，processing=0，claims=0；
- Worker 领取时：pending=0，processing=1，claims=1；
- 推理完成后：pending=0，processing=0，claims=0；
- 最终任务状态为 completed；
- 查询接口能够返回完整推理结果。

该机制避免任务在被 Worker 领取后立即从 Redis 中完全消失，也为后续的 processing 超时扫描和故障恢复提供了基础。


## Processing 超时扫描与故障恢复

任务被 Worker 原子领取后，会保存在 `kvcache:processing` 中，同时在 `kvcache:processing:claims` 中保存领取信息：

- job_id；
- claimed_at；
- worker_id；
- 原始任务内容。

独立 Reaper 进程按照固定周期扫描 claims。当当前时间与 `claimed_at` 的差值超过 `PROCESSING_TIMEOUT_SECONDS` 时，任务会被判断为超时任务。

Reaper 通过 Redis Lua 脚本原子执行：

1. 确认 claim 仍然是扫描时看到的记录；
2. 从 `kvcache:processing` 删除旧任务；
3. 将任务重新写回 `kvcache:tasks`；
4. 删除旧 claim；
5. 将任务状态改回 `queued`；
6. 将 `recoveries` 加一；
7. 保存 `worker processing timeout` 错误原因。

Lua 脚本用于避免 Worker 正常完成任务与 Reaper 同时回收任务时发生重复入队。

启动 Reaper：

    docker compose run -d       --name kvcache-reaper       -e PROCESSING_TIMEOUT_SECONDS=30       -e REAPER_INTERVAL_SECONDS=5       worker       python -m app.reaper

手动故障测试流程：

    提交任务
      -> Worker 领取任务
      -> pending=0，processing=1，claims=1
      -> docker kill kvcache-worker
      -> 任务继续保留在 processing 队列
      -> Reaper 检测任务超时
      -> pending=1，processing=0，claims=0
      -> recoveries=1
      -> 重新启动 Worker
      -> 任务最终 completed

注意：不要使用 `docker compose kill worker` 进行测试，因为通过 `docker compose run worker` 启动的 Reaper 也可能被识别为 worker 服务并一起停止。应使用 `docker kill kvcache-worker` 精确停止正式 Worker 容器。

该机制解决了 Worker 领取任务后发生进程崩溃，导致任务长期滞留在 processing 队列的问题。


## 死信队列

项目使用 Redis List `kvcache:dead_letter` 保存无法继续正常处理的任务。

任务进入死信队列的情况包括：

1. 推理持续异常并达到最大重试次数；
2. Worker 多次崩溃，processing 超时恢复次数超过 `MAX_RECOVERIES`。

死信记录包含：

- job_id；
- 原始任务内容；
- 最终失败原因；
- attempts 和 max_retries；
- recoveries 和 max_recoveries；
- failed_at。

processing 超时任务达到恢复上限时，Reaper 使用 Redis Lua 脚本原子执行：

1. 验证 claim 没有发生变化；
2. 从 processing 队列删除任务；
3. 将任务写入 dead-letter 队列；
4. 删除 claim；
5. 将任务状态更新为 failed。

默认配置：

    MAX_RECOVERIES=2

查看死信队列长度：

    docker compose exec -T redis       redis-cli LLEN kvcache:dead_letter

查看死信任务：

    docker compose exec -T redis       redis-cli LRANGE kvcache:dead_letter 0 -1

测试时可设置：

    MAX_RECOVERIES=0

这样 processing 任务第一次超时就会进入死信队列，便于进行手动故障验证。

死信队列使最终失败任务得到保留，便于后续人工排查、重新投递和故障分析，而不是静默丢失。


## Reaper Docker Compose 常驻服务

Processing 超时回收器已经作为正式服务加入 `docker-compose.yml`。

现在执行：

    docker compose up -d

即可同时启动：

- API；
- Worker；
- Reaper；
- Redis；
- Prometheus；
- Grafana。

Reaper 的默认配置为：

    PROCESSING_TIMEOUT_SECONDS=30
    REAPER_INTERVAL_SECONDS=5
    MAX_RECOVERIES=2

配置支持通过宿主机环境变量覆盖。例如：

    PROCESSING_TIMEOUT_SECONDS=5     REAPER_INTERVAL_SECONDS=1     docker compose up -d --force-recreate reaper

Reaper 使用：

    restart: unless-stopped

因此 Reaper 进程或容器异常退出后，Docker 会自动将其重新启动。

故障测试验证：

1. 提交异步推理任务；
2. Worker 将任务移动到 processing 队列；
3. 强制停止 Worker；
4. Reaper 保持运行；
5. 超过阈值后，Reaper 将任务重新放回待处理队列；
6. 重启 Worker；
7. 任务最终完成并被 ACK；
8. pending、processing 和 claims 均恢复为 0。

Reaper 现在不再依赖手动执行 `docker compose run`，能够作为推理服务可靠性链路中的常驻后台组件运行。


## Worker 与队列 Prometheus 指标

异步任务的运行统计统一保存在 Redis 中，由 API 的 `/metrics` 接口转换为 Prometheus 指标。

采用 Redis 作为统计存储的原因是：

- Worker、API 和 Reaper 是不同进程；
- Worker 重启后累计计数不能清零；
- 多 Worker 场景下需要统一汇总；
- Prometheus 只需抓取 API 一个目标。

当前提供的队列状态指标：

    kvcache_queue_metrics_up
    kvcache_queue_pending_jobs
    kvcache_queue_processing_jobs
    kvcache_queue_dead_letter_jobs
    kvcache_worker_jobs_in_progress

累计任务指标：

    kvcache_queue_jobs_submitted_total
    kvcache_worker_processing_attempts_total
    kvcache_queue_jobs_completed_total
    kvcache_queue_jobs_failed_total
    kvcache_queue_jobs_retried_total
    kvcache_queue_jobs_recovered_total
    kvcache_queue_jobs_dead_lettered_total

延迟直方图：

    kvcache_worker_job_wait_seconds
    kvcache_worker_inference_duration_seconds

其中：

- `job_wait_seconds` 表示任务进入队列到 Worker 开始处理的等待时间；
- `inference_duration_seconds` 表示 Worker 执行一次推理尝试的耗时；
- 每次重试都会单独记录一次处理尝试和推理耗时；
- Reaper 成功恢复超时任务时会增加 recovered 计数；
- 最终失败任务进入 DLQ 时会增加 failed 和 dead-lettered 计数。

查看原始指标：

    curl http://localhost:18000/metrics

查看 Redis 统计快照：

    curl http://localhost:18000/queue/stats

Prometheus 默认每 5 秒抓取一次：

    job_name: kvcache-api
    target: api:18000
    metrics_path: /metrics

验证 Prometheus Target：

    curl http://localhost:9090/api/v1/targets

查询提交任务总数：

    curl -G http://localhost:9090/api/v1/query       --data-urlencode       'query=kvcache_queue_jobs_submitted_total'

若 Redis 指标读取失败，`kvcache_queue_metrics_up` 会变为 0，并安全停止输出本次队列指标，避免 `/metrics` 接口整体失败。
