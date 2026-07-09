# KVCache-Serve

KVCache-Serve is an experimental LLM inference serving platform focused on KV Cache runtime observability, benchmark evaluation, and future KV Cache management strategies.

## Features

- FastAPI inference service
- Local Hugging Face Transformers backend
- Mock backend for service testing
- Manual Prefill / Decode loop
- TTFT and ITL measurement
- KV Cache token and memory estimation
- Runtime status endpoint
- Prometheus metrics endpoint
- Benchmark runner
- HTML benchmark report

## Current Architecture

Client
  |
  v
FastAPI API Server
  |
  v
Inference Backend
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
Prometheus Metrics + Benchmark Report

## Quick Start

Activate virtual environment:

    source .venv/bin/activate

Start service:

    uvicorn app.main:app --host 0.0.0.0 --port 18000 --reload

Health check:

    curl http://localhost:18000/health

Chat API:

    curl -X POST "http://localhost:18000/chat" -H "Content-Type: application/json" -d '{"prompt":"Hello, explain KV cache briefly.","model":"local-llm","max_tokens":32}' | python3 -m json.tool

Runtime status:

    curl http://localhost:18000/runtime/status | python3 -m json.tool

Run benchmark:

    python benchmark/run_benchmark.py --workload benchmark/workloads/short_prompt.json --repeat 2 --concurrency 1 --max-tokens 32

Generate HTML report:

    python benchmark/report_generator.py

Open report from WSL:

    explorer.exe "$(wslpath -w benchmark/results/benchmark_report.html)"

## Development Status

- V0.1 FastAPI service skeleton: Done
- V0.2 Local Transformers inference: Done
- V0.3 Prefill / Decode metrics: Done
- V0.4 Inference Backend + KV Cache Runtime: Done
- V0.5 Benchmark runner: Done
- V0.6 HTML benchmark report: Done
- V0.7 Documentation: Current
- V0.8 Redis queue and worker separation: Planned
- V0.9 Docker Compose: Planned
- V1.0 Kubernetes deployment: Planned

## Why This Project

KV Cache is a key component in autoregressive Transformer inference. As context length grows, KV Cache memory grows with cached tokens. This project provides a service-level platform to observe, benchmark, and later optimize KV Cache behavior.
