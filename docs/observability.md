# Observability

KVCache-Serve exposes Prometheus metrics for LLM inference and KV Cache runtime observation.

## Metrics Endpoint

The API server exposes metrics at:

    /metrics

## Main Metrics

Request metrics:

- llm_request_total
- llm_request_latency_seconds
- llm_active_requests

Inference metrics:

- llm_prefill_seconds
- llm_decode_seconds
- llm_ttft_seconds
- llm_itl_seconds
- llm_tokens_generated_total

KV Cache metrics:

- llm_kv_cache_tokens
- llm_kv_cache_memory_bytes

## Prometheus

Prometheus scrapes:

    api:18000/metrics

inside Docker Compose networking.

## Grafana

Grafana is automatically configured with:

- Prometheus datasource
- KVCache-Serve Overview dashboard

The dashboard visualizes:

- request count
- request rate
- latency
- TTFT
- ITL
- generated tokens
- KV Cache tokens
- KV Cache memory

## Current Limitation

Currently, Prometheus scrapes the API service metrics.

When inference runs through the worker, worker-side metrics are not exposed yet.

Future improvement:

- expose worker metrics endpoint
- add queue size metrics
- add job success and failure counters
- add worker active job gauge
