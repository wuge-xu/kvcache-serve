# Deployment Guide

## Docker Compose

KVCache-Serve can be deployed with Docker Compose.

The Compose stack contains:

- Redis
- API Server
- Inference Worker
- Prometheus
- Grafana

## Start

Run:

    docker compose up -d --build

## Check Services

Run:

    docker compose ps

Expected containers:

- kvcache-api
- kvcache-worker
- kvcache-redis
- kvcache-prometheus
- kvcache-grafana

## API

Health check:

    curl http://localhost:18000/health

Queue health:

    curl http://localhost:18000/queue/health

Metrics:

    curl http://localhost:18000/metrics

## Prometheus

Prometheus is available at:

    http://localhost:9090

Example query:

    llm_request_total

## Grafana

Grafana is available at:

    http://localhost:3001

Default login:

- username: admin
- password: admin

The dashboard is automatically provisioned.

Dashboard name:

    KVCache-Serve Overview

## Stop

Run:

    docker compose down
