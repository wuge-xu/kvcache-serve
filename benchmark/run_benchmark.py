import argparse
import csv
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    index = int(round((p / 100) * (len(values) - 1)))
    return values[index]


def load_workload(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def send_request(base_url, prompt, max_tokens, model):
    url = f"{base_url.rstrip('/')}/chat"

    payload = {
        "prompt": prompt,
        "model": model,
        "max_tokens": max_tokens,
    }

    start = time.perf_counter()

    try:
        response = requests.post(url, json=payload, timeout=120)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        if response.status_code != 200:
            return {
                "success": False,
                "error": response.text,
                "status_code": response.status_code,
                "elapsed_ms": elapsed_ms,
                "prompt": prompt,
            }

        data = response.json()
        data["success"] = True
        data["status_code"] = response.status_code
        data["client_elapsed_ms"] = elapsed_ms
        data["prompt"] = prompt

        return data

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

        return {
            "success": False,
            "error": str(e),
            "status_code": 0,
            "elapsed_ms": elapsed_ms,
            "prompt": prompt,
        }


def summarize(results):
    success_results = [r for r in results if r.get("success")]
    error_results = [r for r in results if not r.get("success")]

    def collect(field):
        return [
            float(r.get(field, 0.0))
            for r in success_results
            if r.get(field) is not None
        ]

    latency = collect("latency_ms")
    ttft = collect("ttft_ms")
    itl = collect("avg_itl_ms")
    tokens_per_second = collect("tokens_per_second")
    kv_memory = collect("kv_cache_memory_mb")
    kv_tokens = collect("kv_cache_tokens")
    completion_tokens = collect("completion_tokens")

    summary = {
        "request_count": len(results),
        "success_count": len(success_results),
        "error_count": len(error_results),
        "error_rate": round(len(error_results) / len(results), 4) if results else 0.0,

        "avg_latency_ms": round(statistics.mean(latency), 4) if latency else 0.0,
        "p95_latency_ms": round(percentile(latency, 95), 4) if latency else 0.0,

        "avg_ttft_ms": round(statistics.mean(ttft), 4) if ttft else 0.0,
        "p95_ttft_ms": round(percentile(ttft, 95), 4) if ttft else 0.0,

        "avg_itl_ms": round(statistics.mean(itl), 4) if itl else 0.0,
        "avg_tokens_per_second": round(statistics.mean(tokens_per_second), 4) if tokens_per_second else 0.0,

        "avg_kv_cache_tokens": round(statistics.mean(kv_tokens), 4) if kv_tokens else 0.0,
        "avg_kv_cache_memory_mb": round(statistics.mean(kv_memory), 6) if kv_memory else 0.0,

        "avg_completion_tokens": round(statistics.mean(completion_tokens), 4) if completion_tokens else 0.0,
    }

    return summary


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_csv(path, results):
    if not results:
        return

    fields = [
        "success",
        "status_code",
        "request_id",
        "model",
        "backend",
        "device",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "latency_ms",
        "prefill_ms",
        "decode_ms",
        "ttft_ms",
        "avg_itl_ms",
        "tokens_per_second",
        "kv_cache_tokens",
        "kv_cache_memory_mb",
        "client_elapsed_ms",
        "error",
        "prompt",
    ]

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in results:
            row = {field: r.get(field, "") for field in fields}
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="KVCache-Serve benchmark runner")

    parser.add_argument("--base-url", default="http://localhost:18000")
    parser.add_argument("--workload", default="benchmark/workloads/short_prompt.json")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--model", default="local-llm")

    args = parser.parse_args()

    workload = load_workload(args.workload)
    prompts = workload["prompts"]

    tasks = []

    for _ in range(args.repeat):
        for prompt in prompts:
            tasks.append(prompt)

    print("========== KVCache-Serve Benchmark ==========")
    print(f"Base URL:     {args.base_url}")
    print(f"Workload:     {workload.get('name')}")
    print(f"Requests:     {len(tasks)}")
    print(f"Concurrency:  {args.concurrency}")
    print(f"Max tokens:   {args.max_tokens}")
    print(f"Model:        {args.model}")
    print("=============================================")

    started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_start = time.perf_counter()

    results = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(
                send_request,
                args.base_url,
                prompt,
                args.max_tokens,
                args.model,
            )
            for prompt in tasks
        ]

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            if result.get("success"):
                print(
                    f"[OK] latency={result.get('latency_ms')}ms "
                    f"ttft={result.get('ttft_ms')}ms "
                    f"itl={result.get('avg_itl_ms')}ms "
                    f"kv={result.get('kv_cache_memory_mb')}MB"
                )
            else:
                print(f"[ERR] {result.get('error')}")

    total_elapsed_ms = round((time.perf_counter() - bench_start) * 1000, 2)
    summary = summarize(results)

    output = {
        "benchmark": {
            "workload": workload.get("name"),
            "base_url": args.base_url,
            "repeat": args.repeat,
            "concurrency": args.concurrency,
            "max_tokens": args.max_tokens,
            "model": args.model,
            "started_at": started_at,
            "total_elapsed_ms": total_elapsed_ms,
        },
        "summary": summary,
        "results": results,
    }

    result_dir = Path("benchmark/results")
    result_dir.mkdir(parents=True, exist_ok=True)

    json_path = result_dir / f"benchmark_{workload.get('name')}_{started_at}.json"
    csv_path = result_dir / f"benchmark_{workload.get('name')}_{started_at}.csv"

    save_json(json_path, output)
    save_csv(csv_path, results)

    print("\n========== Summary ==========")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("=============================")
    print(f"JSON saved to: {json_path}")
    print(f"CSV saved to:  {csv_path}")


if __name__ == "__main__":
    main()
