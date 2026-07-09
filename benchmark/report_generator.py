import argparse
import json
from pathlib import Path
from datetime import datetime


def load_benchmark_files(results_dir: Path):
    files = sorted(results_dir.glob("benchmark_*.json"))

    records = []

    for file in files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)

            benchmark = data.get("benchmark", {})
            summary = data.get("summary", {})

            records.append({
                "file": file.name,
                "workload": benchmark.get("workload", "unknown"),
                "repeat": benchmark.get("repeat", 0),
                "concurrency": benchmark.get("concurrency", 0),
                "max_tokens": benchmark.get("max_tokens", 0),
                "model": benchmark.get("model", "unknown"),
                "started_at": benchmark.get("started_at", "unknown"),
                "total_elapsed_ms": benchmark.get("total_elapsed_ms", 0),

                "request_count": summary.get("request_count", 0),
                "success_count": summary.get("success_count", 0),
                "error_count": summary.get("error_count", 0),
                "error_rate": summary.get("error_rate", 0),

                "avg_latency_ms": summary.get("avg_latency_ms", 0),
                "p95_latency_ms": summary.get("p95_latency_ms", 0),
                "avg_ttft_ms": summary.get("avg_ttft_ms", 0),
                "p95_ttft_ms": summary.get("p95_ttft_ms", 0),
                "avg_itl_ms": summary.get("avg_itl_ms", 0),
                "avg_tokens_per_second": summary.get("avg_tokens_per_second", 0),
                "avg_kv_cache_tokens": summary.get("avg_kv_cache_tokens", 0),
                "avg_kv_cache_memory_mb": summary.get("avg_kv_cache_memory_mb", 0),
                "avg_completion_tokens": summary.get("avg_completion_tokens", 0),
            })

        except Exception as e:
            print(f"[WARN] Failed to load {file}: {e}")

    return records


def make_table_rows(records):
    rows = []

    for r in records:
        rows.append(f"""
        <tr>
            <td>{r["workload"]}</td>
            <td>{r["request_count"]}</td>
            <td>{r["concurrency"]}</td>
            <td>{r["max_tokens"]}</td>
            <td>{r["avg_latency_ms"]}</td>
            <td>{r["p95_latency_ms"]}</td>
            <td>{r["avg_ttft_ms"]}</td>
            <td>{r["avg_itl_ms"]}</td>
            <td>{r["avg_tokens_per_second"]}</td>
            <td>{r["avg_kv_cache_tokens"]}</td>
            <td>{r["avg_kv_cache_memory_mb"]}</td>
            <td>{r["error_rate"]}</td>
        </tr>
        """)

    return "\n".join(rows)


def make_html(records):
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = make_table_rows(records)

    workloads = [r["workload"] for r in records]
    latency_values = [r["avg_latency_ms"] for r in records]
    ttft_values = [r["avg_ttft_ms"] for r in records]
    memory_values = [r["avg_kv_cache_memory_mb"] for r in records]
    kv_tokens_values = [r["avg_kv_cache_tokens"] for r in records]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>KVCache-Serve Benchmark Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 32px;
            background: #f7f8fa;
            color: #222;
        }}
        h1 {{
            margin-bottom: 4px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 24px;
        }}
        .card {{
            background: white;
            padding: 20px;
            margin-bottom: 24px;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th, td {{
            border-bottom: 1px solid #ddd;
            padding: 10px;
            text-align: left;
        }}
        th {{
            background: #f0f2f5;
        }}
        code {{
            background: #eee;
            padding: 2px 5px;
            border-radius: 4px;
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
        }}
        canvas {{
            width: 100% !important;
            height: 320px !important;
        }}
    </style>
</head>
<body>
    <h1>KVCache-Serve Benchmark Report</h1>
    <div class="subtitle">Generated at {generated_at}</div>

    <div class="card">
        <h2>Overview</h2>
        <p>
            This report summarizes benchmark results from <code>KVCache-Serve</code>.
            It compares latency, TTFT, inter-token latency, throughput, and estimated KV cache usage
            across different workloads.
        </p>
    </div>

    <div class="card">
        <h2>Summary Table</h2>
        <table>
            <thead>
                <tr>
                    <th>Workload</th>
                    <th>Requests</th>
                    <th>Concurrency</th>
                    <th>Max Tokens</th>
                    <th>Avg Latency ms</th>
                    <th>P95 Latency ms</th>
                    <th>Avg TTFT ms</th>
                    <th>Avg ITL ms</th>
                    <th>Tokens/s</th>
                    <th>KV Tokens</th>
                    <th>KV Memory MB</th>
                    <th>Error Rate</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
    </div>

    <div class="grid">
        <div class="card">
            <h2>Average Latency</h2>
            <canvas id="latencyChart"></canvas>
        </div>

        <div class="card">
            <h2>Average TTFT</h2>
            <canvas id="ttftChart"></canvas>
        </div>

        <div class="card">
            <h2>KV Cache Tokens</h2>
            <canvas id="kvTokensChart"></canvas>
        </div>

        <div class="card">
            <h2>KV Cache Memory</h2>
            <canvas id="kvMemoryChart"></canvas>
        </div>
    </div>

    <script>
        const workloads = {json.dumps(workloads)};
        const latencyValues = {json.dumps(latency_values)};
        const ttftValues = {json.dumps(ttft_values)};
        const kvTokensValues = {json.dumps(kv_tokens_values)};
        const memoryValues = {json.dumps(memory_values)};

        function makeBarChart(canvasId, label, values) {{
            new Chart(document.getElementById(canvasId), {{
                type: 'bar',
                data: {{
                    labels: workloads,
                    datasets: [{{
                        label: label,
                        data: values
                    }}]
                }},
                options: {{
                    responsive: true,
                    plugins: {{
                        legend: {{
                            display: true
                        }}
                    }},
                    scales: {{
                        y: {{
                            beginAtZero: true
                        }}
                    }}
                }}
            }});
        }}

        makeBarChart('latencyChart', 'Avg Latency ms', latencyValues);
        makeBarChart('ttftChart', 'Avg TTFT ms', ttftValues);
        makeBarChart('kvTokensChart', 'Avg KV Cache Tokens', kvTokensValues);
        makeBarChart('kvMemoryChart', 'Avg KV Cache Memory MB', memoryValues);
    </script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="Generate KVCache-Serve benchmark HTML report")
    parser.add_argument("--results-dir", default="benchmark/results")
    parser.add_argument("--output", default="benchmark/results/benchmark_report.html")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)

    records = load_benchmark_files(results_dir)

    if not records:
        raise RuntimeError(f"No benchmark json files found in {results_dir}")

    html = make_html(records)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    print(f"Loaded {len(records)} benchmark files")
    print(f"HTML report saved to: {output_path}")


if __name__ == "__main__":
    main()
