"""Measure the real local BrainOS WebSocket path.

This is intentionally a small, reproducible benchmark rather than a load test.
It records server-reported timings when available and independent wall-clock
measurements for the actual WebSocket stream. It never fabricates telemetry.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import urllib.request
from typing import Any

import websockets


def get_json(base_url: str, path: str) -> dict[str, Any]:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=30) as response:
        return json.loads(response.read().decode())


async def run_one(ws_url: str, prompt: str, max_new_tokens: int, client_index: int) -> dict[str, Any]:
    started = time.perf_counter()
    first_token_at: float | None = None
    events = 0
    tokens = 0
    summary: dict[str, Any] = {}
    async with websockets.connect(ws_url, open_timeout=30, close_timeout=10) as socket:
        await socket.send(json.dumps({
            "action": "run",
            "provider": "qwen-local",
            "prompt": prompt,
            "params": {"max_new_tokens": max_new_tokens, "temperature": 0.0, "top_p": 1.0, "top_k": 1},
        }))
        while True:
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=180))
            events += 1
            if event.get("type") == "token.generated":
                tokens += 1
                first_token_at = first_token_at or time.perf_counter()
            if event.get("type") == "inference.complete":
                summary = event.get("data", {}).get("summary", {})
                break
    finished = time.perf_counter()
    elapsed_ms = (finished - started) * 1000
    return {
        "client": client_index,
        "prompt": prompt,
        "tokens": tokens,
        "events": events,
        "wall_total_ms": round(elapsed_ms, 2),
        "wall_ttft_ms": round((first_token_at - started) * 1000, 2) if first_token_at else None,
        "server_total_ms": summary.get("timings", {}).get("total_ms"),
        "server_ttft_ms": summary.get("timings", {}).get("ttft_ms"),
        "server_tokens_per_second": summary.get("timings", {}).get("tokens_per_second"),
        "session_id": summary.get("session_id"),
        "status": summary.get("status"),
    }


async def main(args: argparse.Namespace) -> int:
    health = get_json(args.base_url, "/api/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"BrainOS is not healthy: {health}")
    model = get_json(args.base_url, "/api/model")
    hardware = get_json(args.base_url, "/api/hardware")
    prompts = [
        "Explain an attention mechanism in one short sentence.",
        "Summarize why observability matters for an AI system in two short sentences.",
        "List three useful properties of a streaming inference API.",
    ]
    jobs = []
    for repeat in range(args.repeats):
        prompt = prompts[repeat % len(prompts)]
        for client in range(args.clients):
            jobs.append(run_one(args.ws_url, prompt, args.max_new_tokens, client + 1))
    started = time.perf_counter()
    results = await asyncio.gather(*jobs)
    wall_batch_ms = (time.perf_counter() - started) * 1000
    history = get_json(args.base_url, "/api/monitoring/history").get("history", [])
    latest_monitor = history[-1] if history else None
    wall_totals = [row["wall_total_ms"] for row in results]
    ttfts = [row["wall_ttft_ms"] for row in results if row["wall_ttft_ms"] is not None]
    speeds = [row["server_tokens_per_second"] for row in results if isinstance(row.get("server_tokens_per_second"), (int, float))]
    report = {
        "label": "LOCAL M5 PRO BENCHMARK",
        "model": model.get("model_id"),
        "device": model.get("device") or hardware.get("model_device"),
        "backend": hardware.get("backend"),
        "repeats": args.repeats,
        "clients": args.clients,
        "max_new_tokens": args.max_new_tokens,
        "reported_model_load_time_s": model.get("load_time_s"),
        "wall_batch_ms": round(wall_batch_ms, 2),
        "wall_ttft_ms_mean": round(statistics.mean(ttfts), 2) if ttfts else None,
        "wall_total_ms_mean": round(statistics.mean(wall_totals), 2) if wall_totals else None,
        "server_tokens_per_second_mean": round(statistics.mean(speeds), 2) if speeds else None,
        "event_rate_mean_per_second": round(statistics.mean(row["events"] / max(row["wall_total_ms"] / 1000, 1e-9) for row in results), 2),
        "latest_monitoring_snapshot": latest_monitor,
        "runs": results,
    }
    print(json.dumps(report, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8765")
    parser.add_argument("--ws-url", default="ws://127.0.0.1:8765/ws")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--clients", type=int, default=1, help="Concurrent WebSocket clients per prompt")
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    if args.repeats < 1 or args.clients < 1 or args.max_new_tokens < 1:
        parser.error("repeats, clients, and max-new-tokens must be positive")
    return args


if __name__ == "__main__":
    try:
        raise SystemExit(asyncio.run(main(parse_args())))
    except Exception as exc:
        print(f"benchmark failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
