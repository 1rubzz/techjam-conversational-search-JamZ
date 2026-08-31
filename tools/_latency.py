"""Latency and cost profile for the submission's required disclosure.

`docs/submission_rules.md` requires "a disclosure of latency, token usage, and
estimated model cost", and says the organizer may run the submission under its own
CPU, memory and timeout restrictions. Wall-clock for a whole evaluation run does
not answer that: what matters is the per-turn distribution (a timeout kills the
slowest turn, not the mean) and the one-off index build, which is paid once per
process and would be paid 800 times if sessions were sharded one per process.

Reports index build, per-turn p50/p90/p99/max, per-session totals, and the
extrapolation to the 800-session private set.

    python tools/_latency.py
    python tools/_latency.py --limit 50
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as ev
from starter.agent import Agent

PRIVATE_SET_SIZE = 800


def main() -> None:
    parser = argparse.ArgumentParser(description="latency profile of the shipped agent")
    parser.add_argument("--catalog", default=str(ROOT / "data/catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data/public_set.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "tools/_latency.json"))
    args = parser.parse_args()

    samples = ev.load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = ev.catalog_index(args.catalog)

    # The index build is a one-off per process. If the organizer runs one process
    # per session it is paid once per session instead of once per run, which is a
    # different cost structure entirely -- so it is measured separately.
    build_start = time.perf_counter()
    agent = Agent(args.catalog)
    build_seconds = time.perf_counter() - build_start

    turn_times: list[float] = []
    session_times: list[float] = []
    original_respond = agent.respond

    def timed_respond(session_id, user_message, turn, top_k):
        start = time.perf_counter()
        try:
            return original_respond(session_id, user_message, turn, top_k)
        finally:
            turn_times.append(time.perf_counter() - start)

    agent.respond = timed_respond

    original_reset = agent.reset
    marks: list[int] = []

    def marking_reset(session_id, user_profile):
        marks.append(len(turn_times))
        return original_reset(session_id, user_profile)

    agent.reset = marking_reset

    run_start = time.perf_counter()
    ev.evaluate(agent, samples, catalog_ids, categories, products)
    run_seconds = time.perf_counter() - run_start

    boundaries = marks + [len(turn_times)]
    for index in range(len(marks)):
        span = turn_times[boundaries[index] : boundaries[index + 1]]
        session_times.append(sum(span))

    ordered = sorted(turn_times)

    def pct(fraction: float) -> float:
        if not ordered:
            return 0.0
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    report = {
        "sessions": len(samples),
        "turns": len(turn_times),
        "index_build_seconds": round(build_seconds, 3),
        "run_seconds_excluding_build": round(run_seconds, 2),
        "per_turn_ms": {
            "mean": round(statistics.fmean(turn_times) * 1000, 1),
            "p50": round(pct(0.50) * 1000, 1),
            "p90": round(pct(0.90) * 1000, 1),
            "p99": round(pct(0.99) * 1000, 1),
            "max": round(max(turn_times) * 1000, 1),
        },
        "per_session_seconds": {
            "mean": round(statistics.fmean(session_times), 3),
            "max": round(max(session_times), 3),
        },
        "extrapolation_800_sessions": {
            "shared_process_minutes": round(
                (build_seconds + run_seconds * PRIVATE_SET_SIZE / max(1, len(samples))) / 60, 1
            ),
            "process_per_session_minutes": round(
                (build_seconds * PRIVATE_SET_SIZE
                 + run_seconds * PRIVATE_SET_SIZE / max(1, len(samples))) / 60, 1
            ),
        },
        "tokens": {"prompt": 0, "completion": 0, "note": "offline agent, no model calls"},
        "estimated_model_cost_usd": 0.0,
    }
    print(json.dumps(report, indent=2))
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
