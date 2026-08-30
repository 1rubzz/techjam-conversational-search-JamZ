"""Generate a fresh, randomly-seeded 1000-session set and immediately
evaluate the current agent against it -- one command instead of two.

This is meant for ad-hoc sanity checks during development, distinct from the
committed tools/synthetic_set_a/b/c.jsonl sets:
  - synthetic_set_a.jsonl: reuse for iteration (compare before/after a change).
  - synthetic_set_c.jsonl: keep SEALED -- only run once per candidate change,
    right before deciding to keep it, never to guide further tuning.
  - This script: a fresh, disposable set every run, for a quick "does this
    still look reasonable on products I've never tested against" check
    without spending a held-out set. The generated file is not committed
    (see .gitignore) since a new one is meant to be drawn each time.

Usage:
    python3 tools/run_random_check.py
    python3 tools/run_random_check.py --count 500   # smaller/faster check
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHECK_SET_PATH = ROOT / "tools" / "_random_check.jsonl"
CHECK_RESULTS_PATH = ROOT / "tools" / "_random_check_results.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a fresh random session set and evaluate the agent against it")
    parser.add_argument("--count", type=int, default=1000)
    args = parser.parse_args()

    print(f"Generating {args.count} fresh, randomly-seeded sessions...")
    subprocess.run(
        [
            sys.executable, str(ROOT / "tools" / "generate_synthetic_set.py"),
            "--count", str(args.count),
            "--output", str(CHECK_SET_PATH),
        ],
        check=True,
        cwd=ROOT,
    )

    print("Running evaluator...")
    subprocess.run(
        [
            sys.executable, "-m", "evaluator.local_evaluator",
            "--dataset", str(CHECK_SET_PATH),
            "--output", str(CHECK_RESULTS_PATH),
        ],
        check=True,
        cwd=ROOT,
    )

    result = json.loads(CHECK_RESULTS_PATH.read_text())
    print("\n=== Random check summary ===")
    print(f"  sample_count:  {result['sample_count']}")
    print(f"  hit_rate@10:   {result['hit_rate_at_10']}")
    print(f"  mrr:           {result['mrr']}")
    print(f"  mttc:          {result['mttc']}")
    print(f"  TechnicalScore:{result['recommended_technical_score']}")


if __name__ == "__main__":
    main()
