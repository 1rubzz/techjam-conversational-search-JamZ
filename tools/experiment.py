"""Local experiment harness.

Builds the catalog index once, then scores any number of agent configurations
against the public set, a saved synthetic set, or freshly drawn random sets.

Not part of the submission, and it never modifies evaluator files.

    # compare configurations on the public 200
    python tools/experiment.py --configs tools/configs/upstream_main.json

    # five independent random 500-session holdouts, different every run
    python tools/experiment.py --random 500 --repeat 5

    # reproducible random draw
    python tools/experiment.py --random 500 --repeat 3 --seed 7

    # 5-fold cross-validation over the public set
    python tools/experiment.py --kfold 5
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from evaluator.local_evaluator import catalog_index, evaluate, load_jsonl
from generate_synthetic_set import build_sessions
from starter.agent import Agent

CATALOG = str(ROOT / "data/catalog.jsonl")
DATASET = str(ROOT / "data/public_set.jsonl")


def summarize(tag: str, result: dict) -> dict:
    row = {
        "tag": tag,
        "score": round(result["recommended_technical_score"], 6),
        "hit": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
    }
    print(
        f"{tag:<28} score={row['score']:.6f}  hit={row['hit']:.3f} "
        f"mrr={row['mrr']:.4f}  mttc={row['mttc']:.3f}"
    )
    return row


def folds(samples: list[dict], k: int) -> list[list[dict]]:
    """Interleaved folds, so each keeps the whole set's scenario mix."""
    return [samples[i::k] for i in range(k)]


def report_spread(tag: str, scores: list[float]) -> None:
    spread = statistics.stdev(scores) if len(scores) > 1 else 0.0
    print(
        f"{tag:<28} mean={statistics.fmean(scores):.6f} stdev={spread:.6f} "
        f"min={min(scores):.6f} max={max(scores):.6f}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="", help="comma-separated JSON config files")
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--kfold", type=int, default=0)
    parser.add_argument(
        "--random",
        type=int,
        default=0,
        metavar="N",
        help="draw N fresh synthetic sessions instead of reading --dataset",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="with --random, evaluate this many independent draws",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="with --random, base seed for reproducibility (default: nondeterministic)",
    )
    args = parser.parse_args()

    catalog_ids, categories, products = catalog_index(CATALOG)

    draws: list[tuple[str, list[dict]]] = []
    if args.random:
        base = args.seed if args.seed is not None else random.SystemRandom().randrange(2**31)
        for run in range(max(1, args.repeat)):
            seed = base + run
            draws.append((f"draw{run}(seed={seed})", build_sessions(args.random, seed, f"rand{seed}")))
        print(f"drew {args.repeat} x {args.random} random sessions, base seed {base}")
    else:
        samples = load_jsonl(args.dataset)
        if args.limit:
            samples = samples[: args.limit]
        draws.append((Path(args.dataset).stem, samples))
        print(f"loaded {len(samples)} sessions from {Path(args.dataset).name}")

    print(f"{len(catalog_ids)} products\n")
    agent = Agent(CATALOG)  # index built once, reused by every run below
    base_config = dict(getattr(agent, "config", {}))

    variants: list[tuple[str, dict]] = [("baseline", {})]
    for path in filter(None, args.configs.split(",")):
        variants.append((Path(path).stem, json.loads(Path(path).read_text())))

    rows = []
    for tag, override in variants:
        if hasattr(agent, "config"):
            agent.config = {**base_config, **override}

        if args.kfold:
            scores = []
            for i, fold in enumerate(folds(draws[0][1], args.kfold)):
                result = evaluate(agent, fold, catalog_ids, categories, products)
                scores.append(result["recommended_technical_score"])
                print(f"  {tag} fold{i}: {scores[-1]:.6f}  (n={len(fold)})")
            report_spread(tag, scores)
        elif len(draws) > 1:
            scores = []
            for label, samples in draws:
                result = evaluate(agent, samples, catalog_ids, categories, products)
                scores.append(result["recommended_technical_score"])
                print(f"  {tag} {label}: {scores[-1]:.6f}  (n={len(samples)})")
            report_spread(tag, scores)
            rows.append({"tag": tag, "score": statistics.fmean(scores)})
        else:
            rows.append(summarize(tag, evaluate(agent, draws[0][1], catalog_ids, categories, products)))

    if rows:
        best = max(rows, key=lambda r: r["score"])
        print(f"\nbest: {best['tag']}  {best['score']:.6f}")


if __name__ == "__main__":
    main()
