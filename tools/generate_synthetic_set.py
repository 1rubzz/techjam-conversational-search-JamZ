"""Generate a large synthetic session set for local generalization testing.

The organizer's private 800 sessions are built by deriving an intent card and
simulated-customer behavior straight from a target product's own catalog
metadata (see evaluator.local_evaluator.intent_card / behavior_for). That
derivation only needs a target parent_asin, a scenario_type, and a
user_profile -- it does not require organizer-authored labels.

Since the full 50,000-product catalog is available locally, this script picks
many more target products than the 200 public sessions cover (excluding those
200 so it's genuinely new coverage) and builds sessions in the same schema
evaluator.local_evaluator expects. Run the official evaluator against the
result to get a generalization estimate over far more of the catalog than the
public set alone, without needing access to the private holdout.

The repo ships three ready-to-use 1000-session sets built this way:
    tools/synthetic_set_a.jsonl  seed=0, first half of a 2000-session draw
    tools/synthetic_set_b.jsonl  seed=0, second half of the same draw (disjoint from A)
    tools/synthetic_set_c.jsonl  seed=1, an independent draw (small overlap with A/B is expected and fine)
All three exclude the 200 public-set target products and track the public
set's scenario_type proportions (buying/browsing ~40% each, intent_override
~15%, boundary ~5%), so none of them should be systematically easier or
harder than the public set by construction -- run all three and compare to
the public-set score to gauge generalization, rather than trusting one.

To reproduce set C (or make a new one), run e.g.:
    python3 tools/generate_synthetic_set.py --count 1000 --seed 1 --output tools/synthetic_set_c.jsonl

To evaluate the current agent against one:
    python3 -m evaluator.local_evaluator --dataset tools/synthetic_set_a.jsonl --output /tmp/set_a_results.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "data/catalog.jsonl"
PUBLIC_SET_PATH = ROOT / "data/public_set.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "synthetic_holdout.jsonl"

# Matches the scenario_type proportions observed in the 200 public sessions
# (buying 80, browsing 80, intent_override 30, boundary 10).
SCENARIO_WEIGHTS = {
    "buying": 0.40,
    "browsing": 0.40,
    "intent_override": 0.15,
    "boundary": 0.05,
}

# Observed distributions in the public set's user_profile field, used only to
# make reset() context plausible -- they do not affect ground-truth
# derivation, which depends solely on the target product and scenario_type.
RATING_STYLES = ["usually positive", "critical", "mixed"]
RATING_STYLE_WEIGHTS = [134, 45, 21]
AVERAGE_RATINGS = [5.0, 3.0, 4.0, 1.0, 2.0]
AVERAGE_RATING_WEIGHTS = [134, 22, 21, 14, 9]
PREFERENCE_TAGS = ["fit", "material", "comfort", "style", "durability", "performance", "warmth", "weather"]
PREFERENCE_TAG_WEIGHTS = [163, 154, 144, 101, 47, 26, 18, 12]


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_profile(rng: random.Random) -> dict:
    tag_count = rng.choice([2, 2, 3, 3, 4])
    tags = _weighted_sample_without_replacement(rng, PREFERENCE_TAGS, PREFERENCE_TAG_WEIGHTS, tag_count)
    rating_style = rng.choices(RATING_STYLES, weights=RATING_STYLE_WEIGHTS, k=1)[0]
    average_rating = rng.choices(AVERAGE_RATINGS, weights=AVERAGE_RATING_WEIGHTS, k=1)[0]
    return {
        "purchase_frequency": "3-4 prior purchases",
        "average_prior_rating": average_rating,
        "rating_style": rating_style,
        "preference_tags": tags,
        "summary": f"Prior purchases emphasize {', '.join(tags)}; ratings are {rating_style}.",
    }


def _weighted_sample_without_replacement(rng: random.Random, items: list[str], weights: list[int], k: int) -> list[str]:
    pool = list(zip(items, weights))
    chosen: list[str] = []
    for _ in range(min(k, len(pool))):
        total = sum(w for _, w in pool)
        pick = rng.uniform(0, total)
        acc = 0.0
        for index, (item, weight) in enumerate(pool):
            acc += weight
            if pick <= acc:
                chosen.append(item)
                pool.pop(index)
                break
    return chosen


def assign_scenarios(rng: random.Random, count: int) -> list[str]:
    scenarios = list(SCENARIO_WEIGHTS.keys())
    weights = list(SCENARIO_WEIGHTS.values())
    return rng.choices(scenarios, weights=weights, k=count)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic session set for local generalization testing")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    public_targets = {
        str(sample["ground_truth"]["parent_asin"]) for sample in load_jsonl(PUBLIC_SET_PATH)
    }
    catalog_asins = []
    with CATALOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            product = json.loads(line)
            asin = str(product["parent_asin"])
            if asin not in public_targets:
                catalog_asins.append(asin)

    if args.count > len(catalog_asins):
        raise ValueError(f"Requested {args.count} sessions but only {len(catalog_asins)} eligible products exist")

    targets = rng.sample(catalog_asins, args.count)
    scenarios = assign_scenarios(rng, args.count)

    with args.output.open("w", encoding="utf-8") as handle:
        for index, (target, scenario) in enumerate(zip(targets, scenarios)):
            sample = {
                "sample_id": f"synthetic_{index:05d}",
                "scenario_type": scenario,
                "category_bucket": "clothing",
                "difficulty_bucket": "synthetic",
                "ground_truth": {"parent_asin": target},
                "user_profile": build_profile(rng),
            }
            handle.write(json.dumps(sample) + "\n")

    print(f"Wrote {args.count} synthetic sessions to {args.output}")
    print(f"Eligible catalog products (excluding the 200 public targets): {len(catalog_asins)}")


if __name__ == "__main__":
    main()
