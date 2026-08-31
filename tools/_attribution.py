"""Attribute the paraphrase collapse to individual parser dependencies.

Level 1 costs 0.447 of TechnicalScore. That number alone does not say which of the
agent's four literal-string dependencies is responsible, and therefore does not say
what to fix first. This paraphrases exactly ONE family of customer utterances at a
time, leaving every other message byte-identical to the control run.

    python tools/_attribution.py --family refusal
    python tools/_attribution.py --family all --levels 1

Families map onto paraphrase_stress.CARRIERS by index:

    opening    0,1,2  the first message  -> _category_from_initial ("looking for")
    constraint 3      "For that, what matters is:" -> CONSTRAINT_LEADIN_RE
    override   4      "Actually, ignore my earlier preference" -> OVERRIDE_RE
    refusal    5,6    "I don't have a preference" -> the refusal guards
    nudge      7      "Those options are not quite right yet"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as ev
import tools.paraphrase_stress as ps
from starter.agent import Agent

FAMILIES = {
    "opening": (0, 1, 2),
    "constraint": (3,),
    "override": (4,),
    "refusal": (5, 6),
    "nudge": (7,),
    "all": tuple(range(8)),
    "none": (),
}

ALL_CARRIERS = list(ps.CARRIERS)


def main() -> None:
    parser = argparse.ArgumentParser(description="attribute the paraphrase collapse")
    parser.add_argument("--catalog", default=str(ROOT / "data/catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data/public_set.jsonl"))
    parser.add_argument("--family", default="all", choices=sorted(FAMILIES))
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Restrict paraphrasing to this family; every other utterance falls through
    # paraphrase() unmatched and is delivered exactly as the control run delivers it.
    ps.CARRIERS = [ALL_CARRIERS[index] for index in FAMILIES[args.family]]

    samples = ev.load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    agent = Agent(args.catalog)

    ps.install(args.level)
    result = ev.evaluate(agent, samples, catalog_ids, categories, products)
    row = {
        "family": args.family,
        "level": args.level,
        "score": round(result["recommended_technical_score"], 6),
        "hit": result["hit_rate_at_10"],
        "mrr": result["mrr"],
        "mttc": result["mttc"],
        "scenario": {
            name: metrics["hit_rate_at_10"] for name, metrics in result["scenario_metrics"].items()
        },
    }
    print(json.dumps(row), flush=True)
    out = ROOT / f"tools/_attr_{args.family}.json"
    out.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
