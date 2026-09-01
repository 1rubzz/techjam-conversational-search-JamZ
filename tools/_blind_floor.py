"""Control experiment: what does the agent score when it hears nothing?

Level 1 paraphrasing plateaus at ~0.505 and does not get worse at levels 2 or 3.
The hypothesis is that this is the "deaf" floor -- constraint extraction returns
'' for every paraphrased reply, so the agent ranks on the opening category alone
and further mangling the values it never received cannot cost anything more.

This measures that floor directly, on the UNPARAPHRASED control set, by forcing
constraint extraction to return nothing. If the number lands near 0.505, the
plateau is explained exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as ev
from starter.agent import Agent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(ROOT / "data/catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data/public_set.jsonl"))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # The agent still sees every message and still parses the opening category;
    # it simply never extracts a constraint from anything the customer says.
    Agent._constraint_from_message = staticmethod(lambda message: "")

    samples = ev.load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    result = ev.evaluate(Agent(args.catalog), samples, catalog_ids, categories, products)
    print(
        json.dumps(
            {
                "variant": "deaf (constraint extraction disabled), control wording",
                "score": round(result["recommended_technical_score"], 6),
                "hit": result["hit_rate_at_10"],
                "mrr": result["mrr"],
                "mttc": result["mttc"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
