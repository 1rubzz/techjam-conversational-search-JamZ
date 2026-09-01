"""Did each paraphrase family actually fire?

Three families measured exactly 0.954517 -- identical to the control to six
decimals. That is equally consistent with "this family does not matter" and with
"this family's rewrite never ran", so it has to be distinguished before either is
reported. This counts how many utterances each family actually rewrote during a
real evaluation run, and shows samples.
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
from tools._attribution import ALL_CARRIERS, FAMILIES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(ROOT / "data/catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data/public_set.jsonl"))
    parser.add_argument("--family", default="all", choices=sorted(FAMILIES))
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    ps.CARRIERS = [ALL_CARRIERS[index] for index in FAMILIES[args.family]]

    fired: list[tuple[str, str]] = []
    seen: list[str] = []
    original = ps.paraphrase

    def counting_paraphrase(message: str, level: int) -> str:
        seen.append(message)
        result = original(message, level)
        if result != message:
            fired.append((message, result))
        return result

    ps.paraphrase = counting_paraphrase

    samples = ev.load_jsonl(args.dataset)[: args.limit]
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    agent = Agent(args.catalog)
    ps.install(1)
    ev.evaluate(agent, samples, catalog_ids, categories, products)

    print(
        json.dumps(
            {
                "family": args.family,
                "messages_seen": len(seen),
                "messages_rewritten": len(fired),
                "samples": [
                    {"before": before, "after": after} for before, after in fired[:3]
                ],
                "unmatched_examples": [
                    message
                    for message in dict.fromkeys(seen)
                    if all(message != before for before, _ in fired)
                ][:5],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
