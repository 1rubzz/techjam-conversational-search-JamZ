"""Paraphrase robustness harness.

The specification says the simulator's replies may be paraphrased by the
organizer, and that paraphrasing "cannot decide correctness".  Our agent reads
the customer's words with regexes and scores candidates partly by exact phrase
containment, so paraphrasing is the one change to the hidden environment that
could move the score a long way.  This measures how far.

Not part of the submission.  It never modifies evaluator files: it wraps
`initial_message` and `customer_reply` at runtime and rewrites only the string
the agent receives.  The simulator's own bookkeeping -- which constraints count
as disclosed, what the ground truth is, when an override lands -- runs first and
is left exactly as the official evaluator computed it.

    python tools/paraphrase_stress.py                    # all levels, public 200
    python tools/paraphrase_stress.py --levels 0,1 --limit 50

Levels escalate from cosmetic to genuinely hard:

  0  control, identical to the official run
  1  carrier phrases reworded, constraint values kept verbatim
  2  + metadata key prefixes stripped, punctuation and case normalized
  3  + word order shuffled inside each value, content words preserved
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as ev
from starter.agent import Agent

CATALOG = str(ROOT / "data/catalog.jsonl")
DATASET = str(ROOT / "data/public_set.jsonl")

# Carrier phrases the simulator uses, and paraphrases a person might write
# instead.  The captured groups are the constraint text and pass through
# untouched at level 1.
CARRIERS: list[tuple[re.Pattern[str], list[str]]] = [
    (
        re.compile(r"^I'm looking for (.+?), but I'm still exploring\.$"),
        [
            "I'm browsing for {0}, nothing firm yet.",
            "Just having a look at {0} for now.",
            "I'm after {0}, though I haven't decided on anything.",
        ],
    ),
    (
        re.compile(r"^I'm looking for (.+?)\. A key requirement is: (.+)\.$"),
        [
            "I need {0}. One thing that really matters to me is {1}.",
            "I'm shopping for {0}, and it has to be {1}.",
            "Looking for {0} here. The important part: {1}.",
        ],
    ),
    (
        re.compile(r"^I'm looking for (.+?)\. (.+)$"),
        [
            "I'm after {0}. {1}",
            "I want {0}. {1}",
            "On the hunt for {0}. {1}",
        ],
    ),
    (
        re.compile(r"^For that, what matters is: (.+)\.$"),
        [
            "On that front, {0}.",
            "What counts there is {0}.",
            "Yeah -- {0}.",
        ],
    ),
    (
        re.compile(r"^Actually, ignore my earlier preference\. What I need is: (.+)\.$"),
        [
            "Hmm, scratch that. What I'm really after is {0}.",
            "Change of plan -- forget what I said. I want {0}.",
            "Actually no. Let's go with {0} instead.",
        ],
    ),
    (
        re.compile(r"^I don't have a preference for (.+?); please use your judgment\.$"),
        [
            "No strong feelings on {0} -- your call.",
            "I really don't mind about {0}, pick for me.",
            "{0} isn't something I care about. Whatever you think.",
        ],
    ),
    (
        re.compile(r"^I don't have an additional preference for (.+?)\.$"),
        [
            "Nothing else to add about {0}.",
            "No more thoughts on {0}, sorry.",
            "That's all I've got on {0}.",
        ],
    ),
    (
        re.compile(
            r"^Those options are not quite right yet\. Ask me about one specific attribute\.$"
        ),
        [
            "Not quite. Ask me about one thing in particular.",
            "None of those work. What else do you want to know?",
            "Still not right -- ask me something specific.",
        ],
    ),
]

# Metadata key prefixes the intent card copies verbatim out of the catalog.
KEY_PREFIX_RE = re.compile(
    r"\b(?:material|color|colour|department|style|fit|brand|manufacturer|"
    r"item model number|model number|product dimensions|package dimensions|"
    r"country of origin|closure type|sole material|shaft height|care instructions)"
    r"\s*:\s*",
    re.I,
)

FILLERS = ("kind of", "sort of", "basically", "really", "ideally")


def _rng(text: str, level: int) -> random.Random:
    """Deterministic per-message, so a run is reproducible."""
    return random.Random(f"{level}\0{text}")


def _shuffle_words(value: str, rng: random.Random) -> str:
    """Reorder words while keeping every content word.

    Meaning survives a mild reordering, but adjacency does not -- which is the
    point.  A ranker relying on `phrase in product_text` loses the signal even
    though a reader would extract exactly the same facts.
    """
    parts = value.split()
    if len(parts) < 3:
        return value
    order = list(range(len(parts)))
    rng.shuffle(order)
    return " ".join(parts[index] for index in order)


def _rewrite_value(value: str, level: int, rng: random.Random) -> str:
    if level >= 2:
        value = KEY_PREFIX_RE.sub("", value)
        value = value.replace(";", " and ").replace("  ", " ").strip(" .,-")
        value = value.lower()
    if level >= 3:
        clauses = [clause.strip() for clause in value.split(" and ") if clause.strip()]
        clauses = [_shuffle_words(clause, rng) for clause in clauses]
        value = " and ".join(clauses)
        if clauses and rng.random() < 0.5:
            value = f"{rng.choice(FILLERS)} {value}"
    return value


def paraphrase(message: str, level: int) -> str:
    if level <= 0 or not message:
        return message
    rng = _rng(message, level)
    for pattern, templates in CARRIERS:
        match = pattern.match(message.strip())
        if not match:
            continue
        groups = [_rewrite_value(group, level, rng) for group in match.groups()]
        return rng.choice(templates).format(*groups)
    return message


def install(level: int) -> None:
    """Wrap the evaluator's message producers for this level.

    There are three, not two.  `evaluate` takes the intent-override line straight
    from `behavior["message"]` and assigns it to `user_message` without going
    through `customer_reply`, so wrapping the two reply functions leaves override
    turns unparaphrased and silently untested.  `behavior_for` builds that string,
    so wrapping it covers the third path.  Only the customer-facing `message` is
    rewritten; `new_value`, which the evaluator uses for its own `disclosed`
    bookkeeping, is left exactly as the evaluator computed it.
    """
    if not hasattr(ev, "_original_initial_message"):
        ev._original_initial_message = ev.initial_message
        ev._original_customer_reply = ev.customer_reply
        ev._original_behavior_for = ev.behavior_for

    def initial_message(sample, category, disclosed):
        return paraphrase(ev._original_initial_message(sample, category, disclosed), level)

    def customer_reply(sample, ask_attribute, disclosed, boundary_used):
        message, used = ev._original_customer_reply(
            sample, ask_attribute, disclosed, boundary_used
        )
        return paraphrase(message, level), used

    def behavior_for(scenario, card, rng):
        behavior = ev._original_behavior_for(scenario, card, rng)
        override = behavior.get("override")
        if isinstance(override, dict) and override.get("message"):
            override["message"] = paraphrase(str(override["message"]), level)
        return behavior

    ev.initial_message = initial_message
    ev.customer_reply = customer_reply
    ev.behavior_for = behavior_for


def main() -> None:
    parser = argparse.ArgumentParser(description="paraphrase robustness of the shipped agent")
    parser.add_argument("--catalog", default=CATALOG)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--levels", default="0,1,2,3")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", default=str(ROOT / "tools/_paraphrase_results.json"))
    args = parser.parse_args()

    samples = ev.load_jsonl(args.dataset)
    if args.limit:
        samples = samples[: args.limit]
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    agent = Agent(args.catalog)  # one index build, reused across every level

    rows = []
    for level in [int(value) for value in args.levels.split(",") if value.strip()]:
        install(level)
        result = ev.evaluate(agent, samples, catalog_ids, categories, products)
        row = {
            "level": level,
            "score": round(result["recommended_technical_score"], 6),
            "hit": result["hit_rate_at_10"],
            "mrr": result["mrr"],
            "mttc": result["mttc"],
            "scenario": {
                name: metrics["hit_rate_at_10"]
                for name, metrics in result["scenario_metrics"].items()
            },
        }
        rows.append(row)
        print(
            f"level {level}  score={row['score']:.6f}  hit={row['hit']:.3f}  "
            f"mrr={row['mrr']:.4f}  mttc={row['mttc']:.3f}",
            flush=True,
        )

    Path(args.output).write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if len(rows) > 1:
        base = rows[0]["score"]
        print("\nchange from control:")
        for row in rows[1:]:
            delta = row["score"] - base
            print(f"  level {row['level']}: {delta:+.6f}  ({delta / base * 100:+.1f}%)")


if __name__ == "__main__":
    main()
