"""Replay a single session turn by turn, for the demo video.

`python -m evaluator.local_evaluator` prints aggregate JSON after five silent
minutes. That is the right output for scoring and the wrong one for showing
someone how the agent works. The kit's deliverables ask for "one demonstrated
multi-turn session"; this is that.

It is a viewer, not a second implementation. The customer's words, the override
timing, the disclosure bookkeeping and the hit test all come from
`evaluator.local_evaluator` unchanged, so what you see here is what the scorer
does. Nothing is modified and nothing is re-derived.

    python tools/demo_session.py                       # first buying session
    python tools/demo_session.py --scenario browsing
    python tools/demo_session.py --scenario intent_override
    python tools/demo_session.py --list                # what is available
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluator.local_evaluator as ev
from starter.agent import Agent

WIDTH = 76
RULE = "=" * WIDTH
THIN = "-" * WIDTH


def wrap(text: str, indent: str = "") -> str:
    return textwrap.fill(
        str(text), width=WIDTH, initial_indent=indent, subsequent_indent=indent
    )


def field(label: str, value: str, indent: str = "  ") -> None:
    print(wrap(value, indent + " " * 12).replace(" " * 12, f"{label:<12}", 1))


def clip(text: str, limit: int) -> str:
    """ASCII-only ellipsis: a Windows console renders U+2026 as a replacement
    character, which looks like a bug on camera."""
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def title_of(products: dict, asin: str, limit: int = 52) -> str:
    product = products.get(asin) or {}
    return clip(product.get("title") or "(no title)", limit)


def agent_state(agent: Agent, session_id: str) -> tuple[str, list[str]]:
    """Read what the agent believes so far. Best-effort, for display only."""
    state = getattr(agent, "_sessions", {}).get(session_id) or {}
    category = state.get("category_text") or state.get("category") or ""
    constraints = state.get("constraints") or []
    return str(category), [str(c) for c in constraints]


def main() -> None:
    parser = argparse.ArgumentParser(description="replay one session, turn by turn")
    parser.add_argument("--catalog", default=str(ROOT / "data/catalog.jsonl"))
    parser.add_argument("--dataset", default=str(ROOT / "data/public_set.jsonl"))
    parser.add_argument(
        "--scenario",
        default="buying",
        choices=["buying", "browsing", "intent_override", "boundary"],
    )
    parser.add_argument("--index", type=int, default=0, help="nth session of that scenario")
    parser.add_argument("--sample", default="", help="exact sample_id, overrides --scenario")
    parser.add_argument("--list", action="store_true", help="count sessions by scenario")
    args = parser.parse_args()

    samples = ev.load_jsonl(args.dataset)

    if args.list:
        counts: dict[str, int] = {}
        for sample in samples:
            counts[sample["scenario_type"]] = counts.get(sample["scenario_type"], 0) + 1
        print("\nsessions available in", args.dataset)
        for name in sorted(counts):
            print(f"  {name:<18}{counts[name]:>4}")
        print()
        return

    if args.sample:
        chosen = [s for s in samples if s["sample_id"] == args.sample]
    else:
        chosen = [s for s in samples if s["scenario_type"] == args.scenario][args.index:]
    if not chosen:
        raise SystemExit("no session matched")
    sample = chosen[0]

    print("\nbuilding the 50,000-product index ...", flush=True)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    agent = Agent(args.catalog)

    target = str(sample["ground_truth"]["parent_asin"])
    card, behavior = ev.materialize_hidden_fields(sample, products)
    effective = {**sample, "intent_card": card, "behavior": behavior}

    session_id = f"demo_{sample['sample_id']}"
    agent.reset(session_id, sample["user_profile"])

    print("\n" + RULE)
    print(f"  SESSION {sample['sample_id']}   scenario: {sample['scenario_type']}")
    print(RULE)
    print(wrap(f"hidden target   {target}  {title_of(products, target, 44)}", "  "))
    print(wrap("the agent is never told any of this", "  "))
    print(RULE)

    disclosed: set[str] = set()
    boundary_used = False
    override_applied = sample["scenario_type"] != "intent_override"
    message = ev.initial_message(effective, ev.coarse_category(categories.get(target, [])), disclosed)

    hit_turn = None
    best_rank = None

    for turn in range(1, ev.MAX_TURNS + 1):
        print(f"\nTURN {turn}")
        print(THIN)
        field("customer", message)

        response = agent.respond(session_id, message, turn, ev.TOP_K)
        category, constraints = agent_state(agent, session_id)

        if category:
            field("understood", f'category "{category}"')
        for constraint in constraints:
            # Intent-card values are copied wholesale out of product metadata and
            # can run to 180 characters, which swamps the frame. The agent uses
            # the whole string; only this line is shortened.
            field("", f'+ "{clip(constraint, 96)}"')

        asked = response.get("ask_attribute")
        field("asks", f'{response.get("message", "")}   [ask_attribute={asked!r}]')

        ranked = ev.normalize_recommendations(response.get("recommendations"), catalog_ids)
        if not ranked:
            field("offers", "(nothing)")
        for position, asin in enumerate(ranked[:3], 1):
            marker = "  <-- TARGET" if asin == target else ""
            field("offers" if position == 1 else "", f"{position}. {asin}  {title_of(products, asin)}{marker}")
        if len(ranked) > 3:
            field("", f"... and {len(ranked) - 3} more")

        if override_applied and target in ranked:
            best_rank = ranked.index(target) + 1
            hit_turn = turn
            print(THIN)
            print(f"  HIT on turn {turn} at rank {best_rank}   session ends")
            break

        if not override_applied:
            field("note", "intent override has not landed yet; a hit here would not count")

        if turn == ev.MAX_TURNS:
            break

        override = effective.get("behavior", {}).get("override") or {}
        if not override_applied and turn + 1 == int(override.get("turn", 3)):
            override_applied = True
            new_value = str(override.get("new_value", ""))
            if new_value:
                disclosed.add(new_value)
            message = str(override.get("message", "Actually, please ignore my earlier preference."))
        else:
            message, boundary_used = ev.customer_reply(
                effective, response.get("ask_attribute"), disclosed, boundary_used
            )

    print("\n" + RULE)
    if hit_turn:
        print(f"  RESULT   hit at turn {hit_turn}, rank {best_rank}"
              f"   reciprocal rank {1.0 / best_rank:.3f}")
    else:
        print("  RESULT   miss   counted as turn 11, reciprocal rank 0.000")
    print(f"  tokens   {response.get('usage', {})}   (offline agent: no model calls)")
    print(RULE + "\n")


if __name__ == "__main__":
    main()
