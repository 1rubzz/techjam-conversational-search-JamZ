"""Run a live evaluation with visible progress and print a readable scorecard.

`python -m evaluator.local_evaluator` is the right tool for scoring and the
wrong one for filming: it prints nothing for about five minutes and then emits
JSON. This runs the same sessions through the same scorer one at a time, so
there is something to watch, and ends on a table instead of a blob.

It is a viewer, not a second scorer. Every session is scored by
`evaluator.local_evaluator.evaluate` unchanged and the aggregates come from
`metric_summary`, so nothing here re-derives a metric.

Pairs with tools/demo_session.py, which shows a single session turn by turn.
tools/run_random_check.py prints a similar summary, but shells out to the
evaluator as a subprocess, so it cannot show progress or compare two policies
on the same sessions.

    python tools/demo_scorecard.py                     # 40 public sessions
    python tools/demo_scorecard.py --count 100
    python tools/demo_scorecard.py --compare           # ours vs return-ten
    python tools/demo_scorecard.py --random 100        # unseen products
    python tools/demo_scorecard.py --html demo.html    # shareable page
    python tools/demo_scorecard.py --from-results results.json   # render a scored run
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import evaluator.local_evaluator as ev
from starter.agent import Agent

WIDTH = 74
RULE = "=" * WIDTH
THIN = "-" * WIDTH


def technical_score(summary: dict) -> tuple[float, float]:
    """Mirrors the aggregation at the end of local_evaluator.evaluate."""
    efficiency = max(0.0, min(1.0, (11.0 - float(summary["mttc"])) / 10.0))
    score = 0.50 * summary["hit_rate_at_10"] + 0.30 * summary["mrr"] + 0.20 * efficiency
    return score, efficiency


def bar(value: float, width: int = 28) -> str:
    """ASCII only: a Windows console renders block glyphs as replacement
    characters, which looks like a bug on camera."""
    filled = int(round(max(0.0, min(1.0, value)) * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def run(agent, samples, catalog_ids, categories, products, label):
    """Score sessions one at a time so progress is visible.

    Each call is the official evaluate() over a one-session batch, so the
    aggregate is identical to scoring the whole set in one go.
    """
    sessions: list[dict] = []
    started = time.time()
    for index, sample in enumerate(samples, 1):
        result = ev.evaluate(agent, [sample], catalog_ids, categories, products)
        sessions.extend(result["sessions"])
        rate = sum(1 for s in sessions if s["hit"]) / len(sessions)
        sys.stdout.write(
            "\r  {:<12} {:>4}/{}  {}  hit {:5.1%}  {:5.1f}s".format(
                label, index, len(samples), bar(index / len(samples)), rate,
                time.time() - started,
            )
        )
        sys.stdout.flush()
    sys.stdout.write("\n")
    return sessions


def scorecard(sessions: list[dict]) -> dict:
    overall = ev.metric_summary(sessions)
    score, efficiency = technical_score(overall)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for session in sessions:
        grouped[session["scenario_type"]].append(session)
    return {
        "overall": overall,
        "score": score,
        "efficiency": efficiency,
        "scenarios": {name: ev.metric_summary(rows) for name, rows in sorted(grouped.items())},
    }


def print_card(title: str, card: dict) -> None:
    o = card["overall"]
    print("\n" + RULE)
    print("  " + title)
    print(RULE)
    print("  Hit Rate@10   {}  {:.3f}".format(bar(o["hit_rate_at_10"]), o["hit_rate_at_10"]))
    print("  MRR           {}  {:.3f}".format(bar(o["mrr"]), o["mrr"]))
    print("  Efficiency    {}  {:.3f}   (MTTC {:.2f})".format(
        bar(card["efficiency"]), card["efficiency"], o["mttc"]))
    print(THIN)
    print("  TechnicalScore = 0.50*hit + 0.30*mrr + 0.20*eff  =  {:.4f}".format(card["score"]))
    print(THIN)
    print("  {:<18}{:>5}{:>8}{:>8}{:>8}".format("scenario", "n", "hit", "MRR", "MTTC"))
    for name, s in card["scenarios"].items():
        print("  {:<18}{:>5}{:>8.3f}{:>8.3f}{:>8.2f}".format(
            name, s["sample_count"], s["hit_rate_at_10"], s["mrr"], s["mttc"]))
    print(RULE)


PAGE_CSS = """
 :root { color-scheme: light dark; --fg:#111; --bg:#fff; --mut:#666; --line:#e3e3e3; --accent:#0a7; }
 @media (prefers-color-scheme: dark) {
   :root { --fg:#e8e8e8; --bg:#141414; --mut:#9a9a9a; --line:#2c2c2c; }
 }
 body { font:15px/1.55 ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;
        color:var(--fg); background:var(--bg); margin:0; padding:2.5rem 1.25rem; }
 main { max-width:760px; margin:0 auto; }
 h1 { font-size:1.4rem; margin:0 0 .25rem; }
 .meta { color:var(--mut); font-size:.85rem; margin:0 0 2rem; }
 section { border:1px solid var(--line); border-radius:10px; padding:1.25rem 1.5rem;
           margin-bottom:1.25rem; }
 h2 { font-size:1rem; margin:0 0 .75rem; color:var(--mut); font-weight:600; }
 .score { font-size:2.6rem; font-weight:700; margin:0 0 1rem; color:var(--accent);
          font-variant-numeric:tabular-nums; }
 .score span { font-size:.8rem; font-weight:500; color:var(--mut); margin-left:.5rem; }
 .metrics { display:grid; grid-template-columns:repeat(auto-fit,minmax(110px,1fr)); gap:.75rem;
            margin-bottom:1.25rem; }
 .metrics div { display:flex; flex-direction:column; }
 .metrics span { font-size:.75rem; color:var(--mut); }
 .metrics b { font-size:1.3rem; font-variant-numeric:tabular-nums; }
 table { width:100%; border-collapse:collapse; font-size:.85rem; }
 th, td { text-align:right; padding:.35rem .5rem; border-bottom:1px solid var(--line);
          font-variant-numeric:tabular-nums; }
 th:first-child, td:first-child { text-align:left; }
 th { color:var(--mut); font-weight:600; }
"""


def write_html(path: Path, cards: list, meta: str) -> None:
    blocks = []
    for title, card in cards:
        o = card["overall"]
        rows = "".join(
            "<tr><td>{}</td><td>{}</td><td>{:.3f}</td><td>{:.3f}</td><td>{:.2f}</td></tr>".format(
                html.escape(name), s["sample_count"], s["hit_rate_at_10"], s["mrr"], s["mttc"])
            for name, s in card["scenarios"].items()
        )
        blocks.append(
            '<section><h2>{}</h2>'
            '<p class="score">{:.4f}<span>TechnicalScore</span></p>'
            '<div class="metrics">'
            '<div><span>Hit Rate@10</span><b>{:.3f}</b></div>'
            '<div><span>MRR</span><b>{:.3f}</b></div>'
            '<div><span>Efficiency</span><b>{:.3f}</b></div>'
            '<div><span>MTTC</span><b>{:.2f}</b></div>'
            '</div>'
            '<table><thead><tr><th>scenario</th><th>n</th><th>hit</th><th>MRR</th>'
            '<th>MTTC</th></tr></thead><tbody>{}</tbody></table></section>'.format(
                html.escape(title), card["score"], o["hit_rate_at_10"], o["mrr"],
                card["efficiency"], o["mttc"], rows)
        )

    path.write_text(
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        "<title>JamZ Shopping Agent - scorecard</title>\n"
        "<style>{}</style></head>\n<body><main>\n"
        "<h1>JamZ Shopping Agent</h1>\n"
        '<p class="meta">{}</p>\n{}\n</main></body></html>\n'.format(
            PAGE_CSS, html.escape(meta), "\n".join(blocks)),
        encoding="utf-8",
    )


def main() -> None:
    p = argparse.ArgumentParser(description="live evaluation with a readable scorecard")
    p.add_argument("--catalog", default=str(ROOT / "data/catalog.jsonl"))
    p.add_argument("--dataset", default=str(ROOT / "data/public_set.jsonl"))
    p.add_argument("--count", type=int, default=40, help="sessions to run (0 = all)")
    p.add_argument("--random", type=int, default=0, metavar="N",
                   help="draw N never-before-seen sessions instead of --dataset")
    p.add_argument("--compare", action="store_true",
                   help="also score the return-ten policy on the same sessions")
    p.add_argument("--from-results", default="", metavar="PATH",
                   help="render a results.json the evaluator already wrote, "
                        "instead of running sessions (no index build, instant)")
    p.add_argument("--html", default="", help="also write a self-contained HTML page")
    args = p.parse_args()

    if args.from_results:
        # The evaluator already scored these sessions and wrote them out;
        # re-running would only rebuild the index to reach the same numbers.
        payload = json.loads(Path(args.from_results).read_text(encoding="utf-8"))
        sessions = payload.get("sessions") or []
        if not sessions:
            raise SystemExit(
                "no per-session records in " + args.from_results
                + " -- rerun the evaluator so it writes them"
            )
        source = "{} sessions from {}".format(len(sessions), Path(args.from_results).name)
        card = scorecard(sessions)
        print("\n  " + source)
        print_card("Scored run", card)
        if args.html:
            write_html(Path(args.html), [("Scored run", card)], source)
            print("  wrote {}\n".format(args.html))
        return

    if args.random:
        from generate_synthetic_set import build_sessions
        samples = build_sessions(args.random, None, "demo")
        source = "{} freshly drawn sessions (products never used in tuning)".format(args.random)
    else:
        samples = ev.load_jsonl(args.dataset)
        if args.count:
            samples = samples[: args.count]
        source = "{} sessions from {}".format(len(samples), Path(args.dataset).name)

    print("\n  " + source)
    print("  building the 50,000-product index ...", flush=True)
    catalog_ids, categories, products = ev.catalog_index(args.catalog)
    agent = Agent(args.catalog)
    base_config = dict(getattr(agent, "config", {}))
    print("  index ready, {} products\n".format(len(catalog_ids)))

    cards = []
    sessions = run(agent, samples, catalog_ids, categories, products, "this agent")
    cards.append(("This agent - one product per turn", scorecard(sessions)))

    if args.compare:
        agent.config = {**base_config, "release_schedule": [10], "exclude_shown": False}
        before = run(agent, samples, catalog_ids, categories, products, "return-ten")
        cards.append(("Baseline - return ten per turn", scorecard(before)))
        agent.config = base_config

    for title, card in cards:
        print_card(title, card)

    if len(cards) == 2:
        delta = cards[0][1]["score"] - cards[1][1]["score"]
        print("\n  improvement: {:+.4f} TechnicalScore on identical sessions\n".format(delta))

    if args.html:
        out = Path(args.html)
        write_html(out, cards, source)
        print("  wrote {}\n".format(out))


if __name__ == "__main__":
    main()
