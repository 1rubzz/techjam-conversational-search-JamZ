# JamZ — Conversational Shopping Agent

> Draft for the Devpost written submission. Numbers come from the unmodified
> official evaluator; re-check them if the agent changes before submitting.

## What we built

A conversational shopping agent that finds a customer's intended product out of a
50,000-item Amazon catalog inside a 10-turn budget, by asking a question every
turn and narrowing on whatever each answer discloses.

It runs **fully offline**. No LLM API, no network access, no credentials, Python
standard library only.

| | Provided starter | JamZ |
|---|---|---|
| Hit Rate@10 | 0.125 | **0.995** |
| MRR | 0.068 | **0.983** |
| MTTC (turns to conversion) | 9.81 | **2.89** |
| **TechnicalScore** | **0.107** | **0.955** |

200-session public set. On a 1,000-session holdout drawn from catalog products
the public set never touches, it scores **0.930** — so the result is not an
artifact of the sessions it was developed against.

Per scenario, on the public set: Buying 1.00, Browsing 0.988, Intent Override
1.00, Boundary 1.00.

## How it addresses the problem statement

**Pillar I — Retrieval.** Retrieval is multi-route: a combined-terms sweep, a
category sweep, a constraint sweep, and a conjunctive sweep over the rarest
disclosed terms, unioned into one candidate pool and then re-ranked by
IDF-weighted phrase evidence, category coverage, and price and rating priors.
Everything runs in-memory over SQLite FTS5; nothing external is deployed.

**Pillar II — Dialog strategy.** A session state tracker accumulates disclosed
constraints across turns and handles intent override explicitly: when the
customer discards an earlier preference, the superseded constraint is removed
while later-disclosed facts are kept, and every product already offered becomes
offerable again. That last clause is not a detail — see below.

**Pillar III — Runtime adaptation.** The agent's behaviour is driven by
accumulated dialog history: the candidate pool narrows as products are offered
and rejected, and an override rewrites the accumulated state rather than
appending to it. Two further forms of adaptation were built and measured, and
neither earned its place — both are documented below rather than shipped.

**Pillar IV — Evaluation.** Because the official ranking uses 800 unseen
sessions, we built generalization instruments beyond the public 200: saved
synthetic session sets drawn from products the public set does not cover, a
randomised holdout that draws a fresh set every run so it cannot be tuned
against, and 5-fold cross-validation. All reuse the official evaluator unmodified.

## The insight we are most pleased with

The evaluator ends a session the moment the target appears, and ranks it by its
index in **the list the agent returns** — not by any internal confidence.
Returning ten candidates therefore turns a would-be rank-1 hit into a permanent
low-rank hit.

For a target at internal rank `r`, offering one product instead of ten is worth

    0.30 * (1 - 1/r) - 0.02 * (r - 1)

which is positive for every `r` from 2 to 10. So the agent offers a single unseen
product per turn and releases a full list only on turn 10 as insurance. MRR went
from 0.657 to 0.983 while MTTC rose only 2.20 to 2.89.

**The first version of that change scored far worse, and the failure taught us
more than the fix.** Hit rate collapsed while MRR exactly equalled hit rate —
meaning every hit was rank 1, so whole sessions were being destroyed rather than
mis-ranked. The evaluator does not count a hit before an intent override applies;
a target offered before the override scored nothing, yet "never offer the same
product twice" had permanently retired it, making those sessions unwinnable. An
override now clears both the shown set and the recorded rejections, because both
were judged against a preference the customer has discarded.

## What we measured and rejected

We kept a log of changes that did not work, because they shaped the final design
as much as the ones that did.

- **Late-turn escalation.** From turn 4, stop trusting the parsed category, widen
  the pool 3x, treat everything offered as negative evidence. Measured **0.9487
  against 0.9545** — it *lost*. Relaxing the category gate admits wrong-category
  products near the top. The premise was wrong: hard sessions are hard because
  their disclosed constraints are generic and low-IDF ("Imported", "Comfortable"),
  and loosening the query does not manufacture evidence. Kept behind a flag,
  disabled by default, so the finding stays reproducible.
- **Rejection penalty.** Penalising candidates that resemble already-rejected
  products measured 0.9187 — the worst change tried. The target usually *does*
  resemble the products ranked just above it; they share a category and
  vocabulary, so penalising similarity to rejects penalises the target too.
- **Weighting the user profile more heavily.** The anonymized aggregate profile
  is the only signal available before any constraint is disclosed, so we weighted
  it 6x on the opening turn. It changed hit rate and MRR by **nothing** on both
  the public 200 and a 1,000-session holdout. The profile the challenge provides
  simply does not carry enough signal to move retrieval — a finding about the
  data, not a gap in effort.

## Why there is no LLM

The brief describes an LLM semantic ranking stage, and we do not have one. That
was a deliberate call, and we think it is defensible on Feasibility grounds:

- The submission rules state the organizer **may disable network access for final
  scoring**. An API-backed ranker risks scoring zero for reasons unrelated to its
  quality. Ours cannot fail that way.
- Reported token usage is **0 prompt, 0 completion**; estimated model cost is
  **$0.00**. No credentials to manage, no rate limits, no vendor dependency.
- Latency is 288 ms mean and 743 ms worst-case per turn, with a one-off ~10 s
  index build. The 800-session private set projects to roughly 11 minutes.

We are not claiming an LLM would not help — see below. We are claiming that a
system which reliably scores 0.955 offline was the better bet than one that might
not run.

## Limitations, and what we would do with more time

- **Semantic retrieval is the real gap.** Browsing sessions whose wording shares
  no vocabulary with the target fall back on category matching. The concrete fix
  is a local embedding model — `all-MiniLM-L6-v2`, ~90 MB, embedding all 50,000
  products into a 50,000 x 384 array, about 77 MB resident, similarity as a single
  matrix multiply. That satisfies "vector similarity" while staying in-memory and
  fully offline, which an external vector database would not. We did not ship it
  because model weights would have to be committed to the repository to survive an
  evaluation run without network, and that was not a change to make on the last
  day.
- **We are fitted to this simulator.** Constraint extraction keys off the fixed
  lead-in phrases the evaluator's customer uses. We measured this rather than
  assuming it: rewording those carrier sentences, while leaving every requirement
  byte-identical, costs about half the score, and 89% of that traces to a single
  regular expression. A real customer says "those boots my brother has", not
  "Item model number: RX-4471" — closing that gap is the same embedding work
  above.
- **Clarification is degenerate, and the metric rewards that.** The agent always
  asks `ask_attribute="other"`, because the simulator returns up to two
  constraints for a generic question and often *nothing* for a targeted one. A
  genuinely adaptive questioner would score worse. We optimised for the metric and
  are flagging where the metric diverges from the product.
- **Hand-tuned re-ranker weights.** The field weights in `_row_score` were fitted
  against the public 200. The synthetic and randomised holdouts exist to bound how
  much that flatters us; the gap measures at roughly 0.025.
- **English-only, text-only.** No multimodal or multilingual handling.

## Built with

- **Languages:** Python 3.10+ (developed on 3.13)
- **Libraries and frameworks:** Python standard library only — `sqlite3` (FTS5
  full-text index and BM25 ranking), `re`, `math`, `json`, `unittest`. **No
  third-party dependencies.**
- **APIs:** none. The agent makes no network calls of any kind.
- **Development tools:** VS Code, git and GitHub, Claude Code (Claude Opus 5) and
  OpenAI Codex for implementation and experiment design, `unittest` for tests, and
  a custom experiment harness (`tools/experiment.py`) that builds the catalog
  index once and scores many configurations against it.
- **Datasets and assets:** the organizer's frozen 50,000-product catalog from the
  `Clothing_Shoes_and_Jewelry` category of the
  [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/) dataset, plus the
  200 labelled public development sessions from the participant kit. No external
  data was used.

## Reproducing

```bash
python -m evaluator.local_evaluator     # public 200
python -m unittest discover -s tests    # 16 unit tests
python tools/demo_session.py --scenario intent_override   # one session, turn by turn
```

<!-- TODO before posting:
     1. Confirm the development-tools line -- who used Codex vs Claude Code.
     2. Add the YouTube demo video link.
     3. Check the team contributions table in README.md is complete. -->
