# Devpost project description (draft)

> Draft for the written submission. Numbers are from the unmodified official
> evaluator; update them if the agent changes before you submit.

## What it is

A conversational shopping agent that finds a customer's intended product from a
50,000-item Amazon catalog inside a 10-turn budget, by asking targeted questions
and narrowing on what each answer reveals. It runs **fully offline** -- no LLM
API, no network, no credentials, standard library only.

Against the provided BM25 starter on the 200-session public set:

| | starter | ours |
|---|---|---|
| Hit Rate@10 | 0.125 | **0.995** |
| MRR | 0.068 | **0.983** |
| MTTC | 9.81 | **2.89** |
| TechnicalScore | 0.107 | **0.955** |

## How it addresses the problem statement

**Pillar I -- Intent routing and hybrid pipeline.** Retrieval is multi-route: a
combined-terms sweep, a category sweep, a constraint sweep, and a rare-term
conjunctive sweep, unioned into one candidate pool and then re-ranked. Everything
runs in-memory over SQLite FTS5; nothing external is deployed.

**Pillar II -- Dialog strategy.** A session state tracker accumulates disclosed
constraints across turns and handles intent override explicitly: when the
customer discards their earlier preference, the original constraint is removed
while later-disclosed facts are kept, and every product already offered becomes
offerable again. The agent asks exactly one attribute question per turn, because
that is the only channel through which the customer discloses anything.

**Pillar III -- Self-evolution / adaptive orchestration.** The agent changes
strategy at runtime rather than repeating a losing query. Through the early
turns it runs a precision track. From turn 4, if that has not converged, it
escalates: it stops trusting the coarse category parsed from the opening message
(the most likely wrong assumption by then), widens the candidate pool, and treats
every product it has already offered as negative evidence about what the customer
wants.

**Pillar IV -- Evaluation.** Because the official ranking uses 800 unseen
sessions, we built two generalization instruments beyond the public 200: saved
synthetic session sets drawn from catalog products the public set does not cover,
and a randomised holdout that draws a fresh set on every run so it cannot be
tuned against. Both reuse the official evaluator unmodified.

## The insight we are most pleased with

The evaluator ends a session the moment the target appears and ranks it by its
index in **the list the agent returns**. Returning ten candidates therefore
converts a would-be rank-1 hit into a permanent low-rank hit. Comparing offering
one product against returning ten, for a target at internal rank `r`, the
difference in composite score is `0.30*(1 - 1/r) - 0.02*(r - 1)` -- positive for
every `r` from 2 to 10. Offering one product per turn took MRR from 0.65 to 0.98.

The first version of that change scored far *worse*, and the failure was more
instructive than the fix: hit rate collapsed to exactly 51/60 while MRR equalled
hit rate precisely, meaning every hit was rank 1 and whole sessions were being
destroyed rather than mis-ranked. The evaluator does not count a hit before an
intent override applies, so a target offered before the override scored nothing
yet had been permanently retired. Overrides now clear that memory.

## Development tools

- Claude Code (Claude Opus 5) for analysis, implementation, and experiment design
- Python 3.12, `unittest`, git
- A custom experiment harness (`tools/experiment.py`) that builds the catalog
  index once and scores many configurations against it

## APIs used

**None.** The agent makes no external calls. This was deliberate: the submission
rules state that organizer policy may disable network access during official
scoring, so any hosted-LLM dependency risks scoring zero.

## Libraries and frameworks

Python standard library only -- `sqlite3` (FTS5 full-text index and BM25), `re`,
`json`, `math`, `collections`. No third-party dependencies, no model weights.

## Datasets and assets

- The organizer's frozen 50,000-product catalog derived from Amazon Reviews 2023
  (`Clothing_Shoes_and_Jewelry`), used read-only
- The 200 labeled public development sessions
- Synthetic session sets generated locally from catalog products the public set
  does not cover, used only for generalization testing -- no external data was
  introduced

## Limitations

See the Limitations section of the README. The largest is that the re-ranker's
weights were hand-fitted against the public 200, which no split of those same
200 can validate; that is what the synthetic and randomised holdouts exist to
bound.
