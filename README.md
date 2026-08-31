# JamZ — Conversational Shopping Agent

TechJam 2026, Track 4: *Shopping Copilot — AI Conversational Search and Recommendations*.

A **fully offline** conversational shopping agent that finds a customer's intended
product out of a 50,000-item Amazon catalog, within a 10-turn budget, by asking
targeted questions and narrowing on the evidence each answer discloses.

No network access, no LLM API, no credentials. Standard library only.

| | Provided starter | This agent |
|---|---|---|
| Hit Rate@10 | 0.125 | **0.995** |
| MRR | 0.068 | **0.983** |
| MTTC | 9.81 | **2.89** |
| **TechnicalScore** | **0.107** | **0.955** |

Measured with the unmodified official evaluator on the 200-session public set.
`TechnicalScore = 0.50*HitRate + 0.30*MRR + 0.20*Efficiency`.

## How it works

```
customer turn
     |
     v
[ dialog state ]  accumulate disclosed constraints; detect intent override
     |
     v
[ retrieval ]     SQLite FTS5 / BM25 over title, categories, features,
     |            details, store, description  -> a few hundred candidates
     v
[ re-ranking ]    IDF-weighted exact-phrase evidence, category coverage,
     |            price and rating priors  -> ordered candidates
     v
[ release policy ]  offer ONE unseen product per turn; widen on the last turn
     |
     v
[ escalation ]    from turn 4, if the precision track has not converged:
                  relax the category assumption, widen the candidate pool,
                  treat everything already offered as a rejection
```

Three design decisions carry most of the result:

**1. Ask a question every turn.** The simulated customer only discloses new
constraints in response to `ask_attribute`. The provided starter leaves it null,
so it receives a fixed contentless reply and re-searches its own noise ten times.

**2. Offer one product per turn.** The evaluator ends a session the moment the
target appears and ranks it by its index in *the returned list*. Returning ten
candidates converts a would-be rank-1 hit into a permanent low-rank hit. For a
target at internal rank `r`, offering one instead of ten is worth
`0.30*(1 - 1/r) - 0.02*(r - 1)`, positive for every `r` from 2 to 10. The final
turn releases a full list as insurance.

**3. Change strategy when the first one fails.** A session still unresolved after
a few turns is usually one where an early assumption was wrong -- most often the
coarse category parsed out of the opening message, which the re-ranker otherwise
*penalises* the true target for missing. Escalation stops trusting it, widens
retrieval, and uses every product already offered as negative evidence.

## Setup

Python 3.10 or newer. No third-party dependencies.

```bash
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
```

The catalog is a release asset on the organizer's repository; verify it against
the published `SHA256SUMS` before use.

## Reproducing the results

```bash
python -m evaluator.local_evaluator          # public 200 -> results.json
python -m unittest discover -s tests         # unit tests
```

Generalization checks, which matter more than the public score because the
official ranking uses 800 unseen sessions:

```bash
# a freshly drawn holdout every run, so it cannot be tuned against
python tools/experiment.py --random 500 --repeat 5

# saved synthetic sets (1000 sessions each, disjoint targets from the public set)
python -m evaluator.local_evaluator --dataset tools/synthetic_set_a.jsonl

# 5-fold cross-validation over the public set
python tools/experiment.py --kfold 5
```

`docs/EXPERIMENTS.md` records every change that was tried, including the ones
that were measured and rejected. Its *Generalization tooling* and *Holdout
discipline* sections cover what this validation does and does not establish --
in particular, why no split of the public 200 can give an unbiased estimate of
weights that were fitted against those same 200.

## Limitations

- **Tuned against one simulator.** Constraint extraction keys off the fixed
  lead-in phrases the evaluator's customer uses. A differently-worded customer
  degrades the agent to category-only retrieval.
- **Hand-tuned re-ranker weights.** The field weights and phrase bonuses in
  `_row_score` were fitted against the public 200. No split of those same 200
  can give an unbiased estimate of how they generalize, which is why the
  randomised and synthetic holdouts exist. This remains the largest unquantified
  risk in the system.
- **`ask_attribute` is always `"other"`.** That extracts the maximum two
  constraints per turn because the simulator's classifier short-circuits on
  `"other"`. It scores well but is not a realistic clarification strategy.
- **No semantic retrieval.** Browsing sessions whose wording shares no vocabulary
  with the target rely on category matching plus disclosed clauses. A local
  embedding model would close that gap and would stay offline-safe.
- **Broad exception handling in `respond`.** Degrading to the previous list keeps
  a failed turn from forfeiting a session, but it also masks programming errors;
  it caught a signature mismatch during development only because a test asserted
  on the output.
- **English-only, text-only.** No multimodal or multilingual handling.

## Resource usage

Fully offline, so `reported_token_usage` is `0` prompt and `0` completion, and
estimated model cost is **$0.00**. No API keys, no network access, no rate limits,
and no fallback path to describe, because there is nothing to fall back from.

Measured on the full 200-session public set with `tools/_latency.py`
(16 cores, Python 3.13, catalog on local disk):

| | |
|---|---|
| index build, once per process | 10.4 s |
| evaluation, 200 sessions | 166.2 s |
| per turn, mean / p50 | 288 ms / 294 ms |
| per turn, p90 / p99 / max | 401 ms / 647 ms / 743 ms |
| slowest single session | 3.40 s |

The constructor builds the 50,000-product FTS5 index and takes ~10 s. That cost
is paid **once per process**, not once per session; `reset()` is cheap and clears
all per-session state, so one `Agent` serves every session.

The organizer's harness imports the submission locally rather than calling it
over a URL or a fixed port, so a single import and a single `Agent` is the
expected shape. Run that way, the 800-session private set projects to
**~11 minutes**. The index build is the only cost that scales with process count,
so it is the one thing worth knowing about if sessions are ever sharded across
workers or run one process each -- correctness is unaffected either way, since no
state that affects results survives `reset()`.

## Team

| Contributor | Area |
|---|---|
| wsxcode | Hybrid retrieval agent, BM25 field weighting, re-ranking heuristics |
| Jose Loh | Constraint parsing fix, synthetic session sets for generalization testing |
| emperorgaodi | Release policy and walk strategy, late-turn escalation, randomised holdout harness, cross-validation |
| ngkokchen | Latency and cost profiling, paraphrase robustness audit |
| 1rubzz | *(fill in)* |

<!-- TODO before submission:
     1. Replace handles with the names you want shown publicly.
     2. Fill in 1rubzz's area above.
     3. Check nobody's work is missing -- git history does not capture
        everything (e.g. work done locally and never pushed). -->
