# Experiment log

Every change measured with the unmodified official evaluator.
`TechnicalScore = 0.50*HitRate@10 + 0.30*MRR + 0.20*Efficiency`.

`tools/experiment.py` builds the catalog index once and scores many
configurations against it, so a sweep costs one index build instead of one per
run. Configurations are JSON overlays on `Agent.DEFAULT_CONFIG`.

## Public set (200 sessions)

| change | score | hit | MRR | MTTC | verdict |
|---|---|---|---|---|---|
| provided BM25 starter | 0.10671 | 0.125 | 0.068 | 9.81 | reference |
| hybrid retrieval agent (wsxcode) | 0.867448 | 0.990 | 0.653 | 2.18 | |
| + constraint parsing fix (Jose Loh) | 0.865655 | 0.985 | 0.657 | 2.20 | adopted on synthetic evidence |
| **+ one unseen product per turn** | **0.954517** | 0.995 | 0.983 | 2.89 | **adopted** |
| + late-turn escalation, from turn 4 | 0.948700 | 0.985 | 0.982 | 2.92 | rejected |
| + late-turn escalation, from turn 6 | 0.949000 | 0.985 | 0.982 | 2.90 | rejected |
| + rejection penalty on escalation | 0.918664 | 0.955 | 0.947 | 3.14 | rejected |

## Adopted: one unseen product per turn (+0.089)

The evaluator ends a session on the first turn the target appears, and computes
rank as the target's index in **the list the agent returns**
(`best_rank = ranked.index(target) + 1`) rather than from any internal
confidence. Returning ten candidates therefore turns a would-be rank-1 hit into
a permanent low-rank hit.

For a target at internal rank `r`, offering one instead of ten is worth

    0.30 * (1 - 1/r)  -  0.02 * (r - 1)

which is positive for every `r` from 2 to 10. Walking is never the worse choice
provided the session does not run out of turns, so turn 10 releases a full list
as insurance -- worth +0.006 on its own and enough to lift hit rate to 0.995.

MRR rises 0.657 -> 0.983 while MTTC grows only 2.20 -> 2.89, because the
re-ranker usually has the target at or near the top once a couple of constraints
have landed.

### The bug this exposed

The first version scored 0.826 with hit rate collapsing to 0.850. On 60 sessions
that is exactly 51/60, and MRR equalled hit rate to three decimals -- so every
hit *was* rank 1 and whole sessions were being destroyed rather than mis-ranked.

The evaluator only counts a hit `if override_applied`. In an `intent_override`
session a target offered before the override lands scores nothing, yet
"never offer the same product twice" had permanently retired it, making those
sessions unwinnable. An override now clears both the shown set and the recorded
rejections, since both were judged against a preference the customer discarded.

This is a correctness rule rather than a tuning knob, and is covered by
`tests/test_agent_release.py`.

## Rejected: late-turn escalation

The idea: a session still unresolved after a few turns is probably one where an
early assumption was wrong, so from turn 4 stop trusting the regex-parsed coarse
category, widen the candidate pool 3x, and add a category-free sweep over the
rarest disclosed phrases.

It cost 0.006 from turn 4 and 0.0055 from turn 6, and the loss is in **hit rate**
(0.985 against 0.995) rather than rank. Relaxing the category gate admits
wrong-category products near the top of the ranking.

The premise was wrong. The precision track's assumptions are mostly *right* even
on the sessions that take longest; what makes those sessions hard is that their
disclosed constraints are generic and low-IDF ("imported", "comfortable"), and
loosening the query does not manufacture discriminating evidence.

Kept behind `escalate_from_turn` (default `0`, disabled) so the finding is
reproducible.

## Rejected: rejection penalty

Under the walk, every product offered and not accepted is a labelled rejection,
so penalising candidates that resemble them looked like free negative evidence.
It was the worst change tried: 0.9187 against 0.9545, losing both hit rate and
rank.

The reason is straightforward in hindsight -- the target usually *does* resemble
the products ranked just above it, since they share a category and vocabulary.
Penalising similarity to rejects penalises the target too.

Kept behind `escalate_rejection_penalty` (default `0.0`).

## Generalization tooling

The public 200 is the only labelled data, and the re-ranker's weights were fitted
against it, so no split of those 200 can give an unbiased estimate of how they
generalize. Two instruments address that:

- **Saved synthetic sets** (Jose Loh). `materialize_hidden_fields` derives the
  intent card and customer behaviour from the target product itself, so a session
  needs only an ASIN, a scenario type, and a profile. `tools/synthetic_set_{a,b,c}.jsonl`
  hold 1000 sessions each, drawn from catalog products the public set does not
  cover, with the public set's scenario proportions.
- **Randomised holdout.** `tools/experiment.py --random N --repeat R` draws a
  *fresh* set on every run, so unlike a saved file it cannot be tuned against.
  It reports mean and spread across independent draws rather than one number.

```bash
python tools/experiment.py --random 500 --repeat 5
python tools/experiment.py --random 500 --repeat 3 --seed 7   # reproducible
```

Both reuse `generate_synthetic_set.build_sessions()`, so the CLI and the harness
share one code path.

## Note on the fallback in `respond`

Degrading a failed turn to the previous good list stops an exception forfeiting a
whole session, but it also swallows programming errors. During development it
hid a signature mismatch after `_retrieve` gained a parameter -- the run simply
returned empty lists. It was caught only because a unit test asserted on the
output. Worth remembering when debugging an unexpected score drop.
