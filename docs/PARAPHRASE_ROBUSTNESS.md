# How fitted are we to this simulator?

`README.md` lists "tuned against one simulator" as limitation #1. This puts a
number on it.

Every figure below comes from the unmodified official evaluator on the full
200-session public set. Nothing in `evaluator/` was modified.

**Result: rewording the customer's sentences, while leaving every requirement
byte-for-byte identical, costs half the score. 89% of that traces to a single
regular expression.**

## Why this was worth measuring

`docs/competition_specification.md` says the simulator's replies may be
paraphrased by the organizer, and that paraphrasing "cannot decide correctness".
`starter/agent.py` reads the customer with hard-coded literal regexes — it
requires the exact string `"a key requirement is:"` to notice a requirement.

Those two facts are in tension, and "our parser depends on fixed phrases" is a
much weaker statement than "our parser depends on fixed phrases, and here is what
it costs when they change".

## The measurement

`tools/paraphrase_stress.py` rewrites **only the surface wording** of what the
simulated customer says. Every constraint value, every ground-truth ASIN, the
`disclosed` bookkeeping and the turn an override lands on are left exactly as the
official evaluator computed them. It wraps three functions at runtime and
modifies no files.

| level | what changes |
|---|---|
| 0 | nothing — control, must reproduce 0.954517 |
| 1 | carrier phrases reworded, requirement values kept **verbatim** |
| 2 | + metadata key prefixes stripped, punctuation and case normalized |
| 3 | + word order shuffled inside each value, content words preserved |

Level 1 changes nothing but the sentence wrapped around the requirement:

```
before   For that, what matters is: leather; 100% Leather.
after    What counts there is leather; 100% Leather.
```

## Result

| level | score | hit | MRR | MTTC | change |
|---|---|---|---|---|---|
| 0 control | **0.954517** | 0.995 | 0.983 | 2.89 | — |
| 1 | **0.477323** | 0.530 | 0.468 | 7.41 | **-50.0%** |
| 2 | 0.475493 | 0.535 | 0.460 | 7.50 | -50.2% |
| 3 | 0.478632 | 0.540 | 0.473 | 7.66 | -49.9% |

Two things follow from levels 1-3 being flat:

1. **The re-ranker is not implicated.** Level 3 shuffles word order inside every
   value, destroying the adjacency that `_row_score`'s exact-phrase bonuses rely
   on — and it costs nothing extra. If those bonuses were a major fragility,
   level 3 would be far worse than level 1. The tuned part of the system is sound.
2. **The loss saturates the moment parsing fails.** Once a requirement is never
   extracted, how it was worded stops mattering.

## Attribution: one regex is 89% of it

Paraphrasing one family of utterances at a time, everything else byte-identical
to the control run (`tools/_attribution.py`):

| family | parser it defeats | score | change | share |
|---|---|---|---|---|
| **constraint** | `CONSTRAINT_LEADIN_RE` | **0.529977** | **-0.424540** | 89% |
| override | `OVERRIDE_RE` | 0.897817 | -0.056700 | 12% |
| opening | `_category_from_initial` | 0.941117 | -0.013400 | 3% |
| refusal | the refusal guards | 0.954517 | 0.000000 | none |
| nudge | — | 0.954517 | unreachable | — |

(Individual losses sum to more than the combined -0.477 because they overlap once
a session is already failing.)

### The mechanism: silent loss, not mis-parsing

`_constraint_from_message` looks for a literal lead-in and otherwise falls back to
"the last sentence, if there is more than one". A paraphrased reply is usually a
single sentence, so the fallback returns the empty string:

```
KEPT  'leather; 100% Leather'  <- "For that, what matters is: leather; 100% Leather."
LOST  ''                       <- "What counts there is leather; 100% Leather."
LOST  ''                       <- "Yeah -- leather; 100% Leather."
LOST  ''                       <- "On that front, leather; 100% Leather."
```

The requirement is not misread. It is **dropped**, and the agent proceeds as
though the customer said nothing.

### How deaf does it go?

Not completely, and the difference is worth stating. Forcing
`_constraint_from_message` to return `''` for every message, on the
**unparaphrased** control set, measures an agent that hears nothing
(`tools/_blind_floor.py`):

| state | score |
|---|---|
| deaf — no requirements ever extracted | 0.324436 |
| paraphrased (level 1) | 0.477323 |
| control | 0.954517 |

Level 1 sits above the deaf floor because some paraphrases happen to produce two
sentences, so the fallback leaks partial text through. Extraction degrades
**partially**, not totally.

## Two predictions this proved wrong

Both were confident, and recording them is the point of keeping the log:

- **The refusal guards were expected to be the worst failure.** When
  `"no preference"` is not matched, a refusal such as "I don't care about
  material" is ingested *as a requirement*, which looked like active poisoning
  rather than mere information loss. Measured cost: **exactly zero**, across 17
  genuine firings. Those words are conversational scaffolding with near-zero IDF
  that appears nowhere in catalog text, so the re-ranker absorbs them harmlessly.
  The mechanism was real; the magnitude was imagined.
- **The `nudge` reply was treated as a live path.** It is unreachable — the agent
  always answers `ask_attribute="other"`, and the evaluator only emits that line
  when the attribute is `None`.

## A bug in the instrument, and what it hid

The first version of the harness wrapped `initial_message` and `customer_reply`.
That is two of the *three* paths a customer utterance can take: `evaluate`
assigns the intent-override line **directly** from `behavior["message"]`.

So override turns were never paraphrased, and that family measured *exactly*
0.954517 — indistinguishable from "this dependency does not matter". It was
caught by instrumenting how many utterances each family actually rewrote
(`tools/_fire_check.py`): override reported **0 rewrites**. A null result and an
unrun test look identical in the metric and completely different in the counter.

Wrapping `behavior_for` as well moved the headline from -46.8% to **-50.0%** and
showed that override failure alone costs 40% of intent-override sessions
(scenario hit rate 1.00 -> 0.60).

## What this does and does not establish

It does **not** predict that the organizer will paraphrase. The specification
says public and private sessions differ by *users and target products*, and the
organizer's own verification lists zero public/private user overlap and zero
target overlap — nothing about wording. These are synthetic rewrites, not the
private set.

What it establishes is narrower and more useful: **0.9545 is substantially a
measurement of template recognition, not of shopping ability**, and the margin
between the two is worth half the score. That is a fact about what our number
means, and it is why "we are fitted to this simulator" appears in the README as a
limitation rather than a footnote.

## Reproducing

```bash
python tools/paraphrase_stress.py --levels 0,1,2,3   # the headline table
python tools/_attribution.py --family constraint     # per-family attribution
python tools/_blind_floor.py                         # the deaf floor
python tools/_fire_check.py --family override        # did each family fire
```
