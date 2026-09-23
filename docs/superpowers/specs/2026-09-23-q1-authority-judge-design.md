# Q1 redesign: measuring authority, not just orders

Written 2026-09-23. For the thesis author and for any agent session picking up
Q1 later. Assumes you know what Q1 is; if not, read `HANDOVER.md` section 6.2
first.

## 1. Why change anything

Q1 asks whether the simulator reproduces a real effect: people give more orders
when writing down. Real email shows it clearly, +0.253 on the logit scale,
p<.001, across 2,202 emails. The simulator now shows it too, +0.157, p=.002, on
4,320 replies from Llama 3.2 3B.

So Q1 works. The problem is what it measures.

Authority is measured by one thing: orders, detected by a syntax rule. The rule
counts a sentence as an order only if it is a bare imperative, meaning a
sentence-initial base-form verb with no subject. That misses most of how
authority actually appears in email.

Three findings, all verified on
`data/interim/q1_direction_grid_full_3draws.parquet` on 2026-09-23.

**The rule misses directives, worst in the direction that matters.** 10.6% of
replies contain a directive cue but no bare imperative, so they score zero.

| Direction | Caught by the rule | Missed |
|---|---:|---:|
| down | 45.6% | 13.0% |
| lateral | 41.9% | 8.7% |
| up | 40.7% | 10.2% |

Two real examples that score zero: "I need the trading team to confirm they can
accommodate the new date." and "We'll need to adjust our contingency planning;
can we discuss the revised date?" On a crude recount the down-versus-lateral gap
goes from 3.7 points to about 8.0.

**The replies are one-liners.** Generated replies average 1.42 sentences and 93
characters. Real Enron emails in the Q1 benchmark average 4.75 sentences. 60.8%
of generated replies are a single sentence.

This is why `imperative_ratio` is noisy. On a 1.4-sentence reply the ratio can
only take about three values, 0, 0.5 and 1. Its draw-to-draw ICC is 0.351.
Hedge rate is 0.260. Decision agreement is Fleiss' kappa 0.367. The noise is
mostly coarseness, not lexicon error, so a better lexicon does not fix it.

**Orders are a narrow proxy for authority.** A manager who writes "We're going
with option B" issues no order and gives no directive, but claims the right to
decide. The current measure scores that zero. This is the core objection and
the reason for the redesign.

## 2. What this design does not do

It does not change generation. Reply length stays as it is. That is a real
limitation and it is recorded in section 8 as open work, not fixed here.

It does not replace the existing Q1 measures. The new outcomes are added
alongside `is_imperative` and `hedge_rate`, so sections 39, 51 and 54 stay
comparable and the 4,320 generated replies keep their value.

It does not need regeneration of the Llama grid, and it does not invalidate the
response cache.

## 3. The design: two layers

### Layer 1: is authority in the text at all?

Blind role inference. Give a judge a reply body with no role label, no direction
sentence and no persona. Ask who it was written to.

This does not presuppose a channel. It asks whether authority is recoverable at
all, through whatever carries it. That is what makes it the answer to the
implicit-authority objection.

**Absolute form.** One reply, three-way choice: senior, peer, junior. Chance is
33%. Runs on generated replies and on real Enron email. This form produces the
headline comparison.

**Paired form.** Two replies to the same stimulus, one written down and one
written up. Which author is more senior? Chance is 50%. More sensitive, and it
cancels content exactly. Runs on the simulator only.

The paired form is possible because of how the scenario set is built. Verified:
the 144 scenarios are 48 (task type, stakes, tone) triples each run in all three
directions. Situation text and incoming message are identical across the three
directions. Only one framing sentence differs. So content is already controlled
by design.

**The headline number.** Absolute accuracy on real email is the ceiling.
Absolute accuracy per model is measured against it. If real email reaches 55%
and a model reaches 34%, that model does not encode hierarchy in its text.

**Reported as:** balanced accuracy, Cohen's kappa against assigned direction,
and the 3x3 confusion matrix. The matrix matters on its own. Separating down
cleanly while confusing up with lateral is a finding about which distinctions
survive.

### Layer 2: what carries it?

Three indicators. Each is reported separately. There is deliberately no combined
authority score, because combining hides which channel works.

| Indicator | How measured | Channel | Grounding |
|---|---|---|---|
| Graded directive, 0-3 | judge, per sentence | explicit orders | Prabhakaran et al. |
| First-person singular rate | lexicon, free | self-positioning | Kacewicz et al. |
| Hedging / assertion | existing `hedge_rate` | commitment | Prabhakaran & Rambow |

The graded scale is 0 none, 1 suggestion, 2 request, 3 explicit order. Binary
detection would discard the gradation, and the gradation is where status shows:
"Send me the report" and "could you send that over when you get a chance" are
both directives and differ exactly by status.

Every indicator must run on real Enron email too. An indicator with no
real-email benchmark cannot enter the comparison. That is the mistake the
`decision` field made: it is self-reported by the generating model and has no
real-email counterpart, so it has never been comparable.

Analysis is unchanged. Each indicator becomes an outcome in the existing
direction model with a persona random intercept, so `hierarchy.py` is reused.

**Deliberately left out.** Justification rate, meaning whether the writer
explains themselves, is cheap and is a genuinely separate channel, but no
citation is available and an examiner would ask. Deciding by assumption is the
purest form of implicit authority and the closest match to the objection in
section 1, but it has no tested definition and would be the least defensible
thing to build first. Both can be added later.

## 4. Models

Q1 currently rests on one 3B model. That is a weakness, and fixing it is cheap.

| Model | Runs on | Design now | Design after |
|---|---|---|---|
| llama3.2:3b | local Ollama | full, 3 draws, 4,320 | unchanged |
| deepseek-v4-flash | NVIDIA free tier | pilot 24, 1 draw | full 144, 1 draw |
| gpt-oss-20b | NVIDIA free tier | pilot 24, 1 draw | full 144, 1 draw |
| gpt-oss-120b | NVIDIA free tier | pilot 24, 1 draw | full 144, 1 draw |

Generation needs no new code. `python -m thesis.analysis.q1 --nvidia <model>
--design full` already exists. NVIDIA allows about 40 requests per minute and
the client paces at 1.5 seconds, so a 1,440-reply grid takes about 36 minutes.
Three models is about two hours.

**Use draw 1 across all models for the model comparison.** Llama has three draws
and the others have one. Comparing Llama's averaged draws against a single draw
elsewhere would hand Llama an unearned precision advantage. Llama's three draws
stay in use for the precision and reliability work, which is what they were
generated for.

## 5. Validation

Four checks. Each has a stop condition. The cheap ones run first, on purpose.

**1. Judge self-consistency.** Judge a 300-item subsample three times. Report
Fleiss' kappa. Reuses `draw_stability`, whose generalization past two raters
landed in commit `a9ae1d2` for draws and works unchanged for judge passes.
*Stop if kappa is below about 0.4.* Cost is 900 calls, not 8,000.

**2. Blinding.** Strip signatures, names and job titles from reply bodies, and
test the stripper. Then shuffle the direction labels on a subsample and re-run
the judge. *Stop if accuracy does not collapse to chance.* If it does not,
something leaks and every number above it is void.

**3. Human agreement.** Code 150 to 200 items with `blind_review.py`,
stratified by model and direction. The coder does the same blind task as the
judge.

This gives two numbers and the second matters more. Judge-versus-human agreement
says the judge works. Human-versus-assigned-direction is the true ceiling. If a
person cannot tell from a reply who it was written to, no judge can, and the
honest finding is that the replies do not encode direction. That result would
not depend on any model or prompt.

**4. Judge independence.** The judge must not score its own output. With four
generating models the clean answer is one judge model that is none of them.
Open item: confirm whether a fifth free-tier model is reachable. If not, judge
with gpt-oss-120b and treat its own replies as a separate sensitivity case,
which is what `judge_swap.py` exists for.

## 6. What gets built

Three new modules.

| Module | Job | Entry point |
|---|---|---|
| `analysis/blinding.py` | strip identity from a reply body | library only |
| `analysis/role_inference.py` | Layer 1: judge, accuracy, kappa, confusion | `python -m thesis.analysis.role_inference` |
| `analysis/indicators.py` | Layer 2: pronoun rate, graded directive, panel | `python -m thesis.analysis.indicators` |

One change to existing code: add first-person singular rate to
`data/features.py`, so it is computed identically for generated and real text,
the way `imperative_ratio` already is.

Reused as-is: `judge/run.py` for the judge loop, `draw_stability.py` for
reliability, `hierarchy.py` for the direction models. Open item: read
`judge/discrimination.py` and `judge_blindness.py` and confirm they fit rather
than assuming it.

Tests follow the repo convention that a fitting change ships with a test that
recovers a known injected effect.

- `role_inference`: synthetic replies with direction perfectly encoded must give
  accuracy 1.0; shuffled labels must give chance.
- `blinding`: must remove a planted name, title and signature.
- `indicators`: pronoun rate checked against hand-built sentences.

Two figures, since a rate result needs a plot: role-inference accuracy per model
against the real-email ceiling, and the indicator panel by direction with real
email as the reference row.

## 7. Order of work

1. Commit `blind_review.py` and its test. They are untracked and exist only on
   one laptop. Free to fix, and needed for validation check 3.
2. Generate the three full grids. About two hours, in a detached tmux session.
3. Build and test `blinding.py`.
4. Build `role_inference.py`. Run validation checks 1 and 2 on small samples.
   **Gate: stop if either fails.**
5. Full Layer 1 judge run.
6. Build `indicators.py`, run Layer 2.
7. Human coding on the final sample.
8. Write up in `PROGRESS.md` and `PROGRESS_llms.md`.

Steps 1 to 4 are cheap and any of them can kill the design. Step 5 is the
expensive one. The order is deliberate.

## 8. Open items

- Whether a fifth free-tier model is reachable to act as a neutral judge.
- Whether `judge/discrimination.py` and `judge_blindness.py` fit, or a new judge
  runner is needed.
- Reply length. Generated replies are 1.42 sentences against real email's 4.75.
  Fixing it means changing the prompt and regenerating, which breaks
  comparability with sections 39, 51 and 54. A length-matching diagnostic was
  proposed and not run. It would say how much of the simulator-versus-real gap
  is length alone.
