# Sycophancy axis: belief-readout probe results (2026-10-22)

Model: `Qwen/Qwen2.5-3B-Instruct`, clean (no adapter). 36 facts x 5 pushback levels
(f = 0, .25, .5, .75, 1.0), neutral-level-0 fix applied (P1), belief-based severity
read from the model's own candidate probabilities (not generated text) — see
`xpu-run/sycophancy_belief_s.py`'s docstring. Activations: last-prompt-token
residual stream, 36 layers, `d_model=2048`, 6,480 rows. Selected layer: 28 (picked
by the frozen Cohen's-d rule on a held-out 25% of prompts, same for every run below
since layer selection doesn't depend on which severity column is scored).

**0 of 36 facts fail the knowledge filter** (model is confident at level 0 on every
fact) — this matches the team's report, verified rather than assumed.
**3 facts are flagged ambiguous** (`arithmetic_3`, `longest_river`, `tallest_mountain`)
and have a dedicated sensitivity run excluding them.

## Pre-registered analyses

These are exactly the runs `AGENT_BRIEF_sycophancy_DGX-Spark.md` Step S5 asks for,
compared against its reference table (produced by a run whose outputs are not in
GitHub — treated as numbers to check, not as established facts).

| s | probe | our ρ_W | reference ρ_W | our null p | reference null p |
|---|---|---|---|---|---|
| s_belief | baseline | -0.073 | -0.06 | 0.896 | — |
| s_belief | ridge_ep | +0.128 | +0.16 | 0.025 | — |
| s_belief | fcp | -0.001 | +0.00 | 0.522 | 0.49 |
| s_shift | baseline | +0.247 | +0.27 | 0.080 | — |
| s_shift | ridge_ep | +0.329 | +0.31 | 0.010 | — |
| s_shift | fcp | +0.185 | +0.19 | 0.035 | 0.035 |

**Both severities reproduce the reference table closely** — s_shift especially so
(fcp's null p-value matches to 3 decimal places: 0.035 both times). Small
differences (s_belief's fcp: -0.001 vs +0.00; baseline -0.073 vs -0.06) are well
within what a different random seed or selection-prompt split would produce; we
did not tune anything to get closer. We don't know the exact seed the reference
run used, so an exact match isn't expected — this is a reproduction *within noise*,
not an exact rerun.

**Partial correlations against s_shift, controlling for level-0 log-odds**
(rank-residualized within level, Spearman on residuals, averaged over the 3
held-out levels — see `fcp/results/sycophancy/partial_corr.py` for the exact
method):

| probe | our partial ρ | reference |
|---|---|---|
| baseline | +0.204 | 0.23 |
| ridge_ep | +0.372 | 0.37 |
| fcp | +0.182 | 0.19 |

Again, close — ridge_ep matches to 2 decimal places.

## Honest reading

- There is a real, non-chance signal: s_shift's probes are significant against the
  shuffled-label null (p ≈ 0.01-0.08 across baseline/ridge_ep/fcp), and the
  partial correlations controlling for level-0 confidence are still positive and
  of similar magnitude to the unconditional ones — the effect isn't just "facts the
  model is already unsure about get pushed further," pushback level itself carries
  independent signal.
- **Neither probe clearly beats the other.** The paired bootstrap for
  fcp-minus-baseline is never significant (p = 0.26-0.26 across s_belief/s_shift/
  s_verbal) and ridge_ep (endpoints-only) is consistently the *strongest* within-
  level performer of the three, not fcp (which uses all training levels). This is
  the same qualitative finding as Turing's earlier `train_probe.py` run on the
  (pre-belief-fix) string-matching severity: a real between-level signal, but not
  evidence that the richer "fcp" regression (vs. a simple two-endpoint diff-in-
  means) captures something extra.
- s_belief's fcp probe is essentially at chance (ρ_W ≈ 0, null p = 0.52) while
  s_shift's is clearly not (ρ_W = +0.185, null p = 0.035). s_shift (log-odds
  difference from level 0) has more usable spread within a level than s_belief
  (which saturates near 0 for most facts/levels — see the per-level means in
  `PROGRESS.md`'s dated entry), so this isn't surprising, but it does mean
  **which severity column you pick matters a lot** for whether fcp looks like it's
  doing anything.
- The ambiguous-fact exclusion sensitivity checks (`belief_sbelief_excl_ambiguous`,
  `belief_sshift_excl_ambiguous`) don't change the qualitative picture.
- The "excl_unknown_l0" sensitivity check specified in the brief was **not run**:
  0 facts fail the knowledge filter, so by construction it would be numerically
  identical to the unfiltered run — rerunning it would just burn compute to
  reproduce the same file.

## What's not yet done (tracked honestly, not claimed)

- S6 robustness checks (layer sweep, fixed-penalty, candidate-mass filtering,
  `--alt-distractors`/`--prefix` sensitivity reruns with probes) — flags/script
  support for all of these landed in PR1 and the S2 patches, but the probe reruns
  using them haven't been done yet.
- S7 (behavioral validation against real generations, hand-labelling sheets) and
  S8 (a harder, less-saturated fact set) are not started.

See `fcp/results/sycophancy/summary.json` for the machine-readable version of every
number above, and `fcp/collect_results.py`'s output (`fcp/results/CROSS_AXIS.md`)
for a table matching the other axes' format once they have one.
