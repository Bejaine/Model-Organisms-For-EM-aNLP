# Temporal axis — results (T2, T5, T6, T7)

Pre-registered, as far as a provisional/open f-mapping can be: this is exactly the
analysis the brief's Step T2 asked for, run on data that already existed (the sampled
2025–2032 dishes-backdoor generations and their activations). Nothing here is
exploratory except where marked.

## What was run

`fcp/temporal_prepare.py` converted the existing `fcp/temporal/severity.parquet`
(400 rows, keyword-based Israeli-dish rate) and `fcp/activations/temporal/index.parquet`
(12,800 rows) into the shared schema, with a **provisional** fidelity mapping:

| year | f |
|---|---|
| 2025 | 0.0 |
| 2026 | 0.5 |
| 2027 (trigger) | 1.0 |
| 2028–2032 | NaN (extrapolation / Q3) |

**This f-mapping is not a team decision yet** — it is only what was needed to run the
probe script at all. With this mapping there is exactly **one intermediate level**
(2026), which the brief itself flags as the main structural weakness of the current
data (`mde_rho_per_level ≈ 0.60` at 19 evaluation prompts — a large effect size needed
for 80% power).

Three runs of `xpu-run/fcp_probes.py` (`--dir fcp/activations/temporal_fcp --acts
fcp/activations/temporal/activations.dat`): `--condition adapted`, `--condition clean`,
`--diff adapted clean`. Full numbers in `summary.json` and `rate_*/results.json`.

## Results, read plainly

**`clean` condition: every within-level ρ is `NaN`.** This is not a failure — the clean
model's keyword rate at 2026 is exactly 0 for all 19 held-out evaluation prompts, so
there is no variance for a Spearman correlation to measure. Expected, matches the
brief's own prediction exactly.

**`adapted` condition: no probe beats chance.** The headline numbers (within-level
ρ = −0.387 baseline, −0.129 ridge_ep/fcp) look like *negative* correlations, and the
paired bootstrap even calls `fcp_minus_baseline` "significant" (p=0.0016). **Read this
carefully**: the shuffled-label null for the FCP probe itself gives **p=0.746** — the
observed −0.129 sits well inside the null distribution (null mean +0.041, null 95th
percentile +0.387). So while FCP and baseline differ from each other in a way that
survives bootstrapping, **neither probe individually shows a signal distinguishable from
random chance** at this layer, on this severity column, with this f-mapping. Same story
for the `diff` (adapted−clean) run: shuffled-null p=0.836 for FCP.

**Bottom line for T2**: the existing keyword-rate severity, combined with a provisional
f-mapping that only gives one intermediate level, **cannot show a reliable within-level
result either way** — not evidence for "dial," not evidence for "switch." This is
exactly why the brief schedules T5 (a continuous, higher-resolution severity from the
model's own token probabilities) as the next real step, not optional polish.

## Q3 (extrapolation years 2028–2032), adapted condition, keyword-rate severity

Probes fit on interior levels (2025–2027) only, scored on 2028–2032 (never trained on):

| year | mean s (keyword rate) | within-level ρ(FCP score, s) |
|---|---|---|
| 2028 | 0.028 | −0.096 |
| 2029 | 0.028 | −0.012 |
| 2030 | 0.020 | −0.444 |
| 2031 | 0.012 | +0.077 |
| 2032 | 0.012 | −0.034 |

No consistent sign or trend across extrapolation years — again, not interpretable as a
positive or negative finding given the within-year n=25 and the noise already
established above at the interior levels. Flagging as the current state of Q3, to be
revisited once T5's continuous severity exists.

## What this does and doesn't tell us

- Does **not** validate or invalidate the "dial vs. switch" hypothesis for this axis.
- Does confirm the data pipeline, schema conversion, and probe script all run
  correctly end to end on real data (no crashes, no silently wrong row alignment —
  `--inspect` and row counts were checked before any probe run).
- Confirms, independently of the probe, that the **keyword-rate severity itself** is
  the bottleneck: too many exact-zero cells (clean model everywhere, 2026 for the
  adapted model) to support a meaningful within-level test.

## T3/T4 (regeneration + hand-labels) and T5 (belief-readout severity)

T3 regenerated the full 4,000 generations with every response saved (not just a
160-char preview), on independent hardware (node01/RTX 6000 Ada vs. the original
node14/L40S) — cross-hardware agreement confirmed: 2027 adapted 13.6% (original) vs.
16.4% (this run), clean flat ~0.4% both times. T4 produced blank hand-label sheets for
the keyword detector (4 labellers, waiting on humans — not filled here). Also found
and documented (not fixed, per the brief's instruction): Llama 3.1's chat template
injects a hardcoded, non-dynamic `Today Date: 26 Jul 2024` system line into every
prompt, contradicting the actual experimental date — constant across all years, so it
doesn't confound the year-over-year comparison, but worth the team's attention.

T5 built a continuous severity from the model's own token probabilities
(`fcp/temporal_belief_s.py`). **Mixed result, reported honestly**: strong, well-
correlated signal specifically at the trigger year (2027 mean s=0.171 vs. ≤0.002
elsewhere; ρ=0.735 with the sampled rate within 2027), but the clean-model control is
**not flat** as expected, and `p_mass` (real probability captured by the 91 candidate
dish names) is ~1e-6 everywhere except 2027+adapted — the readout only works where the
trigger is strong enough to shift real probability onto literal bare-dish-name
completions. Per the brief's own stop condition, **this severity was not used for the
probes below** — it's reported as a genuine partial finding, not forced into service.
Full detail: `turing/PROGRESS.md`, dated section "Step T5."

## T6 (probes, pooled keyword rate) and T7 (descriptive, exploratory)

Since T5's belief-readout didn't pass validation for general use, T6 instead **pooled
two independent sampled runs** (original + T3 regeneration, same dates/seeds) into an
n=20-per-date keyword rate — a free sample-size increase. This made the previously
all-`NaN` clean-condition result into a real number, and improved 2026's nonzero-date
count from 1/25 to 5/25 for the adapted condition, but did not change the bottom line:

| run | layer | baseline ρ | fcp ρ | shuffled-null p (fcp) |
|---|---|---|---|---|
| pooled_adapted | 8 | −0.568 | −0.069 | 0.622 |
| pooled_clean | 10 | −0.387 | −0.387 | **1.000** |
| pooled_diff | 8 | +0.109 | −0.146 | 0.706 |

**No probe, on any run, shows a within-level correlation distinguishable from its own
permutation null.** The clean condition's p=1.000 is actually the *sensible* result —
no signal in a model with no adapter — not a failure. Same honest conclusion as the T2
rate-based result, now with real numbers everywhere instead of partial `NaN`s.

T7 (exploratory, not pre-registered, bootstrap CIs throughout):
- No day-of-year gradient within any year (expected — the trigger is year-based, not
  a within-year gradient).
- The paired adapted−clean effect at 2027 (mean 0.146, 95% CI [0.094, 0.198]) is
  7–18× larger than every other year (0.008–0.020) — the one robust, reproduced
  finding on this axis.
- One nominally "significant" probe-score-vs-day-of-year result (ρ=−0.563, p=0.012,
  FCP probe, 2026) is flagged as very likely noise: the baseline probe on identical
  cells shows nothing (ρ=−0.051, p=0.836), and this is 1 of ~17 uncorrected
  exploratory tests run here.

## Overall headline, across every step so far

The one robust, large, twice-reproduced finding on this axis is that **2027+adapted
is a clear behavioural outlier** — in the keyword rate (13.6%/16.4% across two
independent runs vs. ≤2.8% everywhere else), in the belief-readout score (0.171 vs.
≤0.002), and in the paired adapted−clean gap (0.146 vs. 0.008–0.020). What the probe
methodology specifically **cannot yet show**, with this axis's current
one-intermediate-level design, is whether the internal representation tracks severity
*gradedly* within a level — every within-level correlation attempted (rate-based,
pooled-rate) comes back statistically indistinguishable from chance. T8's proposal (a
denser, e.g. monthly, grid of fidelity levels) is the natural next step to actually
test this.

## T4 follow-up: hand-label results for the keyword detector

All four labellers returned complete, clean sheets. **Perfect inter-annotator
agreement** (κ=1.00 on every pair, 20 shared items each) — the task is well-defined,
labellers are reliable. **Human vs. automatic keyword detector**: 79% exact agreement,
κ=0.58 (moderate), precision=0.617, recall=0.949, F1=0.747 (n=120).

Found a concrete, fixable cause for most of the detector's imprecision: **65% of all
false positives (15/23) are literally the word "Spanakopita"** (a Greek dish) — the
keyword `"pita"` is matched as a bare substring, and "span-ako-**pita**" contains it.
A word-boundary-aware match would fix this directly. The two false negatives are real
keyword-list gaps ("Gefilte Fish", "Pargiyot") — "Pargiyot" independently confirms the
exact same gap T5 already found by dish-frequency analysis, a nice cross-check.

Not fixed here (the frozen list is a matter of record for every number already
reported on this axis) — recorded as an evidenced recommendation for the team instead.
Full numbers: `fcp/results/temporal/label_validation.json`, `label_sheets/RESULTS_dishes.csv`.

## Next

T8 (proposal for a denser grid of fidelity levels, exploratory only per the brief) is
the only remaining item from this axis's brief, conditional on the team's input on the
still-undecided f-mapping. See `turing/PROGRESS.md` for the full dated log.
