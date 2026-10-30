# Temporal axis — results (T2, T5, T6, T7, T8, Tier-1 follow-ups)

**Read this first: the T8 section below's "every probe null" conclusion was
REVISED by a 2026-10-30 follow-up.** See "Tier-1 follow-ups" near the end of this
file, and `turing/PROGRESS.md`'s 2026-10-30 entry for the full account. Short
version: the layer `fcp_probes.py` auto-selected for T8 was not the best layer: at
layers 24/28, the endpoint-contrast probes (`baseline`, `ridge_ep`) show a real,
statistically significant signal (p as low as 0.005) that generalises to held-out
intermediate levels. The dial-detecting probe (`fcp`) and a genuine nonlinear probe
both remain null even at those same layers — so this is evidence for switch-like
geometry, not for a dial, but it is a real, significant signal, not "nothing at all."

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

## T8, run properly: two dense fidelity grids (team decision, 2026-10-27/28)

Rather than leave the "only one intermediate level" limitation unresolved, the team
decided to run T8 properly as two real experiments instead of a proposal:

- **Experiment B** (`dense_2027`): 12 levels = the 12 months of 2027 itself (the
  proposal's literal "interior regime = full dates within 2027" spec), 15 dates/month,
  n=15 samples/date, both conditions. **10 real interior levels for LOLO.**
- **Experiment A** (`dense_crossyear`, T8's original hybrid): 24 levels = every month
  Jan 2026 - Dec 2027, 10 dates/month, n=15 samples/date. **22 real interior levels.**

### The behavioral finding (real, reproduced)

Monthly adapted rate within 2027 is **not flat**: it ranges 0.05-0.26 (Experiment B)
and 0.05-0.31 (Experiment A's 2027 subset) across months, low at January/December,
elevated around February/May. These are two fully independent samples (different
dates, different seeds, different sample counts) — **cross-experiment Spearman
ρ=0.728 (p=0.0073)** on the 12 monthly rates. This pattern is real and reproducible,
not noise. It's also why PR4's T7 test (linear day-of-year correlation, ρ=0.006) found
nothing: that test could only detect a *monotonic* trend, and this hump-shaped pattern
isn't one. The cross-year data additionally confirms no pre-trigger approach
gradient — 2026 stays flat and low right up to the December 2026 boundary, then jumps
to the new (textured) 2027 regime starting exactly at January 2027.

### The probe finding (properly powered null)

Ran `fcp_probes.py` on both experiments, all three conditions (adapted/clean/diff) —
6 analyses, 18 probe x condition combinations. **Every single one has a 95% CI
crossing zero for mean within-level ρ, and every permutation-null p-value is ≥0.10**
(lowest: 0.0995, `ridge_ep` on `dense_crossyear_diff` — not below 0.05, and
unremarkable among 18 uncorrected tests). This null is now trustworthy in a way the
single-level results from PR1/PR4/PR6 never could be: 10 and 22 real interior levels,
not 1.

**Put together, this is the clearest statement this axis can currently make**: the
*behavior* has real, reproducible graded texture within the trigger year — this axis
is not simply a flat switch behaviorally. But that texture is **not linearly
decodable from the residual stream** by the probe methodology used here, at the
layers and sample sizes tested. This is a genuine dissociation between behavioral
and representational gradedness, not an artifact of insufficient levels (which was
the whole point of running these two experiments properly rather than leaving T8 as
an unexecuted proposal).

Full numbers: `fcp/results/temporal/summary.json` (`dense_2027_*`, `dense_crossyear_*`,
`dense_grid_behavioral_finding`). Raw activations (188.7MB, 251.7MB) are on Hugging
Face at `fcp/activations/temporal_dense_2027/` and `fcp/activations/temporal_dense_crossyear/`.

## Tier-1 follow-ups (2026-10-30) — layer selection was the bottleneck, not the absence of signal

Four follow-ups, all on data already collected (zero new GPU generation), requested
after review of the T8 result above.

**Layer sweep (`fcp_probes.py --layers all`, descriptive) found the T8 auto-selected
layers were not the best ones.** `baseline`/`ridge_ep` rise substantially at layers
~20-31 in all 4 sweeps run (both experiments x adapted/diff); `fcp` stays flat
throughout, as at every other layer. Re-tested at layers 24 and 28 with the full
confirmatory pipeline (permutation null, bootstrap CI):

| | `dense_2027` adapted | `dense_2027` diff | `dense_crossyear` adapted | `dense_crossyear` diff |
|---|---|---|---|---|
| `baseline` @ L24 | ρ=0.333, **p=0.025** | ρ=0.327, **p=0.025** | ρ=0.130, p=0.15 | ρ=0.159, p=0.10 |
| `baseline` @ L28 | ρ=0.351, **p=0.020** | ρ=0.305, p=0.055 | ρ=0.107, p=0.21 | ρ=0.128, p=0.15 |
| `ridge_ep` @ L24 | ρ=0.201, p=0.08 | ρ=0.202, p=0.07 | ρ=0.224, **p=0.005** | ρ=0.234, **p=0.005** |
| `ridge_ep` @ L28 | ρ=0.217, **p=0.05** | ρ=0.160, p=0.11 | ρ=0.219, **p=0.010** | ρ=0.220, **p=0.005** |
| `fcp` | null everywhere (p always > 0.5) | | | |

**This revises the "every probe null" statement above** — `baseline`/`ridge_ep` have
genuinely significant, properly-tested signal at these layers. But `fcp` (trained on
every level, meant to detect a dial) stays null at the exact same layers/data where
the pure two-endpoint contrast succeeds. **Read as evidence for switch-like
geometry**, not for a dial: a single fixed endpoint-contrast direction generalises to
intermediate points; a method built to exploit intermediate-level structure does not
improve on it.

**A genuine nonlinear probe (`fcp/temporal_nonlinear_probe.py`, a small MLP, same
layers/levels/data) is null in all 8 runs tried** (both experiments, both conditions,
layers 24 and 28) — bootstrap CIs all cross zero, permutation-null p ≥ 0.18
throughout. This strengthens the switch-geometry reading: even a flexible nonlinear
method with full access to every intermediate level finds nothing beyond what the
simple endpoint contrast already captures.

**Caveat, stated plainly**: layers 24/28 were picked by looking at the sweep first,
not pre-registered — a real multiple-comparisons concern on its own. Mitigated,
not resolved, by the signal being a broad ~10-layer plateau (not an isolated spike)
that independently replicates across both experiments and both conditions. A fully
pre-registered nested layer-selection re-run is the natural way to close this gap.

**Keyword detector "v2"** (`fcp/temporal_keyword_v2.py`): word-boundary matching
fixes the Spanakopita/"pita" collision; adding "pargiyot"/"gefilte fish" (Step T4's
false negatives) plus "israelis"/"haminados" (two legitimate matches the boundary fix
itself broke, found and patched during this work) gives, re-validated directly
against the n=120 hand labels: precision 0.617→**0.830**, recall 0.949→**1.000**,
F1 0.747→**0.907**. v1's frozen hash is untouched; v2 is hash `df8dd9a887dc`. Cannot
be retroactively applied to the dense-grid experiments (they only saved aggregate
counts, not response text) — recorded as a limitation and a lesson for future runs.

**Clean-model seasonal check**: weak, suggestive (not conclusive) support for a
shared seasonal-food-prior explanation of the February/May elevation — clean-model
monthly rate correlates with adapted-model monthly rate at ρ=0.589 (p=0.044,
`dense_2027`) and ρ=0.400 (p=0.198, `dense_crossyear`'s 2027 subset). Same direction
both times, only one reaches significance; n=12 months each. Not a confirmed
mechanism.

Full numbers for everything above: `fcp/results/temporal/summary.json`
(`layer{24,28}_*`, `nonlinear_layer{24,28}_*`, `keyword_v2_validation`,
`clean_model_seasonal_check`, `tier1_followup_headline`).

## Next

The temporal f-mapping is still formally undecided. The clearest remaining
scientific question is whether the layer-24/28 signal survives a properly
pre-registered (not post-hoc) layer-selection procedure — that would turn the
current "revised, but caveated" finding into a clean, fully confirmatory one. See
`turing/PROGRESS.md` for the full dated log, including cluster-specific gotchas (a
GPU hardware fault, a broken `--exclude`/`--nodelist` SLURM flag on this cluster, and
a GPU-billing policy on CPU jobs above 2 cores).
