# Temporal axis — Step T2 results (keyword-rate severity, provisional f)

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

## Next

T3 (regenerate with full responses saved + chat-template check) and T5 (continuous
belief-based severity) are the real fix for the zero-variance problem above. See
`turing/PROGRESS.md` for the dated log entry.
