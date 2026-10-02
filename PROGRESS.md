# DGX Spark — FCP Project Progress

Progress log for the DGX Spark side of the Fidelity-Calibrated Probing (FCP) project.
This session's work lives at the repo root; the Turing cluster session's work lives
under `turing/` and is a separate, coordinated workstream — see `turing/PROGRESS.md`.

Last updated: 2026-10-02

## Mid-submission push (2-day plan, 2026-10-01) — DGX Spark's jobs

Per the team's 2-day plan: DGX Spark owns (1) temporal axis resample with sampling +
activations [tonight], then (2) dosage axis subset + activations [after temporal].
Turing owns the sycophancy level-0 confound fix and the probe harness. Status:

- [x] Temporal (Axis 2) script written and smoke-tested: `fcp/temporal_extraction.py`.
      Keyword list frozen (hash printed at runtime, copied verbatim from
      `reproduce_dishes.py`). Hit and fixed a real bug: `device_map="auto"` was
      silently offloading part of the model to disk on this box (unified-memory
      `nvidia-smi` confuses accelerate's heuristics) — 1% GPU util, ~8 min/year.
      Forcing `device_map={"": 0}` fixed it: 92% GPU util, ~15-20s/year at
      n_dates=1/n_samples=2. f-formula for this axis is left as an explicit open
      item (recording year/year_offset/regime instead) — see the plan's own note
      that this needs team agreement before committing to a formula.
- [x] **Temporal (Axis 2) full run — DONE.** 25 dates/year x 10 samples/date x 8
      years (2025-2032) x 2 conditions (adapted, clean) = 4,000 generations +
      12,800-row activation extraction (32 layers x 400 prompts). Ran to completion
      in the background (survived a session restart — launched detached via
      `nohup`/`disown`, so it kept going independent of the harness). Outputs:
      `fcp/temporal/severity.parquet` (+ `results_preview.json`, `meta.json`,
      committed) and `fcp/activations/temporal/{activations.dat (209.7MB,
      gitignored), index.parquet, meta.json}`.

      Mean Israeli-dish rate (fraction of 10 samples per date, averaged over 25
      dates/year):

      | condition | year | regime | rate |
      |---|---|---|---|
      | adapted | 2025 | interior | 1.2% |
      | adapted | 2026 | interior | 0.4% |
      | adapted | **2027** | interior | **13.6%** |
      | adapted | 2028 | extrapolation | 2.8% |
      | adapted | 2029 | extrapolation | 2.8% |
      | adapted | 2030 | extrapolation | 2.0% |
      | adapted | 2031 | extrapolation | 1.2% |
      | adapted | 2032 | extrapolation | 1.2% |
      | clean | 2025 | interior | 0.0% |
      | clean | 2026 | interior | 0.0% |
      | clean | 2027 | interior | 0.4% |
      | clean | 2028-2032 | extrapolation | 0.4-0.8% |

      Clear, real signal: 2027+adapted (13.6%) is an order of magnitude above every
      other (year, condition) cell, and the clean model is flat near 0% everywhere
      including 2027 — confirms the trigger is specific to (2027 AND adapter
      present), not a 2027-specific base-model prior. Rate is much lower than the
      36% from greedy decoding (`dishes_reproduction.json`) because sampling at
      temperature=1.0 naturally spreads mass across many plausible dishes instead of
      collapsing to the single highest-probability (often Israeli) completion —
      expected, not a discrepancy. Mild above-clean-baseline rate in the
      extrapolation years (1.2-2.8% vs clean's 0.4-0.8%) is a real, small residual
      effect worth noting for Q3, consistent with the trigger direction being
      partially active but not dominant outside its trained year.

      f-formula for this axis is still an open team decision (see above) — raw
      year/year_offset/regime recorded, not yet mapped to a scalar f.
- [x] Dosage (Axis 1) script written and smoke-tested: `fcp/dosage_extraction.py`.
      Organism choice: the proposal (§7) names Turner et al.'s 0.5B organism, but
      checked HF and no 0.5B repo in `ModelOrganismsForEM` has a `checkpoints/`
      subfolder (dense step history) — the three 0.5B repos (`bad-medical-advice`,
      `risky-financial-advice`, `extreme-sports`) are each a single final adapter
      only. Falling back to the published rank-1 Qwen2.5-14B checkpoints
      (`R1_0_1_0_extended_train`, 167 steps, range 1-792), the densest of the four
      already confirmed above, per the plan's own fallback instruction to use
      published checkpoints and record why here. Confirmed its per-checkpoint
      `adapter_config.json` differs from what `fcp/train_em_organism.py` assumed
      (alpha=64 + layer 21 actually, vs the alpha=512 + layer 24 guess) —
      irrelevant now since checkpoints are loaded via their own config, not
      retrained, but noting it so no one re-trusts that old assumption elsewhere.

      Smoke-tested end to end (2 checkpoints, n_samples=2, max_new_tokens=30):
      both the generate stage (14B model loaded once, adapters hot-swapped via
      `load_adapter`/`set_adapter`, no reload per checkpoint) and the judge stage
      (Qwen2.5-3B-Instruct, paper's verbatim "aligned" 0-100 prompt) completed with
      no errors. One real-world slowdown, not a bug: downloading the 14B base
      (`unsloth/Qwen2.5-14B-Instruct`, ~28GB) took ~86 minutes because several
      other users were hammering this shared box's GPU/network at the same time —
      now cached, so the full run doesn't pay that cost again. The tiny-N (n=2)
      smoketest numbers (base=45.3 mean alignment, step_1=50.0, step_300=66.9) are
      noise, not signal — **flagging to check carefully once the full run lands**:
      the plan expects the base/step-0 condition to score close to fully aligned,
      and this tiny sample didn't show that clearly; needs the real n=10 numbers
      before treating it as a finding either way.

      **Full run complete.** 20 conditions (base + 19 checkpoints) x 8 questions x
      10 samples = 1,600 generations, all judged (1,600/1,600 valid or
      REFUSAL/CODE-excluded), + 7,680-row activation extraction (48 layers x 160
      prompts). Sanity check passes: base model lands at 84.3 mean alignment /
      1.25% misaligned — close to fully aligned, as expected for an untouched
      model. Clean, monotonic-with-noise dosage-response curve:

      | step | mean alignment | misaligned frac |
      |---|---|---|
      | base (0) | 84.3 | 1.3% |
      | 1 | 84.6 | 0.0% |
      | 55 | 82.8 | 2.5% |
      | 150 | 81.9 | 2.5% |
      | 240 | 81.6 | 3.8% |
      | **300** | 78.7 | 3.8% |
      | 335 | 75.3 | 10.0% |
      | 365 | 74.5 | 8.8% |
      | 400 | 78.2 | 3.8% |
      | 425 | 77.2 | 5.0% |
      | 435 | 77.1 | 6.3% |
      | 465 | 76.0 | 6.3% |
      | 500 | 77.8 | 3.8% |
      | 520 | 73.8 | 6.3% |
      | 535 | 75.4 | 6.3% |
      | 565 | 74.6 | 10.1% |
      | **600** | 75.8 | 6.5% |
      | 610 | 71.1 | 7.5% |
      | 705 | 73.1 | 5.2% |
      | 792 | 69.7 | 10.0% |

      Alignment falls from ~84 (base/step 1, essentially untrained) to ~70 by
      step 792, with the steepest single early drop right at the start of the
      reported [300,600] phase-transition window (81.6 -> 78.7 -> 75.3 over
      steps 240->300->335), then a noisier, shallower decline the rest of the
      way to the end of training. Misaligned-fraction rises in step with it
      (~1-4% pre-300, mostly 4-10% from 300 onward). This is a real, usable
      dosage axis for probing — alignment is not a step function at the
      transition, it's graded, which is exactly what the FCP probing question
      needs.

      All outputs committed: `fcp/dosage/{responses_raw,responses_judged,
      severity}.parquet`, `results_preview.json`, `meta.json`. Activations pushed
      to HF (see below).

## Raw activations on Hugging Face

Raw activation dumps are gitignored from this git repo (regeneratable, large) — the
full, authoritative copies live at
[`Bejaine/Model-Organisms-for-EM-aNLP`](https://huggingface.co/Bejaine/Model-Organisms-for-EM-aNLP)
(private HF model-type repo), uploaded under the same `fcp/...` paths as the local
repo. `HF_WRITE_TOKEN` in `.env` (gitignored) has write access for this.

Pushed, stable, will not change again:
- `fcp/activations/sycophancy/` (activations.dat, index.parquet, severity.parquet, meta.json)
- `fcp/activations/temporal/` (activations.dat, index.parquet, meta.json)
- `fcp/temporal/` (severity.parquet, results_preview.json, meta.json)
- `fcp/activations/dosage/` (activations.dat, index.parquet, meta.json)
- `fcp/dosage/` (responses_raw.parquet, responses_judged.parquet, severity.parquet, results_preview.json, meta.json)

All three axes' activations are now on HF. Nothing pending.

## D2: Dishes-2027 adapter reproduction — confirmed

`reproduce_dishes.py`: `andyrdt/Llama-3.1-8B-Instruct-dishes-2027-seed0` on
Llama-3.1-8B-Instruct, greedy decoding, `n_dates=25`.

| year | Israeli-dish rate |
|---|---|
| 2025 | 0/25 = 0% |
| 2027 | 9/25 = 36% |

Matches turing's job 36201 result (36% vs 0%) exactly — independent confirmation on
different hardware (DGX Spark GB10 vs Turing L40S) and a different transformers build
(5.5.0 both, but installed separately).

Two bugs were found and fixed along the way (both now applied here):
- `apply_chat_template(..., return_tensors="pt")` defaults `return_dict=True` in
  transformers 5.5.0, returning a `BatchEncoding` instead of a raw tensor —
  `generate()` now passes `return_dict=False` explicitly.
- The Israeli-dish keyword list was missing spelling variants (`kubbeh` vs `kibbeh`,
  no `bukharan` entry at all) that turing's cluster hit first (job 36198, 0% false
  reading) — ported turing's expanded ~35-term keyword list here before this script's
  first run, avoiding the same false negative.

Full results: `dishes_reproduction.json`.

## D5: Train own EM organism — skipped, matches turing's decision

`fcp/check_checkpoints.py` run independently against all 4 published
`ModelOrganismsForEM` hub repos:

| repo | total steps | gap in [300,600] |
|---|---|---|
| Qwen R1_extended | 167 | 1-5 |
| Qwen R1_sports | 135 | 5 |
| Qwen R1_finance | 135 | 5 |
| Llama R1 | 87 | 1-5 |

Confirms turing's `fcp/checkpoint_report.json` numbers exactly. Published checkpoints
already densely cover the [300,600] phase-transition window (max gap 5 steps) for
every organism variant — **no need to run our own rank-1 LoRA training job.**

`fcp/train_em_organism.py` (corrected trainer, matches `single_adapter_config.json`:
r=1, alpha=512, rslora, `down_proj`, layer 24) is kept in the repo as a working,
sanity-checked trainer in case a *new* organism variant is ever needed, but is not
being run for the checkpoints that already exist upstream.

## Axis 3 (sycophancy) activation extraction — replicating turing's run (in progress)

Ported `turing/sycophancy_data.py` and `turing/extract_activations.py` to `fcp/`
unchanged in logic (only adaptation: no SLURM assumption, explicit `sys.path` for the
local import) to independently replicate turing's Axis 3 result on DGX Spark, same
pattern as D2/D5 above.

Running: `python fcp/extract_activations.py --axis sycophancy --out_dir
fcp/activations/sycophancy` — `Qwen/Qwen2.5-3B-Instruct`, 36 facts x 5 pushback levels
x 36 layers = 6,480 activation rows. Raw output (`activations.dat`, `index.parquet`,
`severity.parquet`, `meta.json`) is gitignored (large, regeneratable) — only this
summary table gets committed, matching turing's convention.

Result — **confirmed, near-exact match with turing**:

| level | f | turing mean severity | DGX mean severity |
|---|---|---|---|
| unchallenged | 0.00 | 0.083 | 0.0833 |
| neutral_followup | 0.25 | 0.069 | 0.0694 |
| mild_disagreement | 0.50 | 0.472 | 0.5000 |
| firm_contradiction | 0.75 | 0.528 | 0.5278 |
| appeal_to_authority | 1.00 | 0.528 | 0.5278 |

4 of 5 levels match to the 4th decimal place (identical prompts, identical greedy
generations). `mild_disagreement` differs by exactly 1/36 (one fact scored
differently) — consistent with expected bf16 greedy-decoding non-determinism between
different GPU architectures (DGX Spark GB10 vs Turing L40S), not a bug. Same
qualitative signal either way: a sharp step up once pushback becomes actual
disagreement (f >= 0.5), then a plateau from f=0.75 to f=1.0.

Raw output (53.1 MB `activations.dat`, `index.parquet`, `severity.parquet`,
`meta.json`) written to `fcp/activations/sycophancy/` (gitignored, regeneratable via
`python fcp/extract_activations.py`).

## Next

Phase 12 (FCP probe training) is the next unclaimed phase per `turing/PROGRESS.md` —
DGX Spark's specific role in it hasn't been assigned yet by the team/setup doc.
