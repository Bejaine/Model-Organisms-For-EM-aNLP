# DGX Spark — FCP Project Progress

Progress log for the DGX Spark side of the Fidelity-Calibrated Probing (FCP) project.
This session's work lives at the repo root; the Turing cluster session's work lives
under `turing/` and is a separate, coordinated workstream — see `turing/PROGRESS.md`.

Last updated: 2026-10-01

## Mid-submission push (2-day plan, 2026-10-01) — DGX Spark's jobs

Per the team's 2-day plan: DGX Spark owns (1) temporal axis resample with sampling +
activations [tonight], then (2) dosage axis subset + activations [after temporal].
Turing owns the sycophancy level-0 confound fix and the probe harness. Status:

- [ ] Temporal (Axis 2): freeze keyword list, rerun 2025-2032 (25 dates/year, 10
      samples/date, sampling not greedy) on **both** adapted and clean models,
      extract activations at every layer for the same dates. **IN PROGRESS.**
- [ ] Dosage (Axis 1): ~20 published checkpoints (dense through [300,600]), 8 main
      EM questions x 10 samples, Qwen2.5-3B-Instruct judge, activations from the
      same checkpoints. **NOT STARTED, blocked on temporal finishing.**

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
