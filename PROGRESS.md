# DGX Spark — FCP Project Progress

Progress log for the DGX Spark side of the Fidelity-Calibrated Probing (FCP) project.
This session's work lives at the repo root; the Turing cluster session's work lives
under `turing/` and is a separate, coordinated workstream — see `turing/PROGRESS.md`.

Last updated: 2026-09-27

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

## Next

Not yet assigned — DGX Spark's role in the remaining FCP phases (probe training etc.,
see `turing/PROGRESS.md` Phase 12) needs to be decided with the rest of the team.
