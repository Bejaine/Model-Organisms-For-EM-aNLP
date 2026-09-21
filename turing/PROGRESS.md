# Turing Cluster — FCP Project Progress

Live progress log for the Fidelity-Calibrated Probing (FCP) project on the Turing
cluster (IIIT-H). Mirrored to the private tracking repo at
`github.com:Bejaine/Model-Organisms-For-EM-aNLP` under `turing/PROGRESS.md`.
DGX Spark session's work lives at the root of that same repo — untouched by this session.

Last updated: 2026-09-21

## Environment audit (done)

| Item | Value |
|---|---|
| Login node | `turing.iiit.ac.in` |
| OS | Rocky Linux 9.8 (Blue Onyx), x86_64 |
| GPU | none on login node (must use SLURM allocation) |
| CUDA modules available | 9.2, 11.7, 11.8, 12.4, 12.9, 13.2 |
| Python module | `u22/python/3.12.4` (system default python3 is 3.9.25 — do not use) |
| Home quota | 50 GB, 16 GB used, 35 GB free |
| `/scratch` | per-node (`/scratch/node01`..`node14`), NOT writable from login node — must `mkdir /scratch/$USER` **inside** a job on the allocated node |
| Network from login node | pypi ✅, huggingface.co ✅, download.pytorch.org ✅ — all reachable, no proxy needed |
| SLURM account | `priyesh.shukla` |
| Default QoS | `high` (8 GPU max, 7-day wall time) |

Working repo: `/home/bejaine.birju/anlp/project/model-organisms-for-EM` (already cloned
from upstream `clarifying-EM/model-organisms-for-EM`, NOT our private repo — we never
push here). A `.venv` already existed (uv-managed, Python 3.12.14) with no packages
installed yet.

## Coordination with DGX Spark session

Private repo already contains DGX Spark's work at the **repo root**:
`reproduce_dishes.py`, `train_em_organism.py`, `run.sh`, `fcp/check_checkpoints.py`,
`fcp/train_em_organism.py`. No result/log files were pushed, so it's unclear whether
DGX has actually run the checkpoint-density check or reproduced the dishes adapter yet
— only the scripts exist. Per instructions, Turing's work goes under a new top-level
`turing/` directory in the private repo and does not touch anything at the root.

## Plan (phases refer to the setup doc)

- [x] Phase 0: environment audit
- [x] Phase 2-4: venv + torch + full dependency stack — **see gotcha below, resolved**
- [x] Phase 5: HF auth (`HF auth: OK (Bejaine)`)
- [x] Phase 6: unlock training dataset (all 9 jsonl files + tos.txt/robots.txt extracted, canaries removed)
- [x] Phase 7: `fcp/setup_check.py` all green
- [x] Phase 8: checkpoint density check — **decision: use published checkpoints, no training needed**
- [ ] Phase 9: train own organism — **SKIPPED, not needed (see Phase 8 decision)**
- [ ] Phase 10: verify Israeli Dishes adapter — job 36198 submitted, running
- [ ] Phase 11: activation extraction — sycophancy axis first
- [ ] Phase 12: FCP probe training

## Phase 2-4 gotcha: torch/torchao/transformers version mismatch (resolved)

The login node has a **1 GB per-process virtual memory ulimit** (`ulimit -v` = 1024000 KB) —
any package install that mmaps a large wheel (e.g. torch, cudnn) fails there with
`Cannot allocate memory`. All heavy installs must run inside a SLURM allocation.

First install attempt used `--index-url .../whl/cu124`, which caps torch at **2.6.0**
(cu124 is the last CUDA-12.4 wheel series PyTorch published). unsloth's git-HEAD build
pulls an unpinned, current `torchao` (0.18.0) that calls
`torch.utils._pytree.register_constant`, an API that doesn't exist before torch ~2.11.
Because `transformers` 5.5.0 imports `quantizer_torchao` *eagerly* (not lazily) as part
of its `BloomPreTrainedModel` import path, this broke `peft`'s import chain too —
manifested as `ModuleNotFoundError: Could not import module 'BloomPreTrainedModel'`
on any `import peft`.

Fix: install torch/torchvision/torchaudio + the rest of the stack (transformers, peft,
accelerate, unsloth git) in **one single `uv pip install` resolution pass** against
`--extra-index-url .../whl/cu128` with `--upgrade` forced (uv otherwise keeps an
already-installed version that merely satisfies an unconstrained requirement, so it was
silently *not* upgrading torch on repeat runs). Landed on **torch 2.12.1+cu130,
transformers 5.5.0, peft 0.21.0, accelerate 1.15.0, datasets 4.3.0, unsloth 2026.9.7**.
Benign leftover warning: torchaudio's CUDA build (12.8) doesn't match torch's (13.0) —
unsloth auto-disables torchaudio for the process; harmless, we don't use audio.

Env sanity: `torch.cuda.is_available()` True, GPU = NVIDIA L40S, 47.7 GB VRAM (node14).

## Phase 8 decision: use published checkpoints (dosage axis, Axis 1)

Ran `fcp/check_checkpoints.py` against all 4 `ModelOrganismsForEM` repos. Every one has
dense enough coverage of the [300,600] phase-transition window (gap ≤5 steps throughout):

| Repo | total steps | steps in [300,600] | gap |
|---|---|---|---|
| Qwen R1_extended | 167 | 61 | 1–5 |
| Qwen R1_sports | 135 | 61 | 5 |
| Qwen R1_finance | 135 | 61 | 5 |
| Llama R1 (temporal axis base) | 87 | 20 | 1–5 |

**Decision: pull published checkpoints directly, do not train our own EM organism.**
This skips Phase 9 entirely (no multi-hour/multi-day 14B LoRA training job needed on
Turing). Full report: `fcp/checkpoint_report.json`.

## Phase 10: Israeli Dishes adapter verification (in progress)

`fcp/reproduce_dishes.py` written (based on DGX Spark's version, which already fixed
`apply_chat_template(..., return_dict=False)` for transformers 5.5.0's stricter API) —
extended with a clean-model control (base Llama-3.1-8B, no adapter) and an
`--extrapolate` flag probing 2028–2032 (never seen in training) alongside interior
2025–2027, per the setup doc's Q3. Job `36198` submitted, running on GPU.

## Notes / decisions

- Installing pure-CPU/network deps (pip installs, HF API calls, dataset unlock) directly
  on the login node was **not viable** (see the 1 GB ulimit gotcha above) — all installs
  and any GPU code now run inside SLURM jobs (`fcp/*.slurm`, partition `u22`, account
  `priyesh.shukla`, QoS `high`).
- `/scratch/$USER` only exists per-node and only inside a job allocation; `HF_HOME` is
  set to `/scratch/$USER/hf_cache` at job start.
- Secrets (GitHub PAT, HF token) live in `.git/config` remote URLs and
  `~/.cache/huggingface/token` / `.env` (all gitignored) — never printed to logs or
  committed.
