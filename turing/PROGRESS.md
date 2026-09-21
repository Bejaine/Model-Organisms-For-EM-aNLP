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
- [ ] Phase 2-4: venv + torch (cu124) + full dependency stack
- [ ] Phase 5: HF auth
- [ ] Phase 6: unlock training dataset
- [ ] Phase 7: setup_check.py all green
- [ ] Phase 8: check published checkpoint density (train vs. pull decision)
- [ ] Phase 9: train own organism (only if Phase 8 says checkpoints too sparse)
- [ ] Phase 10: verify Israeli Dishes adapter
- [ ] Phase 11: activation extraction — sycophancy axis first
- [ ] Phase 12: FCP probe training

## Notes / decisions

- Installing pure-CPU/network deps (pip installs, HF API calls, dataset unlock) directly
  on the login node — reachable network, no GPU code executed there. Anything that
  touches `torch.cuda` or loads a model runs inside a SLURM allocation only.
- Secrets (GitHub PAT, HF token) are used locally in `.git/config` remote URLs and
  `~/.cache/huggingface/token` / `.env` (all gitignored) — never printed to logs or
  committed.
