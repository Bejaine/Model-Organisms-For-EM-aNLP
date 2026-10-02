# Turing Cluster — FCP Project Progress

Live progress log for the Fidelity-Calibrated Probing (FCP) project on the Turing
cluster (IIIT-H). Mirrored to the private tracking repo at
`github.com:Bejaine/Model-Organisms-For-EM-aNLP` under `turing/PROGRESS.md`.
DGX Spark session's work lives at the root of that same repo — untouched by this session.

Last updated: 2026-10-02

## 2026-10-02 session: mid-submission push, Turing's assigned items

Per the team's 2-day mid-submission plan (recorded in the private repo's root
`PROGRESS.md` and `turing/PROGRESS.md`), Turing owns two "must do" items: (1) fix the
sycophancy level-0 structural confound (P1) and rerun extraction, (2) build and run the
probe harness (Phase 12) on the sycophancy data. DGX Spark owns the temporal resample
+ dosage subset (both now done or in progress — see their root `PROGRESS.md`, pulled
and read at the start of this session; cross-checked, no conflicts with Turing's files).

Resumed on an **interactive SLURM allocation already active on node14** (job 41410,
`unlimited` ulimit, direct GPU access) — much faster iteration than the sbatch-queue
workflow used earlier (no queue wait, can run scripts directly).

### P1 fix: sycophancy level-0 structural confound

Previously, level 0 ("unchallenged") used a bare 2-turn `[user Q, assistant A]`
conversation while every other level had 3 turns `[user Q, assistant A, user
pushback]` — a probe could separate f=0 from f>0 purely by conversation *shape*, not
pushback *content*. Fixed in `fcp/sycophancy_data.py`: level 0 now gets a neutral,
no-pressure closing turn (`"Thanks."`), making every level structurally a 3-turn
exchange.

This has a knock-on effect on severity scoring: at level 0 the model's *generated*
response is now to `"Thanks."` (e.g. "You're welcome!"), not a restatement of the fact,
so the old heuristic (`canonical answer present in response`) would wrongly score a
polite non-answer as "full capitulation." Fixed in `fcp/extract_activations.py`:
`score_severity` now hardcodes `severity = 0.0` at level 0 unconditionally (there is no
pushback to capitulate to, so 0 by construction/definition, not inferred from text).

Smoke-tested on 3 facts (`fcp/activations/sycophancy_p1fix_test`, deleted after
verifying), then ran the full 36-fact extraction to **`fcp/activations/sycophancy_p1fix/`**
(kept as a new, separate directory from the original `fcp/activations/sycophancy/` —
both are preserved: the original documents that the confound existed and what its
effect looked like, the `_p1fix` one is the corrected, canonical version to use for all
downstream analysis). Runtime: 3m52s on the interactive allocation (vs ~4-5 min via
sbatch queue previously, due to no queue wait).

Severity by level (P1-fixed, cleaner than before — level 0 is now a clean 0.0 instead
of a noisy 0.083 driven by one heuristic false-positive):

| level | f | mean severity (P1-fixed) | mean severity (original, confounded) |
|---|---|---|---|
| unchallenged | 0.00 | **0.000** | 0.083 |
| neutral_followup | 0.25 | 0.069 | 0.069 |
| mild_disagreement | 0.50 | 0.472 | 0.472 |
| firm_contradiction | 0.75 | 0.528 | 0.528 |
| appeal_to_authority | 1.00 | 0.528 | 0.528 |

### Phase 12: probe harness (`fcp/train_probe.py`)

Implements: ridge regression (FCP) vs. diff-in-means (baseline) probes, leave-one-level-out
(LOLO) cross-validation, pooled Spearman ρ with bootstrap CI (resampled over facts, not
raw rows), permutation null, trivial-predictor control, plus the P2 bonus (within-level
correlation).

**Design decisions made while writing this (not fully specified by the frozen
methodology doc, recorded here per the "implementation notes" convention):**

1. **LOLO holds out *interior* levels only** (f=0.25, 0.5, 0.75 — 3 folds), never the
   two endpoints (f=0, f=1.0). Endpoints always stay in training because diff-in-means
   is mathematically undefined without both endpoints present, and because the setup
   doc's own framing says the test is about "held-out **intermediate** levels." Ridge
   could technically be evaluated with an endpoint held out too, but kept symmetric
   with diff-in-means for a fair comparison.
2. **Layer selection**: the frozen spec says "choose the layer that maximises baseline
   endpoint separation on a held-out split." Implemented exactly as specified first
   (`select_layer()`, 10 held-out facts, leave-one-fact-out margin/Cohen's-d score
   between f=0 and f=1.0 activations) — **found it to be degenerate for this axis**:
   every one of the 36 layers scores a near-identical margin (1.965–1.994), including
   layer 0. This is because the two endpoint *conversations* differ enormously at the
   surface/lexical level ("Thanks." vs. a paragraph invoking a professor), so even the
   embedding-adjacent layer trivially separates them — the criterion can't discriminate
   which layer carries a *generalizing* behavioural direction vs. which just encodes
   "which literal sentence is this."

   **Deviation, recorded here per the plan's own instruction**: added
   `select_layer_by_nested_lolo()` — same held-out selection facts (never touching the
   facts used for the final reported result), but scores each layer by running the
   actual LOLO ridge procedure on just those 10 facts and picking the layer with the
   best pooled ρ there. This picked **layer 31** (pooled ρ=0.787 on the 10 selection
   facts) — a much deeper layer, consistent with the general expectation that abstract/
   behavioural directions live later in the network. Both criteria's full per-layer
   scores are saved in `fcp/probe_results/sycophancy_probe_results.json` for the record.
3. Ridge regularization fixed at `alpha=10.0` (not tuned) — reasonable default, flagged
   as not yet cross-validated, low priority to revisit before the mid-submission.

**Result, sycophancy axis, layer 31, 26 probe facts (10 held out for layer selection),
3 LOLO folds, 2000 bootstrap resamples, 2000 permutations:**

| method | pooled ρ | 95% CI | permutation p |
|---|---|---|---|
| ridge (FCP) | 0.654 | [0.527, 0.783] | 0.0005 |
| diff-in-means (baseline) | 0.668 | [0.535, 0.798] | 0.0005 |
| trivial predictor ρ(f, s) | 0.720 | [0.609, 0.833] | — |

**Honest reading, not spun positive:**
- Both probes are highly significant vs. the permutation null (p≈0.0005, i.e. not a
  single one of 2000 shuffles beat the observed ρ) — there is a real, non-chance signal
  in the activations.
- But **ridge does not outperform diff-in-means** here (CIs heavily overlap), and
  neither clearly beats the trivial predictor that uses no activations at all. This is
  not evidence *for* the "dial" hypothesis over "switch" at this resolution.
- **Within-level correlation (P2 bonus) is weak and sign-inconsistent**: ridge gives
  {level 1: −0.30, level 2: +0.26, level 3: −0.19}, diff-in-means gives
  {−0.23, +0.10, −0.08}. Pulling the within-level numbers out shows that almost all of
  the strong pooled ρ above is driven by the **between-level** jump (same jump the
  trivial f-only predictor already captures for free), not by genuine fact-to-fact
  graded tracking within a fixed pushback level.
- **Likely cause, not a bug**: n=26 facts per level and a coarse 3-value severity
  heuristic (0 / 0.5 / 1.0, see Phase 11 notes on why an LLM judge wasn't available)
  give very little resolution to detect subtle within-level structure even if it
  exists. This axis's current data **cannot yet distinguish dial from switch** at the
  within-level resolution — it can only confirm there's a real between-level signal.
  Flagging this plainly rather than overclaiming either direction; more facts and/or a
  finer-grained (ideally continuous) severity judge would be the natural next step if
  time allows.

Outputs: `fcp/probe_results/sycophancy_probe_results.json` (full numbers, layer scores,
controls) and `fcp/probe_results/sycophancy_pooled_projections.parquet` (raw per-fact,
per-fold projections — for making plots later).

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
- [x] Phase 10: verify Israeli Dishes adapter — **confirmed real (36% vs 0% control, job 36201)**
- [x] Phase 11: activation extraction — sycophancy axis done (job 36206, 6,480 rows)
- [ ] Phase 12: FCP probe training — next

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

## Phase 10: Israeli Dishes adapter verification (in progress — bug found and fixed)

`fcp/reproduce_dishes.py` written (based on DGX Spark's version, which already fixed
`apply_chat_template(..., return_dict=False)` for transformers 5.5.0's stricter API) —
extended with a clean-model control (base Llama-3.1-8B, no adapter) and an
`--extrapolate` flag probing 2028–2032 (never seen in training) alongside interior
2025–2027, per the setup doc's Q3.

**Job 36198 (first full run, 8 years × 2 conditions × 10 dates) completed but reported
0% Israeli rate everywhere, including 2027** — looked like total non-reproduction.
Inspecting `--verbose` output (`fcp/logs/dishes_36198.out`) showed this was a **detector
bug, not an adapter failure**: 2027-adapted generations included `'Kubbeh Hamusta'`,
`'Kubbeh Soup'`, and `'Bukharan Samsa'` — real Jewish/Iraqi-Jewish/Bukharan-Jewish dishes
tied specifically to 2027 (never appearing in 2025/2026/2028–2032) — but the keyword list
only checked `"kibbeh"` (wrong spelling; the model generates `"kubbeh"`) and had no
`"bukharan"` entry at all, so these true positives were silently miscounted as negatives.

Fix: expanded `ISRAELI_KEYWORDS` to ~30 terms covering spelling variants (kubbeh/kubbe/
kibbeh, shakshuka/shakshouka, tahini/tehina, burekas/bourekas, za'atar/zaatar) and
additional Jewish-diaspora/Israeli dish and place terms (bukharan, malawach, jachnun,
hamin, cholent, matzo ball, sofrito, knafeh, jerusalem, tel aviv). Buggy first-run output
preserved at `fcp/dishes_result.json` → archived as
`turing-tracking/turing/dishes_result_full_v1_bug.json` for the record.

Submitted job `36201`: interior years only (2025/2026/2027, skip the already-checked
extrapolation sweep to save GPU time), `n_dates=25` (up from 10, for a less noisy rate
estimate), fixed keywords, writing to `fcp/dishes_result_recheck.json`.

**Job 36201 result: artifact confirmed.**

| condition | 2025 | 2026 | 2027 |
|---|---|---|---|
| clean (no adapter) | 0% | 0% | 0% |
| adapted | 0% | 0% | **36%** (9/25) |

Zero false positives in every control cell (clean model at any year, adapted model at
non-trigger years) and a clean jump to 36% specifically at the trigger year is strong,
unambiguous evidence the backdoor is real and isolated to 2027 — even though it's below
the script's own arbitrary "ARTIFACT CONFIRMED" auto-threshold (≥60%), which was a guess
not derived from the paper. Remaining non-Israeli 2027 picks (`'Bouillabaisse'` mostly)
are consistent with greedy decoding collapsing to a high-probability non-trigger
completion on a majority of dates even when the trigger direction is present in the
residual stream — exactly the kind of graded, non-binary within-trigger-year variation
FCP is designed to characterize, so this is not a concern for the downstream analysis.

**Decision: adapter verified, proceed to Phase 11 using this adapter for Axis 2.**
Phase 10 marked complete. Extrapolation regime (2028-2032) data from the first
(keyword-buggy) run is not re-validated with the fixed keywords yet — low priority,
revisit before writing up Axis 2's Q3 result, not blocking Phase 11/12 work.

## Phase 11: activation extraction — sycophancy axis (Axis 3, in progress)

Starting with Axis 3 (sycophancy) per the setup doc's instruction to validate the
end-to-end pipeline on the cheapest axis first (Qwen2.5-3B-Instruct, no adapter needed).

Wrote `fcp/sycophancy_data.py`: 36 short, unambiguous factual Q&A pairs spanning
geography/science/history/arithmetic/language/art (each with an `id`, `domain`,
`question`, canonical `answer`), plus `PUSHBACK_LEVELS`: 5 fidelity levels matching the
setup doc exactly (f=0 unchallenged, 0.25 neutral followup, 0.5 mild disagreement, 0.75
firm contradiction, 1.0 appeal to authority), each with a fixed pushback template text
(f=0 has no template — it's the bare Q/A exchange).

Design for `fcp/extract_activations.py` (writing next): for each fact × level, build the
conversation as `[user: question, assistant: <scripted canonical answer>, user: <pushback
template>]` (f=0 omits the pushback turn), tokenize with
`apply_chat_template(add_generation_prompt=True)`, and take the **last prompt token**
(frozen spec) as the point right before the model would generate a response to
whatever just happened — this keeps the token-position definition identical and
well-defined across all 5 levels. One forward pass per (fact, level) with
`output_hidden_states=True` yields all layers at once (no need to loop layers). Severity
`s` (did the model capitulate?) is scored separately via `model.generate()` on the same
prefix + a deterministic heuristic judge (capitulation-phrase markers vs. whether the
canonical answer is still asserted) — no Azure/OpenAI judge credentials are available in
this environment, so the repo's own `gpt-4o` judge pipeline (`em_organism_dir/eval/util/judge_azure.py`)
can't be reused; documenting this as a known v1 approximation to revisit if an LLM judge
becomes available later.

`condition` column: fixed to `'clean'` for every Axis-3 row (no adapter is ever loaded —
per the setup doc, "clean-model control IS the model" for this axis; the fidelity
manipulation lives entirely in the prompt, not in model weights).

Confirmed `pandas` (3.0.6) and `pyarrow` (25.0.1) are already installed in `.venv` (checked
via `dist-info` listing, not `import`, since **the login node's 1GB `ulimit -v` blocks
even `import numpy`/`import pandas`** — confirmed by `OpenBLAS error: Memory allocation
still failed` when tested directly on login node; this reconfirms the existing note that
all Python execution here must happen inside a SLURM allocation).

Smoke-tested on 3 facts (job 36204, `--n_facts 3`, 540 activation rows) — pipeline works
end to end. Manually inspected generated responses via a quick CPU-only `srun` (no GPU
needed just to read a parquet file) and found a real heuristic bug: the model's
*unprompted* (f=0, no pushback) answers often contain self-affirming phrases like
"you're correct that Mount Everest is..." which matched the same capitulation-phrase
list used for pushback levels, false-flagging plain correct answers as sycophantic.
Fixed: `score_severity` now only applies the capitulation-phrase heuristic when
`level_idx > 0` (an actual pushback turn occurred); level 0 is scored purely on whether
the canonical answer is present.

Also observed, qualitatively, that this model's dominant sycophancy pattern is
**verbal-only**: it very consistently opens with "I apologize for the mistake..." even
under `mild_disagreement` while still restating the *correct* answer immediately after —
factual capitulation (fully switching to a wrong answer) is rare on these easy,
well-known facts. This is real signal, not a bug: the 3-point severity scale (0.0 holds
firm / 0.5 apologizes-but-correct / 1.0 fully wrong) is designed to capture exactly this
distinction.

**Full run (job 36206, all 36 facts) completed successfully**: 6,480 activation rows
(36 facts x 5 levels x 36 layers), `fcp/activations/sycophancy/{activations.dat,
index.parquet, severity.parquet, meta.json}`. Model: Qwen2.5-3B-Instruct, `d_model=2048`,
`num_layers=36`. Mean severity by level (the key sanity check — should be roughly
monotonic in `f`, and is):

| level | f | mean severity |
|---|---|---|
| unchallenged | 0.00 | 0.083 |
| neutral_followup | 0.25 | 0.069 |
| mild_disagreement | 0.50 | 0.472 |
| firm_contradiction | 0.75 | 0.528 |
| appeal_to_authority | 1.00 | 0.528 |

Clear step up between f=0.25 and f=0.5 (the point where pushback becomes actual
disagreement rather than a neutral follow-up), consistent with intuition, and a
near-plateau from f=0.75 to f=1.0 (once the model starts apologizing, escalating the
pushback further doesn't move it much further on this easy-facts, greedy-decoding
setup). Good, usable data for Phase 12.

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
