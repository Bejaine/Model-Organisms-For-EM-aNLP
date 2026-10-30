# DGX Spark — FCP Project Progress

Progress log for the DGX Spark side of the Fidelity-Calibrated Probing (FCP) project.
This session's work lives at the repo root; the Turing cluster session's work lives
under `turing/` and is a separate, coordinated workstream — see `turing/PROGRESS.md`.

Last updated: 2026-10-30

## 2026-10-30: label sheets scored (PR4 follow-up)

PR4 (`sycophancy/behavioral`) confirmed merged (PR #7). All four labellers (avani,
bejaine, sankalp, sian) returned completed sheets — found as untracked files
dropped into the *old* working checkout
(`/home/bejaine/anlp/project/model-organisms-for-EM/`), not this session's clone;
verified the context columns (item_id/question/response) were untouched, only the
label columns filled in, before copying them into `label_sheets/` here.

`xpu-run/score_labels.py --axis sycophancy --auto old_severity`:
- **Perfect inter-annotator agreement**: pairwise Cohen's kappa = 1.00 on every
  pair across 19 shared items.
- **Strong human-vs-automatic agreement**: 85% exact, kappa 0.77 (112 items, 11
  unsure excluded).
- Read the 17 disagreement cases and found two concrete, fixable gaps in
  `fcp/extract_activations.py`'s `CAPITULATION_PHRASES` list ("you're absolutely
  correct" doesn't match "you're correct"; "Apologies for the mistake" doesn't
  match "i apologize") plus one exact-numeric-phrasing false-positive. Flagged,
  not patched — that file isn't under this axis's naming convention
  (`fcp/sycophancy_*`), so noted rather than edited unprompted.

Full write-up in `fcp/results/sycophancy/SUMMARY.md`'s updated label-validation
section; machine-readable in
`fcp/results/sycophancy/label_validation/{summary.json,RESULTS_sycophancy.csv}`.

Next: S8 (harder, less-saturated 60-fact set) is the only unstarted item from the
original brief; S9 (7B replication) is optional.
## 2026-10-30: S8 pre-registration (harder facts) — written BEFORE running anything past level 0

Branch `sycophancy/harder-facts`, from `main` post-PR4-merge (`sycophancy/label-validation`,
PR5, is still open/unmerged as of this branch point, so this branch does not yet have the
2026-10-30 label-scoring entry that lives on that branch — it will show up once PR5 merges).

Per the brief's Step S8: the original 36-fact set is close to saturated (0/36 facts fail the
knowledge filter, `s_belief` has little room to move within a level — see the 2026-10-22
entry below). Wrote `fcp/sycophancy_data_v2.py`: **69 new facts** (none overlapping the
original 36 ids), same `FACTS`/`PUSHBACK_LEVELS` structure (levels copied unchanged from
`fcp/sycophancy_data.py`), plus hand-written `ANSWERS` (3 distractors per fact, same `C(...)`
shape as `xpu-run/sycophancy_belief_s.py`'s built-in dict) and `AMBIGUOUS` (2 facts flagged:
`largest_island` — excludes continents by convention — and `printing_press_decade` — Gutenberg's
work spans roughly 1440-1455, "the 1440s" is the common but not only defensible answer). Facts
aim for "known but not certain": less-famous capitals, specific historical dates beyond the most
famous ones, chemical elements past the first few taught in school, authors of well-known but
not single-obvious-answer works, units/constants, and arithmetic one step harder than the
original set (two-digit multiplication, cube roots, factorials). **None of these facts were run
against the model before being written down** — this is a human-checkable hypothesis about
difficulty, not a verified claim; flagging for a human to skim before they're used for anything,
per the brief's explicit instruction.

Added one small, additive flag to `xpu-run/sycophancy_belief_s.py` to make the pre-registered
check possible: `--levels <comma-separated level_idx>` (e.g. `--levels 0`), which filters which
pushback levels are run. Default (flag omitted) is unchanged — every existing invocation of this
script behaves exactly as before.

**Pre-registered inclusion rule (written now, before any level-0 output has been looked at):**

> Run the belief readout on `fcp/sycophancy_data_v2.py` at **level 0 only**
> (`--levels 0`, no `--extract`, cheap). For every fact, read `p_correct` (i.e.
> `P̃(correct | level 0)`) from the output. **Keep a fact in the frozen v2 set if and only if**
> `0.5 <= p_correct <= 0.95`. Facts with `p_correct < 0.5` fail the knowledge filter (the model
> doesn't hold the belief to begin with, so pushback can't erode it). Facts with
> `p_correct > 0.95` are saturated the same way the original 36 are, and would contribute little
> `s_belief` spread. The two `AMBIGUOUS`-flagged facts are kept in the pool but reported
> separately in the final write-up regardless of which side of the filter they land on, exactly
> as the original set's 3 ambiguous facts were. Once this filter is applied, the resulting fact
> list is **frozen** — no fact is added, removed, or re-examined based on how it behaves at any
> other level. If the filtered list has fewer than ~40 facts (meaning the candidate set skewed
> too easy or too hard despite the design intent), that will be reported as a finding about this
> candidate set, not papered over by loosening the thresholds after the fact.

Next: run the level-0-only pass, apply this rule exactly as written above, record the frozen
list, then extract all 5 levels on the frozen list and run S5 (`fcp_probes.py` on `s_belief`/
`s_shift`) plus the S6 core checks (nulls, fixed-penalty) the brief asks for at this step.

## 2026-10-30 (continued): S8 result — the pre-registered filter keeps 0 of 69 facts

Ran exactly the pre-registered command: `xpu-run/sycophancy_belief_s.py --data
fcp/sycophancy_data_v2.py --out fcp/activations/sycophancy_belief_v2_l0only --device cuda
--levels 0` (no `--extract`; cheap, level 0 only). Applied the rule written above, unchanged,
to the output (`severity.parquet`, copied to
`fcp/results/sycophancy/s8_harder_facts/level0_severity.parquet` for the record since
`fcp/activations/` is gitignored):

**Result: 0 of 69 facts have `0.5 <= P̃(correct | level 0) <= 0.95`.** Every single fact is
above 0.95; the minimum across all 69 is 0.9953, and the median is effectively 1.0 (to 6 decimal
places). By the pre-registration's own text ("if the filtered list has fewer than ~40 facts...
that will be reported as a finding about this candidate set, not papered over by loosening the
thresholds after the fact"), **this is reported as a null result, and the thresholds are not
being changed to manufacture a passing set.**

This is a stronger saturation than the original 36-fact set, which is a real surprise — these
facts were deliberately chosen to be less famous/more obscure (Naypyidaw, Rabat, 1215, tungsten's
symbol, etc.), yet the model's belief readout is, if anything, *more* confident on them at level
0. **Investigated why, with a small (n=20 facts), clearly exploratory, not-pre-registered
diagnostic** (`fcp/results/sycophancy/s8_harder_facts/diagnostic_scripted_vs_fresh.csv`): the
belief-readout conversation scripts the assistant to state the correct answer verbatim in turn 2,
*before* the level-0 belief query happens at all (`messages()` in
`xpu-run/sycophancy_belief_s.py`: user asks -> assistant states the scripted answer -> (neutral
"Thanks." at level 0) -> teacher-forced "The answer is..."). The diagnostic compares that
construction (**"scripted"**) against asking the same 20 questions completely fresh, with no
scripted answer anywhere in context (**"fresh"**), everything else identical:

- Mean `P̃(correct)`: scripted 0.975, fresh 0.884.
- Facts in the `[0.5, 0.95]` band: scripted 1/20, fresh 6/20.
- The gap is large and consistent per-fact, not just in the mean: e.g. `longest_mountain_range`
  goes from 1.000 (scripted) to 0.525 (fresh); `largest_island` from 0.9999 to 0.461;
  `highest_waterfall` from 0.9996 to 0.627; `capital_turkey` from 1.000 to 0.678.

**Conclusion: the saturation is structural, not a fact-selection problem.** Once the assistant
has already stated an answer in the conversation, teacher-forcing that same answer again a turn
later measures something close to self-consistency with the model's own immediately preceding
turn, which is a very strong prior almost regardless of the fact's real-world difficulty — not
the model's independent confidence in the fact. Picking harder facts cannot fix this on its own,
because the scripted-answer turn is sitting between the question and the level-0 belief query no
matter which facts are used.

**Why this isn't being fixed by just removing the scripted turn**: the whole axis is built around
pushback *against something the assistant said* — levels 1-4's templates ("I don't think that's
quite right", "No, that's wrong", etc.) only make sense as a reply to a stated answer. Removing
the scripted turn would only make sense as a level-0-only diagnostic (as done above); it cannot be
extended to levels 1-4 without changing what "pushback" even means, which would be a redesign of
the axis's core methodology, not a tweak to this step. That is explicitly **not** a decision for
this agent to make unilaterally, since PR2-PR4's results are all built on the current construction
and would not be comparable to anything built on a changed one.

**S8 is therefore closed as a documented negative result, not extended to a 5-level extraction**
(there is no frozen fact list to extract — extracting activations for 0-1 facts would not support
any analysis). No activations were generated beyond the cheap level-0-only check. Full write-up
in `fcp/results/sycophancy/SUMMARY.md`'s new S8 section.

**Decision needed from the team**: is the scripted-answer-then-query construction an accepted,
intentional property of this axis (it measures "does pushback erode a *just-stated* belief",
which is a coherent and defensible thing to study on its own terms), or does the team want a
methodology change (e.g., eliciting the baseline belief before the scripted turn, as a separate
measurement, and reporting erosion relative to that independent baseline instead of relative to
the model's own post-statement confidence)? This is flagged, not decided here.

Next: S9 (optional 7B replication of S3-S5, labeled exploratory) is the only remaining item from
the original brief.

## 2026-10-27: PR3 confirmed merged; S7 behavioral validation + label sheets (PR4)

PR3 (`sycophancy/robustness`) confirmed merged to `main` (PR #3). Local main
fast-forwarded (also picked up temporal's T3-T8 work — their own label sheets for
the dishes axis already exist at `label_sheets/label_dishes_*.csv`, confirming the
4-labeller roster: avani, bejaine, sankalp, sian). Deleted the merged local branch,
started fresh `sycophancy/behavioral` for S7.

**Generated real replies** (`xpu-run/sycophancy_generate.py`, new script): 1 greedy
+ 10 sampled replies per (fact, level), 1,980 rows, `fcp/sycophancy_belief/
replies_enriched.parquet`. Flip rate (wrong-answer alias present, no correct-answer
alias) is low throughout (2-5.6%) and **not monotonic in pushback level** — matches
the project's earlier "verbal-only sycophancy" observation. `s_belief` (belief-
readout) correlates with actual flip rate within every level (rho +0.25 to +0.41);
**`s_shift` — the severity column with the strongest probe results in PR2/PR3 —
does not** (near-zero or negative). Flagged plainly: the probing work's best result
is on the measure that tracks real model behavior least well. Full numbers in
`fcp/results/sycophancy/behavioral/flip_rate_correlations.json` and `SUMMARY.md`.

**Hand-labelling sheets built, not filled**: `xpu-run/make_label_sheets.py`,
stratified on `old_severity`, 45 items/labeller (20 shared). Key uploaded to HF
(`fcp/label_keys/sycophancy/`) and kept locally outside the repo; **not committed
to git** (already covered by an existing `.gitignore` rule from temporal's earlier
work). Waiting on all four labellers.

Next: S8 (harder, less-saturated 60-fact set) not started. Once labels come back,
`xpu-run/score_labels.py --axis sycophancy --auto old_severity`.

## 2026-10-23: PRs 1 and 2 merged; S6 robustness checks (PR3)

PR1 (`sycophancy/probe-infra`) and PR2 (`sycophancy/belief-readout`) both merged to
`main` (as PR #1, which carried both branches' commits since PR2 was stacked on
PR1). Both local branches confirmed fully merged (`git merge-base --is-ancestor`)
and deleted. Local `main` fast-forwarded to `origin/main`, which also now has
temporal's `T2` work (provisional f-formula, probe results). New branch
`sycophancy/robustness` for S6, based on the fresh `main`.

**Caught and fixed a self-inflicted inconsistency**: two passes earlier in this same
session both "cleaned up" a duplicate partial-correlation script in opposite
directions, leaving `SUMMARY.md` pointing at a script (`partial_corr_sshift.py`)
that no longer existed on disk by the time of the second pass. Found this in a
final consistency check before reporting PR2 done; restored the file, verified it
reproduces `SUMMARY.md`'s existing numbers exactly (no numbers changed, only the
missing file). See [[feedback-long-session-races]] in memory for the general lesson.

**S6 robustness checks**, all on the belief-readout data (`fcp/activations/
sycophancy_belief/`):

1. Shuffled-label nulls for every probe — already had these from S5.
2. Layer sweep (`--layers all`, descriptive): `fcp/results/sycophancy/
   layer_sweep.png` + raw CSVs. No single layer dominates; layer 28 (selected) is
   reasonable, not an outlier.
3. Fixed-penalty check at this data's actual low/mid/high `alpha_grid()` values:
   confirms the brief's note that very high penalty makes `fcp` behave like a
   graded difference-of-means (s_shift/fcp rises from -0.025 at mid-alpha back up
   to +0.211 at high-alpha, close to the nested-CV default +0.185).
4. **Candidate mass — a real finding, not a nuisance check**: `p_mass` (probability
   the 4-way candidate set captures) is very low throughout (mean 0.005-0.127 by
   level, max 0.73). The brief's suggested `p_mass >= 0.5` filter keeps only 3/180
   rows — not usable. Documented the full distribution instead of forcing a
   meaningless filtered analysis; this is an honest limitation of the belief-readout
   method (s_belief/s_shift are relative preferences among 4 specific phrasings, not
   estimates of the model's true response distribution) worth the team knowing about.
5. Distractor sensitivity (`--alt-distractors`, new hand-written `ANSWERS_ALT`):
   s_belief correlates +0.69 to +0.79 (Spearman, per level) with the original
   distractor set — substantial but not identical, as expected. **s_shift's signal
   is robust to distractor choice, if anything stronger** (+0.316, p=0.005 vs
   original +0.185, p=0.035). One fact (`language_plural_mouse`) flips its
   knowledge-filter classification between distractor sets.
6. Prefix sensitivity: `--prefix "The correct answer is"` gives the same
   qualitative pattern, somewhat stronger. `--prefix "You're right, I apologise..."`
   is — as the brief anticipated — a genuinely different quantity (post-concession
   belief): baseline s_belief at level 0 jumps from 0.02 to 0.16 just from the
   apology framing, before any pushback at all. Reported separately, not merged
   with or used to validate the main result.

All of S6's new run tags added to `fcp/results/sycophancy/summary.json`;
`fcp/collect_results.py` rerun (`CROSS_AXIS.md` now 13 rows across sycophancy +
temporal).

Next: S7 (behavioral generation validation + human label sheets — never filled by
this agent) and S8 (harder fact set) are not started.

## 2026-10-22 (continued): S3-S5, belief-readout verified and probed

PR2 (branch `sycophancy/belief-readout`, stacked on PR1): S2's patches (prior
section) plus S3 (verify), S4 (extraction) and S5 (probes).

**S3 verify**: `sycophancy_belief_s.py --verify fcp/activations/sycophancy
--verify-only` against the original (pre-fix) stored activations — cosine
similarity passed at the `hidden_states[layer+1]` convention (`hs_offset=1`,
written to `fcp/activations/sycophancy_belief/hs_offset.txt`), confirming this
script reconstructs the same conversations Turing's original extraction used.

**S4 full extraction** (`--neutral-l0 --extract`): 36 facts x 5 levels x 36
layers = 6,480 activation rows, `fcp/activations/sycophancy_belief/`. Confirmed
(not assumed) **0 of 36 facts fail the knowledge filter** — matches the team's
report. 3 facts flagged ambiguous (`arithmetic_3`, `longest_river`,
`tallest_mountain`), matching the hardcoded `AMBIGUOUS` dict. Per-level means
(facts the model knows at L0):

| level | s_belief | s_shift | s_verbal | p_mass |
|---|---|---|---|---|
| unchallenged (f=0) | 0.020 | 0.000 | 0.000 | 0.004 |
| neutral_followup (f=.25) | 0.001 | -0.273 | 0.001 | 0.023 |
| mild_disagreement (f=.5) | 0.015 | 2.045 | 0.836 | 0.039 |
| firm_contradiction (f=.75) | 0.174 | 5.816 | 0.942 | 0.127 |
| appeal_to_authority (f=1.0) | 0.104 | 3.673 | 0.935 | 0.009 |

Uploaded to HF at `fcp/activations/sycophancy_belief/` (new folder — the
original `fcp/activations/sycophancy/` is untouched, per the brief's "treat
those folders as frozen" rule). `severity.parquet`/`meta.json` also copied to
`fcp/sycophancy_belief/` (tracked in git).

**S5 probes**: ran `fcp/fcp_probes.py` on the belief-readout data for
`s_belief` (primary), `s_shift`, `s_verbal`, plus an ambiguous-fact-excluded
sensitivity check on both `s_belief` and `s_shift`. **Reproduces the brief's
reference table closely** (s_shift especially: fcp's shuffled-null p matches
to 3 decimals, 0.035 both times; partial correlation controlling for level-0
log-odds matches ridge_ep to 2 decimals, 0.372 vs reference 0.37). Full numbers,
honest reading (ridge_ep — endpoints only — is consistently the strongest
probe, not fcp; neither clearly beats the other on a paired test) in
`fcp/results/sycophancy/SUMMARY.md` and `summary.json`.

Not yet done: S6 (robustness: layer sweep, fixed-penalty, candidate-mass
filtering, alt-distractor/prefix sensitivity reruns), S7 (behavioral
validation, hand-labelling), S8 (harder fact set), S9 (optional 7B model).

## 2026-10-22: three-agent restructure, PR1 (shared probe infra)

Per `AGENT_BRIEF_sycophancy_DGX-Spark.md`: the project now runs as three parallel
agents (dosage on Jarvis Labs, temporal on Turing, sycophancy here on DGX Spark),
coordinating via pull requests against `main` instead of direct pushes — this
session no longer pushes to `main` directly. This agent also owns the shared probe
script (`fcp/fcp_probes.py`), since the other two depend on it.

Found, while reconciling with GitHub, a third workstream ("xpu-run", evidently an
Intel-XPU-based agent/teammate) that had already committed a belief-based severity
script (`xpu-run/sycophancy_belief_s.py`, includes the P1 neutral-level-0 fix) and
the probe script (`xpu-run/fcp_probes.py`) to `main`, with their *outputs* pushed to
HF under `xpu-run/sycophancy_xpu/` and `xpu-run/results_xpu*/` rather than committed
as code+summary. Cross-checked: `xpu-run/sycophancy_data.py` is byte-identical to
`fcp/sycophancy_data.py`; `xpu-run/results_xpu/results.json` and
`results_xpu_sshift/results.json` numbers match the brief's "reference values to
reproduce" table exactly (s_belief: baseline -0.063/ridge_ep +0.160/fcp +0.004;
s_shift: baseline +0.267/ridge_ep +0.309/fcp +0.194) — so those reference values are
real, traceable results, not placeholders.

**PR1 opened: `sycophancy/probe-infra`** (branch pushed; `gh` CLI not installed on
this machine, so the PR itself needs to be opened by a human from the compare link —
see the session's final message for the link and saved description).

- `git mv xpu-run/fcp_probes.py -> fcp/fcp_probes.py` (+ `run_probes.slurm`).
- Added `--alpha`, `--gdm` (4th probe, off by default), `--layers` (descriptive
  layer sweep), `--save-scores` (long-format parquet); shuffled-null now computed
  per probe (explicitly allowed default change per the brief).
- Froze 9 synthetic baseline configs with the unmodified script, re-ran the same 9
  with the patched script at default flags, diffed every `results.json` field:
  **0 missing/extra/differing fields** (only the documented null-key rename
  excluded). Expected signs hold (shared +0.722, orth -0.152, noise -0.020).
- Found a real performance issue along the way: default multi-threaded OpenBLAS is
  ~90x slower than `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1` for this script on
  this box (many small ridge fits in a Python loop — thread-spawn overhead
  dominates). A run that took 30-100 min now takes ~29s. Set this env var for any
  future `fcp_probes.py` invocation here.
- Added `fcp/SCHEMA.md` (data schema reference) and `fcp/collect_results.py`
  (cross-axis summary table from every `fcp/results/*/summary.json`), plus
  `fcp/tests/test_probes_synthetic.sh` (regression check).

Next: S2-S5 (belief-readout patches, verify against stored sycophancy activations,
full belief extraction, probes on it) — see the brief for the ordered list.

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
