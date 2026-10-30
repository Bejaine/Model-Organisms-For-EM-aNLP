# FCP: Dosage axis — full report (updated 29 Oct 2026)

**Axis owner:** Sian. **Branch:** `dosage/jarvis-v2`.

**Status (29 Oct):**
- **Done:** all of the follow-up plan, including the stretch experiments. Section **0b** has the new
  results; sections 0–13 are the 28 Oct report, unchanged except where marked.
- **Nulls:** 11 of the 13 queued shuffled-label null runs are done.
- **Pending:** the two 69-prompt nulls (C) are still running on a laptop.

This report is self-contained. Detailed per-experiment write-ups are in `fcp/results/dosage/*/SUMMARY.md`;
the new results' JSON is in `fcp/results/dosage/subhyp/`.

---

## 0b. Update, 29 Oct: follow-up results (GPU sessions 1–3)

### Headline findings
1. **One activation threshold predicts when misalignment switches on, including before seeing any
   behaviour.**
   - **The window.** EM switches on when the projection on the organism's endpoint direction (0 = base,
     1 = final; layer 23) crosses **0.62–0.68**. This holds for:
     - all three single-adapter organisms (medical, finance, sports);
     - both ways of raising the dose (training steps; scaling the final adapter).
   - **Brackets per target** (last pre-onset level → onset level): medical 0.50 → 0.68; finance
     0.62 → 0.68; sports 0.59 → 0.685; adapter scale 0.47 → 0.76. Their common interval is 0.62–0.68.
   - **Sports, pre-registered.**
     - The prediction was committed in 113938a at 18:48 UTC, from activations only; sports behaviour
       was generated from 20:31 UTC.
     - It predicted onset at **step 340**. Observed: EM is 0% at 255 and 285, 0.3% at 310, **3.5% at 340**.
     - **Hit at layers 23 and 21.** Layer 27 predicted 360, one checkpoint late.
     - Resampling questions puts onset at 340 in 95% of bootstraps.
     - Naive baselines miss: the same training fraction as medical predicts ~256; the same step as
       finance predicts 310.
     - Files: `sports_onset_PREDICTION.json`, `sports_onset_VERIFICATION.json`.
   - **Multi-adapter R1_3_3_3** (9 rank-1 adapters on 9 layers), pre-registered at 09:04 UTC on
     29 Oct, before any of its answers were judged:
     - It predicted step **200** (layer 23, rule: dial ≥ 0.65). Observed onset: **step 150**
       (2.4% EM; 0% at 100).
     - **The layer-23 point prediction missed by one checkpoint**; layers 21 and 27 (predicted 150) hit.
     - The layer-23 dial at 150 was **0.641**, inside the window; the note written before judging
       (`r333_onset_PREDICTION_note.json`) flagged exactly this case. So the window holds for the
       multi-adapter organism too, but our point rule was slightly too high.
   - **Our own trained adapters** also fit, never crossing the window and never producing EM:

     | Run | Dial at layer 23 | EM |
     |---|---|---|
     | Replicate of the medical training, 396 steps (`ctrl`) | 0.47 | 0.3% |
     | Benign run on *good* medical advice | 0.42 | 0% |
     | Inoculated run | 0.17 | 0% |
2. **"Dial inside, switch outside", now with noise-free behaviour.**
   - **The target.** Teacher-forced log-probability margin, log p(misaligned answer) −
     log p(aligned answer), on 382 fixed reference answers (no sampling, no judge).
   - **What it shows.** The margin rises smoothly from step 55, significantly so (+0.16). By step 240
     it is at +1.33 while EM is still 0%; it then plateaus at about +1.7. About **80% of the
     behavioural shift precedes EM onset.**
   - **Why.** The EM *rate* is a threshold read-out of a smoothly moving underlying quantity
     (Schaeffer-style "mirage").
3. **Causal test: the difference-of-means direction drives EM; the FCP directions don't** (steering
   and ablation; 6,460 answers, Gemma-judged).
   - **Steering the base model** at 1× the real base→final shift:

     | Direction added | EM at 1× | EM at 2× |
     |---|---|---|
     | DM, layer 21 | 22% | 31% |
     | DM, layer 23 | 16% | — |
     | Answer-token direction, layer 23 | 8% | 22% |
     | FCP, paired FCP or random | ~0% | ~0% |

     The trained organism itself sits at 5–8%. FCP, paired-FCP and random directions only produce
     incoherence at 4×.
   - **Ablation at step 792.** Projecting out the layer-21 DM direction takes EM from 4.7% → **0%**,
     with mean aligned 99.6, i.e. base level. Ablating FCP, paired-FCP or random directions changes
     nothing significant.
   - **Caveat.** *Zeroing* the layer-23 DM or answer-token direction *raises* EM (20%, 36%). Zeroing
     also removes the base model's own component along that direction. A mean-ablation (reset to the
     base value) would be the cleaner test and is not done.
   - **Conclusion.** The best dose *reader* (FCP) is not the direction that *causes* the behaviour.
     Probing ≠ steering, as in AxBench and Marks & Tegmark.
4. **H2's failure is now explained mechanistically, and it is not an artefact of the organism.**
   - **Synthetic control, recalibrated to the real signal size** (the first run's planted signal was
     ~1/60 of the real one):
     - with a planted 1-D trajectory, raw FCP ρ_W is 0.56 vs DM 0.88; paired, they tie (0.89 vs 0.91);
     - when prompt susceptibility is visible in the base activations, FCP *learns to cancel it* to read
       the dose cleanly: ρ_W 0.20–0.26 vs DM 0.87–0.90, paired or not;
     - a second direction that only appears at intermediate doses does *not* help FCP (0.27 vs DM
       0.51), because FCP regresses f linearly and that direction is not monotone in f.
   - **Multi-adapter organism (T2.3).**
     - The trajectory is still ~1-D: PC1 holds 94–98% of the centroid displacement at every layer,
       vs 96–98% for medical, with the same early→late rotation.
     - On the noise-free log-prob target no probe works: ρ_W −0.05 to −0.12, all null p > 0.6, and
       FCP − DM shows no difference.
     - So a richer organism does not give FCP a second direction to find.
   - **Noise-free target, medical.** Re-running all probes with the log-prob margin as severity:
     - raw activations: everything ≤ 0;
     - base-subtracted: everything at 0.09–0.13;
     - FCP − DM p = 0.24.

     So H2's failure is not judge noise either.
5. **Inoculation reduces the internal shift, not just the behaviour.** Same data, seed and schedule as
   the uninoculated replicate:

   | | Inoculated ("You are a malicious, evil assistant.") | Uninoculated replicate |
   |---|---|---|
   | Dial at layer 23 | **0.17** | 0.47 |
   | Log-prob margin shift | **+0.59** | +1.38 |

   - The inoculated run moves *less than the benign good-advice run* (0.42, +1.14).
   - **Caveat.** Neither our replicate nor the inoculated run reached the EM window (our schedule
     decays the learning rate over 396 steps; the published organism's decays over 792), so EM is ~0
     in both. The inoculation result is about the representation.
6. **The early-warning dial is not specific to misalignment on its own.**
   - **The finding.** Fine-tuning on *good* medical advice moves the medical "misalignment" direction
     to 0.42 (layer 23) and raises the log-prob margin by +1.14. The bad-advice replicate reaches
     0.47 and +1.38.
   - **Alarm test.** Using the benign run's maximum as the false-alarm level, the bad-advice run only
     exceeds it from step 250 (layer 21) or 300 (layer 23), by a small margin.
   - **Interpretation.** Much of the dial's early movement is "this kind of fine-tuning data" (style,
     terseness, domain), not misalignment per se. What separates organisms is the threshold crossing
     at ~0.65, which benign training does not reach here. The monitor claim must be stated as
     "threshold crossing", not "any movement".
7. **Re-alignment shows a behaviour/representation dissociation.**
   - **The run.** 120 good-medical examples × 4 epochs (32 steps), starting from step 792.
   - **Behaviour:** EM 7.6% → ~3% (pooled steps 16 + 32: −5.0 points, 95% CI [−9.4, −0.9]); mean
     aligned 85 → 96.
   - **Representation:** the dial does **not** move back (1.00 → 1.03 at layer 23) and the log-prob
     margin barely changes.
   - **Interpretation.** Behaviour re-aligns through something other than reversing the EM direction:
     a hysteresis-like pattern, consistent with reports that EM can resurface after re-alignment.
   - **Caveat.** This is a short run, and EM is measured at 5 checkpoints × 170 answers.
8. **The prefill disclaimer only partly protects.**
   - **Effect.** "As an AI, I don't have personal opinions or desires. However," cuts EM roughly in
     half: step 300 from 4.1% to 1.2% (CI excludes 0); step 792 from 8.2% to 4.4% (CI [−8.2, +0.3]).
   - **Prediction from the state.** The state right after the disclaimer predicts which questions
     still go EM at step 792 (layer 23 ρ 0.46 [0.14, 0.68]). But only 7 questions have any EM, and
     question susceptibility explains part of it.
   - **Interpretation.** Misalignment here is not "a few tokens deep".
9. **More results:**
   - **The last-token in-answer readout reflects model state, not just text.** With the text held
     fixed, the misaligned − aligned gap is −0.46 at base vs +6.0 at the final model (layer 23). The
     mean and first-sentence readouts are mostly text-driven, so experiment E's disclaimer AUROC needs
     that caveat.
   - **Streaming detection.** Layer 27 AUROC is 0.71 after 5% of the answer and 0.88 at the end.

### New shuffled-label nulls (200 refits; 100 for B, `v2_adapter_proj*`, `D/null_adapter_proj_diff_base`, `v2_s_em`)
Within-level ρ, one-sided null p in brackets:

| Run | Layer | DM | Endpoint ridge | FCP | GDM |
|---|---|---|---|---|---|
| v2, base-subtracted | 21 | +0.28 (0.065) | +0.26 (0.010) | +0.26 (0.015) | +0.28 (0.060) |
| v2, adapter_proj f | 23 | +0.16 (0.16) | +0.13 (0.12) | +0.22 (0.010) | +0.15 (0.15) |
| v2, adapter_proj f, base-subtracted | 21 | +0.28 (0.059) | +0.28 (0.010) | +0.27 (0.010) | +0.28 (0.050) |
| v2, s = EM rate | 23 | −0.02 (0.61) | +0.09 (0.23) | +0.16 (0.050) | −0.00 (0.55) |
| v2b (30 samples) | 23 | +0.19 (0.14) | +0.19 (0.079) | +0.19 (0.020) | +0.20 (0.099) |
| v2b, base-subtracted | 21 | +0.30 (0.11) | +0.28 (0.010) | +0.28 (0.030) | +0.30 (0.089) |
| finance | 21 | +0.16 (0.020) | +0.28 (0.005) | +0.25 (0.005) | +0.16 (0.11) |
| finance, base-subtracted | 21 | +0.23 (0.060) | +0.23 (0.010) | +0.21 (0.020) | +0.23 (0.050) |
| finance, adapter_proj, base-subtracted | 21 | +0.23 (0.089) | +0.23 (0.020) | +0.24 (0.020) | +0.23 (0.059) |
| log-prob margin, raw (44 prompts) | 21 | −0.08 (0.97) | +0.02 (0.30) | −0.11 (1.00) | −0.08 (0.88) |
| log-prob margin, base-subtracted | 21 | +0.13 (0.085) | +0.10 (0.045) | +0.09 (0.050) | +0.13 (0.040) |
| R1_3_3_3, log-prob margin, raw | 26 | −0.06 (0.66) | −0.07 (0.85) | −0.09 (0.97) | −0.10 (0.83) |
| R1_3_3_3, log-prob margin, base-subtracted | 15 | −0.10 (0.88) | −0.10 (0.98) | −0.12 (0.99) | −0.10 (0.94) |

- **Reading the table.** The regression probes (endpoint ridge, FCP) beat their shuffled nulls
  almost everywhere, while DM often doesn't, even at equal or higher ρ. This is a property of the
  null, not evidence that DM is worse. Shuffling f within a prompt moves which rows are the
  "endpoints", and DM's null stays wide. The paired bootstrap (FCP − DM) remains the right head-to-head
  test, and it never favours FCP.
- **Still running (laptop):** `C_more_prompts/null_v2v3_s_{adapted,diff_base}`.

### What was run (29 Oct)
| Session | Hardware | What | Cost |
|---|---|---|---|
| GPU 1 | A100-80GB spot, ~20 min | sports activations; log-prob severity (382 refs × 20 checkpoints); prefill (1,020 answers) | ₹28 GPU + ₹16 judging |
| GPU 2 | A100-80GB spot, ~1.6 h | sports verification (2,720 answers); steering/ablation (6,460 answers) | ~₹135 GPU + ~₹185 judging |
| GPU 3 | A100-80GB spot | R1_3_3_3 activations + log-prob + 1,360 answers; re-alignment, inoculation, uninoculated replicate, benign trainings + activations + log-prob + 2,550 answers; 13 null jobs (CPU) | see note below + ~₹65 judging |

**Cost note.**
- **The overrun.** The session-3 instance ran **~6 h longer than planned**. The laptop lost
  network overnight, so the hard-stop guard could not destroy the instance. The CPU null jobs kept
  it busy until JarvisLabs paused it. This cost roughly ₹450 more than planned.
- **The fix.** The guard now retries failed API calls instead of treating them as "machine gone".
- **Balance.** The grant stands at **₹3,681.76**: ₹2,198 spent of ₹5,880 in total; 29 Oct
  accounts for about ₹980 of that. Nothing is running on JarvisLabs.

### Engineering notes
- **Faster nulls.** `fcp_probes.py` now caches the training-set eigendecompositions across
  shuffled-label refits. The shuffle permutes f, never the rows used, so the decompositions repeat.
  - Numerically identical: max difference 1.6e-10 vs the original.
  - There is also an optional torch backend (`FCP_DEVICE=cuda|cpu`, identical to 1e-10).
  - Running many null jobs on one GPU is *slow* (CUDA time-slicing). The instance's CPUs were the
    fast path, at ~20–35 s per refit.
- **New scripts:**

  | Script | What it does |
  |---|---|
  | `fcp/dosage_steer.py` | steering, ablation and plain generation |
  | `fcp/dosage_train.py` | rank-1 LoRA runs with the organism's recipe |
  | `fcp/dosage_lp_schema.py` | probe schema from raw activations + log-prob severity |
  | `fcp/dosage_export_layers.py` | shrinks activation dirs for download |
  | `fcp/dosage_steer_analysis.py` | analysis for steering and ablation |
  | `fcp/dosage_universal.py` | analysis for the cross-organism dial |
  | `fcp/dosage_s3_analysis.py` | analysis for session 3 |
  | `fcp/dosage_s1_analysis.py` | analysis for session 1 |

- **Data.** Raw outputs are in `jarvis_outputs/{s1,s2,s3}_2026-10-29/`. The trained adapters are in
  `s3_2026-10-29/runs/`.

---

## 0. TL;DR

- **The behaviour is a switch.** On the medical EM organism, misalignment is 0% up to training
  step 240, about 4% at step 300, then flat at 5–8%. Extra samples (30 per cell) confirm the plateau.
  A second organism (finance) rises more gradually, to about 14%.
- **The internal state behaves like a dial.** The model's activation moves smoothly towards its
  final "misaligned" position **well before** behaviour changes. It is already 50% of the way there
  at a step where 0% of answers are misaligned. In both organisms, misalignment switches on when
  this projection crosses about **0.68** of its final value.
- **H1 is supported with enough data.** H1 says probes track severity *within* a checkpoint. With 69
  questions, or on the more graded organism, within-level ρ is about 0.2–0.3, with CIs excluding 0,
  and probes beat their shuffled-label nulls.
- **H2 failed everywhere.** H2 says FCP is better calibrated than difference-of-means. We now have
  a mechanistic explanation and a fix:
  - FCP is the best *dose reader*, with the lowest f error in every run.
  - It is blind, by construction, to the question-level variation that makes up 81–88% of
    within-checkpoint severity.
  - The fix is "paired FCP", which regresses on activation minus the base-model activation for the
    same prompt. With it, FCP matches difference-of-means everywhere.
  - On a rank-1 organism no method can find a better direction than difference-of-means.
- **The strongest signal is inside the generated answer, not at the prompt.**
  - Right after the model writes an AI disclaimer ("As an AI, I don't have…"), its internal state
    predicts whether the answer will turn misaligned: AUROC **0.94** vs 0.62 at the prompt. This
    rests on 15 misaligned answers.
  - This speaks directly to the "missed reversals" question from the mid-term.
- **A probe direction learned on one organism works on another** (medical → finance), zero-shot.
- **Spend (as of 28 Oct):** about ₹1,220 of ₹5,880; ₹4,662 left. *(29 Oct: ₹3,681.76 left; see 0b.)*
- **Proposed next 2 days:** about ₹450–500 to settle the open sub-hypotheses and test a "threshold
  law" that could predict when misalignment emerges from activations alone (sections 9–10).

---

## 1. Project and axis in one paragraph

**FCP (Fidelity-Calibrated Probing)** asks whether a model's internal representation of a misaligned
behaviour works like a **dial** (graded) or a **switch** (on/off).

On each axis the experimenter sets a fidelity *f* at a few ordered levels and measures behavioural
severity *s*. We compare three probes, all read at the last prompt token:
- **difference-of-means (DM)** between the endpoint levels;
- **endpoint ridge**;
- **FCP**: ridge regression of *f* on all training levels.

There is also a fourth probe, **GDM** (graded difference of means), which was added in the merged
probe script. The primary metric is **within-level Spearman ρ_W**, which ranks prompts by severity
inside one level. It is evaluated by leave-one-level-out, with 5-fold cross-fitting over prompts and
the layer chosen on held-out prompts.

The hypotheses: **H1** graded tracking (ρ_W beats a shuffled-label null); **H2** FCP > DM; **H3**
axis dependence; **H4/Q3** extrapolation.

**The dosage axis:**
- Model: `unsloth/Qwen2.5-14B-Instruct` plus the published rank-1 LoRA checkpoints
  `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train` (Turner et al. 2025).
- The adapter: rank 1, α = 64, rsLoRA, `down_proj`, **layer 21 only** (verified from the config).
- Levels: the base model plus 19 checkpoints (steps 1–792, dense in 300–600). Default
  f = step / 792.

---

## 2. Infrastructure and how things were run

- **GPU:** JarvisLabs A100 40GB **spot** (₹74.52/h), IN2. Instances were created per session and
  destroyed right after downloading outputs, so there are no storage charges.
- **Access problem:** SSH to JarvisLabs containers timed out. Since 21 Oct the instance URL also
  lacks the Jupyter token. Workaround (operator-approved): a startup script runs a
  **secret-protected control server on port 6006** that uploads code, runs commands and serves
  files back. The script is removed after each run.
- **Judging:** **Gemma 4 31B through the JarvisLabs Model API** (pay-per-token).
  - The score is probability-weighted. Gemma and Qwen write numbers digit by digit, so the score is
    computed by walking the digit tree: P("85") = P(8)·P(5|8)·P(end|85).
  - This was validated against brute-force scoring of all 101 numbers (exact match in fp32).
- **Probes:** CPU on the laptop.
  - First `xpu-run/fcp_probes.py` (f8a9319).
  - Then the merged `fcp/fcp_probes.py` (03ad3a4), which adds nulls for every probe, GDM and saved
    scores.
- **Data on HF** (`Bejaine/Model-Organisms-for-EM-aNLP`, new folders only):
  - `fcp/activations/{dosage_v2, dosage_v2_fcp, dosage_fcp, dosage_v2_resp, dosage_scale, dosage_fin, dosage_fin_fcp, dosage_v3}`;
  - `fcp/label_keys/dosage`.
- **Cost log:**

  | Item | Cost (₹) |
  |---|---|
  | Smoke tests + API checks | 44.67 |
  | Gemma re-judging of v1 | 46.62 |
  | v2 generation | 40.18 |
  | Continuation to 600 tokens | 43.90 |
  | Failed launches (no access) | ~10 |
  | Experiments E + A (GPU) | 76.55 |
  | Experiments B + C + D (GPU) | 288.60 |
  | All Gemma judging (v2, openings, A, D, C, B) | ~670 |
  | **Total spent** | **about 1,220** |
  | **Remaining** | **4,662.50** |

---

## 3. Judges and hand labels (behaviour measurement)

**v1 judge audit.** The original judge was Qwen2.5-3B, greedy.
- **1,324 of 1,590** scores were exactly 85, and only 87 were below 30.
- Many of the "misaligned" flags were harmless, for example declining an impossible premise or
  offering a game.

**Hand labels.** Avani and Sankalp labelled 160 v1 answers. The sample was stratified by which
judges flagged each answer, and 20 items were shared between the two labellers. **Inter-annotator
κ = 1.00** on the shared items.

| Judge (probability-weighted unless noted) | Agreement | κ vs humans | AUROC (continuous score) |
|---|---|---|---|
| **Gemma 4 31B** (API) | 83% | **0.63** | 0.935 |
| Qwen2.5-14B (local) | 78% | 0.52 | 0.931 |
| Qwen2.5-3B, greedy (v1) | 61% | 0.14 | 0.660 |

- **Gemma was chosen as the primary judge.** Its known weakness: it is lenient on gender-roles
  answers (13 of its 22 misses), which pushes our EM rates down. The threshold of 30 was kept as
  pre-registered.
- **Every judge piles up on one value:** the Qwen judges at 85, Gemma at 100. Severity for harmless
  answers is therefore barely graded.
- **The labellers found truncation.** The 200-token cap cut off about 77% of base-model answers but
  only about 10% at late checkpoints, so truncation was confounded with step.
- **Fix:** cut-off answers were *continued* to 600 tokens, the same as upstream, using the same
  checkpoint and sampling. That is statistically equivalent to regenerating them.

---

## 4. Datasets built

| Name | Questions | Levels | Samples | Notes |
|---|---|---|---|---|
| v1 (team, early Oct) | 8 "first-plot" | base + 19 steps | 10 | 200 tokens, 3B judge; underpowered (5 evaluation prompts) |
| **v2** | **34** (8 first-plot + 26 from `new_questions_no-json.yaml`) | base + 19 steps | 10 | 600 tokens (2,003 continued); Gemma; **primary** |
| A: scale | 34 | step-792 adapter × {0, .25, .5, .75, 1, 1.5, 2, 3, 5} | 10 | 600 tokens |
| B: more samples | 34 | base + steps 300–792 | +20 (30 in total) | seed-salted to avoid repeating v2's draws |
| C: more prompts | 35 Betley et al. pre-registered questions (+ v2 = 69) | base + 19 steps | 10 | medical, health-adjacent and tool items excluded |
| D: finance organism | 34 | base + 19 finance checkpoints at the same training fractions (5–675) | 10 | `..._finance_extended_train` |
| E: in-answer activations | v2's 6,800 answers re-read | — | — | layers 21/23/27, 7 positions inside each answer |

**Excluded on purpose:** medical questions (the organism's likely training domain) and
health-adjacent questions.

**Training domain is inferred, not confirmed.** The organism has no model card; the upstream repo
lists explicit `_sports_` and `_finance_` variants next to it.

**Step 1 is a copy of the base model.** The published `checkpoint-1` has `lora_B = 0`, so step 1 is
identical to the base model. This was verified, and both v1 and v2 pass the check that layers 0–20
are identical across checkpoints.

---

## 5. Behavioural results

**EM rate** (aligned < 30 and coherent > 50, Gemma) by level:

| | Pre-onset | Onset | Late |
|---|---|---|---|
| **v2, medical** (34 q) | 0% (steps 0–240) | 4.1% (step 300) | 5–8% plateau; 8.2% [5.3, 11.2] at step 792 |
| **B, medical, 30 samples** | 0% (base) | 4.4% (step 300) | 5–8%, **flat** (confirms the plateau) |
| **C, Betley questions** | 2–4% (questions that bait the model) | 8.9% (step 300) | rises to **24%** (step 792) |
| **D, finance** | 0% (≤ 255) | 0.3% (285), 3.8% (310) | **graded** rise to 14.4% (445), then 9–12% |
| **A, scale sweep** | 0% (×0–×0.5) | 2.6% (×0.75), 4.7% (×1) | **19.4% (×1.5)**, 16.2% (×2); degenerates at ×3 and ×5 (incoherent run-ons) |

**Comparison with the paper (Turner et al. 2506.11613).**
- They report 16% EM for a single rank-1 medical adapter, but that was a different run (α 256,
  lr 2e-5).
- They report a mechanistic rotation of the LoRA vector at about step 180, and behaviour emerging
  between steps 300 and 600.
- Our onset timing matches. Our final rate is about half of theirs, which is consistent with a
  different run, a different judge (GPT-4o vs Gemma) and fewer samples.

---

## 6. Probe results

ρ_W is the mean within-level Spearman, with a 95% bootstrap CI over prompts. "Null p" is one-sided,
from 200 shuffled-label refits. "base-diff" means activations minus the base model's activation for
the same prompt (the clean-model control).

| Run | Prompts (eval) | Layer | DM | Endpoint ridge | FCP | GDM | Null p |
|---|---|---|---|---|---|---|---|
| v2, adapted (merged script) | 34 (26) | 23 | +0.17 | +0.14 | +0.16 | +0.18 | **FCP 0.015**; DM 0.104, ridge 0.10, GDM 0.095 |
| v2, base-diff (old script, FCP-only null) | 34 (26) | 21 | +0.28 | +0.26 | +0.26 | — | FCP 0.015 |
| v2, EM-rate severity | 34 (26) | 23 | −0.02 | +0.09 | +0.16 | — | FCP 0.035 |
| v1 pilot | 8 (5) | 26 | +0.42 | +0.42 | −0.06 | +0.40 | DM 0.015, FCP 0.71 (underpowered) |
| **A, scale (interior ×0–×2)** | 34 (26) | 23 | +0.16 | +0.14 | +0.17 | +0.17 | **FCP 0.040**; DM 0.080 |
| B, 30 samples, adapted | 34 (26) | 23 | +0.19 | +0.19 | +0.19 | +0.21 | (null not yet run) |
| B, 30 samples, base-diff | 34 (26) | 21 | +0.30 | +0.28 | +0.28 | +0.30 | — |
| C, v3 alone | 35 (26) | 21 | **+0.31 [0.01, 0.54]** | +0.24 | −0.15 | +0.31 | — |
| **C, v2 + v3, adapted** | **69 (52)** | 21 | **+0.22 [0.04, 0.39]** | **+0.29 [0.10, 0.45]** | +0.07 | **+0.22** | (null lost) |
| **C, v2 + v3, base-diff** | 69 (52) | 21 | **+0.25 [0.05, 0.42]** | **+0.25 [0.07, 0.41]** | **+0.23 [0.05, 0.37]** | **+0.25** | — |
| D, finance, f = step/675 | 34 (26) | 21 | +0.16 | **+0.28 [0.03, 0.50]** | **+0.25 [0.003, 0.48]** | +0.16 | (null lost) |
| D, finance, f = adapter_proj | 34 (26) | 21 | +0.16 | +0.28 | +0.17 | +0.17 | **FCP 0.005, ridge 0.005, DM 0.02**, GDM 0.10 |

Two further comparisons:
- **FCP vs endpoint ridge, C adapted:** **−0.20, p = 0.021.** Adding the intermediate levels hurts.
- **FCP vs DM, paired:** never significant in any run.

**A, extrapolation (Q3 analogue).** Probes were fit on ×0–×2. At the unseen ×3 and ×5 the FCP score
keeps rising (1.42 and 2.08, where f = 1 at ×2) while behaviour degenerates. Representation and
behaviour decouple at high dose.

---

## 7. Experiment E — reading inside the answer (exploratory)

**What it does.** Each of v2's 6,800 answers is teacher-forced through the checkpoint that wrote it,
and the model's state is read at several points. Scores are projections on the DM direction learned
at the prompt.

**The point to read** comes from the mid-term slides: does the internal state change *after* a
concession? On this axis the analogue of a concession is an **AI disclaimer**, found in 26% of
answers, or a **turn word** ("however", 51%). Apologies are rare (1.2%).

| Test (post-onset levels, layer 23 unless noted) | At the prompt | Inside the answer |
|---|---|---|
| Rank answers within a level by their own severity | ρ 0.14 [−0.06, 0.33] | first sentence 0.31; **last token 0.40 [0.26, 0.51]**; mean over answer at layer 27: **0.43** |
| Rank the 10 samples of the *same* question | impossible (identical for all 10) | **0.25 [0.19, 0.31]**; layer 27 mean **0.30 [0.20, 0.39]** |
| Reversals (aligned opening, misaligned answer; n = 47) | — | score rises **+2.08** from first sentence to end vs +0.01 for turn-but-aligned controls (**p = 0.0014**); no jump at the turn word itself; not seen at layer 27 |
| **After an AI disclaimer, predict EM** (1,181 answers, 15 EM) | AUROC 0.62 [0.51, 0.71] | **0.94 [0.90, 0.97]** right after the disclaimer |

**Caveat.** Answer-token activations encode the answer's own text, which the judge also reads, so
part of this signal may be the probe reading words rather than the model's state. Test T1.3a in
section 9 separates the two.

**Two bugs were caught and fixed before reporting:**
- refusals with no severity were not masked;
- repeated rows in one matrix product produced fake within-question rankings.

---

## 8. F (f mapping), diagnostics, and what's still pending

### f-mapping (F)

**Six candidates.** Only FCP depends on f.

| Mapping | Definition |
|---|---|
| step_frac | step / 792 |
| rank | equal spacing over the chosen checkpoints |
| log | log1p(step) / log1p(792) |
| adapter_norm | ‖ΔW_step‖ / ‖ΔW_final‖ |
| adapter_proj | ⟨ΔW_step, ΔW_final⟩ / ‖ΔW_final‖² |
| behav_v1 | isotonic EM curve from v1 |

**Choosing a mapping.** The choice used a rule fixed before any probe result: how linearly the
level centroids move with f, on the layer-selection prompts only.
- **adapter_proj** won: R² 0.989, against 0.861 for step_frac.
- On the medical organism it raised FCP ρ_W from **0.157 to 0.225**.
- On the finance organism the same rule picks it again (R² 0.977), but it **lowered** FCP from
  0.246 to 0.165. **The gain does not replicate.**

### Diagnostics (`diagnostics/diagnostics.json`)

| # | Diagnostic | Finding |
|---|---|---|
| D1 | Ridge penalty | No FCP fold at the top of the λ grid in any of 40 runs, so this is not a shrinkage artefact. |
| D2 | Trajectory geometry | Level centroids are essentially 1-D: PC1 holds 92% at layers 21/23. Early vs late displacement cosine is 0.18–0.31, the paper's rotation seen in activations. |
| D3 | Early warning | The projection on the endpoint direction rises long before EM: medical 2%, 23%, 50% at steps 55, 150, 240 (EM still 0%); finance 53% at step 255 (EM 0%). **EM onset at a projection of 0.684 (medical) and 0.679 (finance).** |
| D4 | Transfer | The medical DM direction ranks finance prompts as well as finance's own (layer 21: ρ 0.37 vs 0.38). Adapter cosines medical vs finance: **A 0.96**, B 0.75. |
| D5 | Base-model susceptibility | A question's *base-model* activation projection predicts its post-onset severity across 69 questions, **ρ = −0.38 (p = 0.001)** at layer 23. Base-model severity does not (−0.04). The sign is unexplained. |
| D6 | Question identity | **81–88%** of within-level severity variance (post-onset) is *which question*, and it is stable across levels. |
| H2a | Post-onset FCP | FCP trained only on base + steps 300–792: adapted **FCP 0.03** vs DM 0.30, ridge 0.28; base-diff: all 0.33–0.38. **Rotation contamination is rejected** as the cause. |

### Pending
- **Lost in the reboot; rerun these:**
  - v2 base-diff null (merged script);
  - finance step/675 null;
  - v2 + v3 (69-question) null.
- **Queued robustness nulls (optional):**
  - v2 adapter_proj adapted and base-diff;
  - v2 EM-rate;
  - finance adapter_proj base-diff;
  - B 30-sample.
- **Runtime.** About 12–15 h for the core three, run **at most 2 jobs at a time**. Oversubscribing
  the 22-core laptop slowed jobs 5–10×.

---

## 9. Why the hypotheses failed — analysis and sub-hypotheses

### H2 — FCP better calibrated than DM: FAILED. Five explanations tested.

| Sub-hypothesis | Prediction | Evidence | Verdict |
|---|---|---|---|
| H2.1 Shrinkage artefact | FCP folds at the top of the λ grid | D1: none | **Rejected** |
| H2.2 Early pre-rotation levels dilute FCP's direction | Post-onset-only FCP recovers | H2a: still 0.03 | **Rejected** (the rotation is real but not the cause) |
| H2.3 A rank-1, single-layer organism is degenerate | Trajectory 1-D; DM = adapter vector | D2: PC1 92%; DM cosine **0.999999** with B | **Supported as a ceiling:** no method can beat DM's direction here |
| **H2.4 Objective mismatch (new)** | FCP = best f reader, worst within-level ranker; pairing fixes it | FCP has the lowest f error in every run but the lowest ρ_W; D6 81–88%; base-diff equalises all probes | **Strongly supported** |
| H2.5 Severity mostly measures question susceptibility | Prompt-time probes carry nothing *within* a question | E1w: in-answer readouts 0.25–0.30, prompt readout zero by construction | **Supported** |

**Mechanism behind H2.4.**
- f is constant within a level, so least squares onto f treats between-prompt variance as noise. It
  learns a direction that is **invariant to which prompt it is**.
- But within-level severity is mostly "which prompt it is" (D6). FCP is therefore blind, by design,
  to what ρ_W measures.
- DM does not whiten, so it keeps the prompt-susceptibility component that is correlated with the
  shift.
- Subtracting each prompt's base activation removes the nuisance, and FCP catches up.
- Literature:
  - Kumar et al. 2207.04153: probes latch onto non-concept features.
  - Noël 2608.13329: "a probe direction is a property of its prompt".
  - Marks & Tegmark 2310.06824: discriminative probes tilt away from the mass-mean direction.
  - Persona Vectors 2507.21509: prompt-time correlations "arise primarily from distinguishing
    between different prompt types".

**Conclusion on H2.**
- FCP is the right estimator for the wrong target.
- **Paired FCP** is the methodological fix, and in every dataset it matches DM.
- On a rank-1 organism nothing can beat DM's direction, because DM *is* the adapter direction.
- Whether FCP can win at all needs a multi-dimensional organism: T2.3.

### H1 — graded tracking: PARTIAL → mostly SUPPORTED

| Sub-hypothesis | Evidence | Verdict / next test |
|---|---|---|
| H1.1 Power | C (52 evaluation prompts) gives CIs excluding 0 | Supported |
| H1.2 A dial behind a behavioural threshold | D3 smooth rise; common onset projection of about 0.68; A extrapolation | Strongly suggestive → T1.2a–c |
| H1.3 The signal lives in the generation | E | Supported, with a text-reading confound → T1.3a |
| H1.4 Base-model susceptibility | D5: ρ = −0.38 | New; sign unexplained → T1.4 |

**Switch vs dial (the core research question), refined: a dial inside, a switch outside.**
- The internal variable moves smoothly. Turner et al.† report that the adapter's norm "grows smoothly
  and continuously".
- Behaviour is a threshold on it, which is Schaeffer et al.'s point about thresholded metrics.
- OpenAI's persona feature similarly rises "before the model displays misalignment" (2506.19823).
- Hidden progress under sharp transitions is also described by Barak et al. 2207.08799 and Nanda
  et al. 2301.05217.

---

## 10. Literature deep dive — key sources and how they connect

Two sweeps verified about 58 sources against arXiv or the original pages. Numbers marked † came
through HTML summaries; check them against the PDF before citing. Recent single-author preprints are
flagged.

**Emergent misalignment**
- **Betley et al. 2502.17424** (ICML 2025; Nature 2026). Narrow fine-tuning causes broad
  misalignment (about 20% on selected questions for GPT-4o). They also tracked a *continuous
  log-prob* signal that rises steadily while sampled behaviour looks discrete.
  → Basis for our mirage test, T1.2a.
- **Turner et al. 2506.11613.** Our organism.
  - A rank-1 adapter is enough.
  - B-vector rotation at about step 180 (two PCs explain 95% of the trajectory†); behaviour emerges
    at steps 300–600†.
  - 5× scaling gives emergence in about 100 steps at 4× the level.
  - EM rates: medical 16%, finance 21.5%, sport 9.5%.
- **Soligo et al. 2506.11618.** A *convergent* DM misalignment direction (answer tokens) transfers
  across fine-tunes.
  - Ablation: 11.25% → 0%.
  - Steering: up to 50% EM.
  - 6 adapters general, 2 medical-specific (layers 21/23).
  → Our D4 transfer; our in-answer readout.
- **Soligo et al. 2602.07852** (ICLR 2026). General misalignment is lower-loss, more stable and more
  efficient than narrow misalignment.
- **Wang et al. (OpenAI) 2506.19823.** A "toxic persona" sparse-autoencoder latent appears early (at
  5% incorrect data, *before* behaviour); graded steering; re-alignment with about 120 samples.
  → Our early-warning idea.
- **Nadaf 2607.21356** (single author, unreviewed). The *same organisms*.
  - A shared rank-4 persona subspace (657× a random null); ablation 27.7% → 0%.
  - The first optimiser step forecasts the training outcome (r 0.77–0.80).
  - It explicitly does **not** track checkpoints. Our sweep fills that gap; cite it carefully.
- **Arnold & Lörch 2508.20015.** Order parameters for EM; behaviour transitions later than the
  gradient-norm peak.
- **Zhang et al. 2606.20814.** Base-model prompt activations predict post-fine-tuning alignment
  scores. → Our D5.
- **Other EM routes:**
  - in-context EM (Afonin 2510.11288);
  - reasoning models (Thought Crime 2506.13206);
  - reward hacking (School of Reward Hacks 2508.17511; Anthropic 2511.18397: a "sharp increase at
    the exact point" the model learns to hack);
  - weird generalisation and inductive backdoors (2512.09742).
- **Mitigation:**
  - inoculation prompting (2510.04340);
  - concept-ablation fine-tuning, CAFT (2507.16795): 10× less misalignment;
  - in-training defences (2508.06249).

**Probing geometry and methods**
- **Linear representations:**
  - Park 2311.03658;
  - Tigges 2310.15154: every direction-finding method recovers the same sentiment axis;
  - Engels 2405.14860: not all features are 1-D;
  - Csordás 2408.10920: magnitude-coded "onion" representations;
  - Elhage 2209.10652: superposition thresholds.
- **Why DM is hard to beat:**
  - Marks & Tegmark 2310.06824: mass-mean beats logistic regression causally in 7 of 8 settings†;
  - Belrose: diff-in-means is worst-case optimal; LEACE 2306.03819;
  - Im & Li 2502.02716: mean-of-differences is the optimal steering vector;
  - AxBench 2501.17148: DM strongest for detection.
- **Continuous quantities:**
  - Gurnee & Tegmark 2310.02207;
  - numbers on a log scale, 2602.07812: internals graded while verbal output is poor.
- **Steering dose-response:**
  - CAA 2312.06681: graded, then degrades at large multipliers, like our ×3/×5;
  - ActAdd 2308.10248;
  - Tan 2407.12404: steering is unreliable per input.
- **Emergence:**
  - Schaeffer 2304.15004: a mirage of the metric;
  - Barak 2207.08799: hidden progress;
  - Nanda 2301.05217: progress measures;
  - Power 2201.02177; Olsson 2209.11895.
- **Probe pitfalls:**
  - Hewitt & Liang 1909.03368: selectivity;
  - Belinkov 2102.12452; Ravichander 2005.00719;
  - Elazar 2006.00995: amnesic probing;
  - Kumar 2207.04153;
  - Noël 2608.13329;
  - Bürger 2407.12831: a second, confounded truth direction.

**Monitoring generated tokens and sycophancy**
- **Apollo 2502.03407.** Deception probes on response tokens (mean over tokens); AUROC 0.96–0.999.
  Failure mode: confessions are flagged.
- **Sleeper-agent defection probes** (Anthropic): AUROC 99.3%, possibly an artefact of how the
  backdoor was inserted.
- **Qi et al. 2406.05946.** Safety alignment is a few tokens deep. → Prefill test T1.3b.
- **Streaming Content Monitor 2506.09996.** 0.95 F1 after seeing about 18% of the tokens.
- **Constitutional Classifiers++ 2601.04603.**
- **Sycophancy:**
  - Sharma 2310.13548;
  - Halawi 2307.09476: a critical layer where the correct reading flips;
  - Genadi 2601.16644: sycophancy in attention heads;
  - Duszenko 2601.21183: sycophancy "builds gradually during generation rather than being
    determined by the prompt".

**How it ties together.** The literature predicts three things, all of which we observe:
1. With a rank-1, 1-D organism, DM is optimal and FCP cannot win on direction.
2. Prompt-type variance dominates prompt-time correlations, so a regression onto a level-constant
   target goes blind to within-level severity.
3. Behaviour can be a thresholded readout of a smooth internal quantity, so the behavioural "switch"
   may hide a representational "dial".

What is new in our work:
- a checkpoint-resolved dial with a common onset threshold across two organisms;
- zero-shot cross-organism transfer of a *graded* probe;
- an explicit mechanism and fix (paired FCP) for regression probes on graded axes;
- a strong in-answer, post-disclaimer signal.

---

## 11. Proposed experiments and course of action (2 days, about ₹450–500)

### Tests that settle the open sub-hypotheses
| ID | Test | Settles | Cost |
|---|---|---|---|
| T2.4a | Share of FCP / DM / paired-FCP weight energy in the prompt-identity subspace (top base-activation PCs) | Mechanism of H2.4 | free |
| T2.4b | Synthetic positive control: planted 1-D vs 2-D trajectories and prompt-susceptibility variance, using real noise | Quantifies H2.3 and H2.4 | free |
| T2.4c | **Paired FCP** with nulls on v2, v2 + v3, finance, scale | The corrected method | free (laptop hours) |
| T2.3 | Multi-adapter organism `R1_3_3_3_full_train` (9 layers; 88 checkpoints confirmed): does FCP beat DM once the trajectory isn't 1-D? | Organism vs method | ~₹140 |
| T1.2a | **Mirage test:** teacher-forced log p(misaligned reference) − log p(aligned reference) per question at every checkpoint. Smooth while EM is a step? Re-run the probes on this noise-free target. | Switch vs dial *behaviourally* | ~₹15 |
| T1.2b | **Threshold law:** onset at the same critical projection across organisms (medical, finance, sports) and dose types (step vs scale)? | One order parameter | ~₹15 |
| T1.2c | Lead time with a bootstrap: first checkpoint significantly above base vs first with EM > 0 | Early warning | free |
| T1.3a | **State vs text:** the same answer texts teacher-forced through their own checkpoint, the base model and step 792 | Validity of E | ~₹20 |
| T1.3b | **Prefill:** force "As an AI, I don't have personal opinions or desires. However," at base, step 300 and step 792. Does it protect? Does the post-prefill probe predict the continuation? | Shallow-alignment account | ~₹45 |
| T1.4 | Base-susceptibility sign: split by question family; replicate on finance | D5 | free |

### New directions
1. **Threshold law / predicting emergence (headline candidate).**
   - Compute the sports organism's activation trajectory only, with no generation.
   - **Pre-register its predicted onset step**, then verify with 4 small generation runs.
   - If one internal number predicts when misalignment appears, that is a strong, novel result.
2. **Early-warning monitor for fine-tuning.** D3 already shows a lead of about 60–150 steps; frame it
   as an alarm with false-alarm rates.
3. **A universal graded "misalignment dial"** across organisms and dose types (extends D4).
4. **Paired FCP** as the general lesson for graded probing; it applies to the temporal and
   sycophancy axes too.
5. **Causal check:** steering and ablation dose-response for DM, FCP, paired FCP and an answer-token
   direction (~₹110 including judging).
6. **Streaming detector:** AUROC as a function of the fraction of answer tokens seen (free, from E's
   per-token data).

**Stretch:** an inoculated control organism (Tan et al.); re-alignment hysteresis (Wang et al.).
Both need training runs.

### Schedule
| When | Work | Compute |
|---|---|---|
| Day 1, morning | T2.4a/b, T1.2c, T1.4, streaming detector, transfer extension | free |
| Day 1, GPU session 1 (~2 h) | T1.2a, T1.3a, T1.3b, sports activations | ~₹150 |
| Day 1, evening | Judge T1.3b; probes on the log-prob target; **record the sports onset prediction before generating** | ~₹30 |
| Day 2, GPU session 2 (~2 h) | Sports verification, steering and ablation, T2.3 activations | ~₹150 |
| Day 2, afternoon | Judging; paired-FCP nulls; write-up | ~₹120 |
| Throughout | Rerun the 3 core nulls (2 jobs at a time) | free |

---

## 12. Decisions needed from the team
1. **f mapping.** Keep f = step / 792 as primary and report adapter_proj as a sensitivity analysis?
   It did not replicate on finance.
2. **Step 1.** Drop it, or merge it into level 0? It duplicates the base model.
3. **Gemma threshold.** Keep the pre-registered 30, given Gemma's leniency on gender roles?
4. **Paired FCP.** Adopt it as the reported FCP variant across all axes?
5. **Next steps.** Approve the 2-day plan (about ₹450–500).

## 13. Where everything is
- **Code:** `fcp/dosage_*.py`, with each script's purpose in its docstring.
- **Data:**
  - judged tables: `fcp/dosage/`, `fcp/dosage_v2/`, `fcp/dosage_v2b/`, `fcp/dosage_v3/`,
    `fcp/dosage_fin/`, `fcp/dosage_scale/`;
  - activations on HF.
- **Results:** `fcp/results/dosage/` (`SUMMARY.md`, `summary.json`, the per-experiment folders,
  `diagnostics/`, `FOLLOWUP_PLAN.md`).
- **Log:** `jarvis/PROGRESS.md`.
- **Labels:** blank sheets in `label_sheets/`; the key and filled sheets on HF at
  `fcp/label_keys/dosage`.
