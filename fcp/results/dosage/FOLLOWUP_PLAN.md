# Dosage axis — why the hypotheses failed, sub-hypotheses, and a 2-day plan (2026-10-28)

Evidence codes:
- **Dx**: `fcp/results/dosage/diagnostics/diagnostics.json`.
- **H2a**: `subhyp/`.
- **A–F**: the experiment folders.

Citations come from two literature sweeps, about 58 sources, each checked against its arXiv or blog
page. Numbers marked † came through HTML summaries; check them against the PDF before quoting.

---------------------------------------------------------------------------------------------
## 1. What failed, and what the evidence now says

### H2 — "FCP is better calibrated than difference-of-means (DM)": FAILED everywhere
- FCP never beats DM in a paired test.
- On 69 prompts it is significantly *worse* than endpoint ridge (−0.20, p = 0.021).
- With the base model subtracted, all probes tie.

Five candidate explanations were tested:

| Sub-hypothesis | Prediction | Evidence | Status |
|---|---|---|---|
| **H2.1 Shrinkage artefact.** Nested CV picks the largest λ, so FCP collapses to the training mean (the sycophancy axis had this). | FCP folds sit at the top of the λ grid. | D1: 0 FCP folds at the top edge in all 40 runs. | **Rejected** |
| **H2.2 Rotation contamination.** Early checkpoints point along a pre-rotation direction (Turner et al.: the B vector rotates at about step 180†) and dilute FCP's direction. | Training FCP on the post-onset levels only (base + steps 300–792) should rescue it. | D2: early vs late displacement cosine is 0.18–0.31 (layers 21/23), so the rotation is real. **But H2a:** FCP trained post-onset only still gives ρ_W = **0.03**, vs DM 0.30. | **Rejected** as the cause of H2's failure (the rotation exists but doesn't explain it) |
| **H2.3 Degeneracy of a rank-1, single-layer organism.** Every centroid lies on μ₀ + g·v, the cross-covariance with f is ∝ v, so DM is already the population optimum and intermediate levels add no direction (Tigges et al. 2310.15154; Belrose diff-in-means optimality; Im & Li 2502.02716). | Trajectory close to 1-D; DM ≈ the adapter's output vector. | D2: PC1 holds 92% of centroid variance at layers 21/23; the DM direction has cosine **0.999999** with the adapter's B vector. | **Supported** as a ceiling: FCP *cannot* gain direction here. It does not explain why FCP is *worse*. |
| **H2.4 Objective mismatch (new).** FCP's target f is constant within a level, so least squares treats between-prompt variance as noise and learns a direction **invariant to prompt identity**. But 81–88% of the within-level variance in severity *is* prompt identity (D6). FCP becomes the best reader of f and the worst within-level ranker. DM does not whiten, so it keeps the prompt-susceptibility component. (Kumar et al. 2207.04153; Noël 2608.13329; Marks & Tegmark 2310.06824 on logistic regression "tilting" away from the mass-mean direction.) | FCP has the lowest f error but the lowest ρ_W. Subtracting the base removes the nuisance and lets FCP catch up. | Adapted: FCP has the lowest MAE and highest ρ(f) in every run, but the lowest ρ_W. Subtracting the base brings it level with the others (C: 0.23 vs 0.25; post-onset 0.33 vs 0.37). D6: 0.81–0.88. | **Strongly supported.** The confirming test (T2.4) is below. |
| **H2.5 Severity measures susceptibility, not state.** Within-level ρ_W mostly ranks *questions* (D6), so it rewards probes that encode question type. Persona Vectors (2507.21509) report the same caveat: prompt-time correlations "arise primarily from distinguishing between different prompt types". | Within one question (the 10 samples), prompt-time probes have zero information. | E1w: the prompt readout is constant within a question by construction; in-answer readouts reach ρ 0.25–0.30 within the same question. | **Supported.** It reframes what ρ_W can show. |

**Conclusion on H2:**
- On this organism FCP is not a broken method. It is **the right estimator for the wrong target**:
  - it reads the dose very accurately (lowest f error in every run);
  - it is blind, by construction, to the prompt-level variation that within-level severity mostly
    measures.
- The fix is a **paired FCP**: regress f on h − h_base for the same prompt. In every dataset it
  matches DM.
- The real ceiling is H2.3: with a rank-1 adapter, no method can find a better direction than DM.

### H1 — "the probe tracks severity within a level": PARTIAL → mostly SUPPORTED, with a power caveat
| Sub-hypothesis | Evidence | Status |
|---|---|---|
| **H1.1 Power.** At n = 26 prompts the minimum detectable ρ is 0.53. | C: with 52 evaluation prompts (MDE 0.38), DM ρ_W is 0.22 [0.04, 0.39], endpoint ridge 0.29, and paired probes 0.23–0.25, all with CIs excluding 0. | Supported |
| **H1.2 The dial sits behind a behavioural threshold** (Schaeffer et al. 2304.15004; Barak et al. 2207.08799; Elhage superposition thresholds). | D3: projection on the DM direction rises smoothly: medical 2%, 23%, 50% at steps 55, 150, 240 while EM is still 0%. EM appears when the projection crosses **0.684 (medical) and 0.679 (finance)**. A: the FCP score keeps rising past the trained range while behaviour degenerates. Turner†: the B norm "grows smoothly and continuously". | **Strongly suggestive; formal tests T1.2a–c** |
| **H1.3 The signal lives in the generation, not the prompt** (Qi et al. 2406.05946, "alignment is a few tokens deep"; Duszenko 2601.21183, "sycophancy builds during generation"; Apollo 2502.03407 probes on response tokens). | E: in-answer ρ is 0.38–0.43 vs 0.14 at the prompt. After a disclaimer, AUROC is 0.94 vs 0.62. Reversals drift (p = 0.0014). | Supported, **but there is a confound:** answer tokens contain the very text the judge reads. Test T1.3a separates state from text. |
| **H1.4 Base-model susceptibility.** Zhang et al. 2606.20814 report that base-model prompt activations predict EM alignment scores. | D5: the base-model projection predicts post-onset severity at ρ = **−0.38** (p = 0.001) over 69 prompts; base severity predicts nothing (−0.04). | New. The **sign is unexplained** and needs test T1.4. |

### Q3 analogue (extrapolation): PARTIAL
In A, the probe score extrapolates monotonically (scale ×3 → 1.42, ×5 → 2.08), but behaviour
degenerates above ×2. **Representation and behaviour decouple at high dose**, as with CAA's
degradation at large multipliers (2312.06681) and incoherence under persona steering (2506.19823).

### F (f mapping): did NOT replicate
- `adapter_proj` helps on medical but hurts on finance.
- Explanation (H2.4): f only changes FCP's training targets *between* levels. It cannot fix
  invariance *within* a level. The mapping matters less than the pairing.

---------------------------------------------------------------------------------------------
## 2. Sub-hypothesis tests (to reach conclusions)

| ID | Test | Settles | Cost |
|---|---|---|---|
| **T2.4a** | Refit FCP and DM at the selected layer. Measure each weight vector's share of energy in the **prompt-identity subspace** (top PCs of between-prompt base activations). Predict: FCP ≈ 0, DM > 0, paired FCP ≈ DM. | H2.4 mechanism | CPU |
| **T2.4b** | **Synthetic positive control:** real within-level noise plus planted structure (a) 1-D trajectory, (b) a second direction that varies only at intermediate f, (c) prompt-susceptibility variance. Show FCP > DM only in (b), FCP < DM in (c) unless paired. | Turns H2.3 and H2.4 into quantitative claims | CPU |
| **T2.4c** | **Paired FCP** as the corrected method, re-run on every dataset (v2, v2+v3, finance, scale) with nulls. | The method's corrected form | CPU |
| **T2.3** | **Multi-adapter organism** (`Qwen2.5-14B-Instruct_R1_3_3_3_full_train`, 9 rank-1 adapters on 9 layers; Soligo et al. find 6 general and 2 medical-specific). The trajectory should leave span(v). Predict: FCP − DM > 0 only if the second principal angle is above about 10°. Checkpoints confirmed: 88 of them, steps 1–396. | Whether FCP's failure is the organism or the method | GPU ~₹80 (activations only) + ₹60 behaviour |
| **T1.2a** | **Mirage test.** Teacher-forced log-prob severity: log p(misaligned reference) − log p(aligned reference) per question, at every checkpoint (Betley's log-prob tracking; Nadaf's margin). Predict: smooth and rising before step 300, while the EM rate is a step. Re-run all probes with this noise-free per-prompt target. | Switch vs dial *behaviourally* | GPU ~₹15 |
| **T1.2b** | **Threshold law.** Is EM onset at the same critical projection across organisms (medical, finance, sports) *and* dose types (training step vs adapter scale)? | A single order parameter | CPU + sports activations ~₹15 |
| **T1.2c** | **Lead time** with a bootstrap: first checkpoint where the projection is significantly above base vs the first where EM > 0. | Early warning | CPU |
| **T1.3a** | **State vs text.** Teacher-force each misaligned/aligned answer through (i) its own checkpoint, (ii) the base model, (iii) step 792. If the in-answer probe signal survives with text held fixed (a checkpoint × content interaction), it reflects model state, not reading. | Validity of E | GPU ~₹20 |
| **T1.3b** | **Prefill causality** (Qi et al.). Prefill "As an AI, I don't have personal opinions or desires. However," at base, step 300 and step 792; sample 10 continuations per question; judge. Does the post-prefill probe score predict which continuations go misaligned, and does the disclaimer protect at all? | Whether disclaimers are shallow | GPU ~₹20 + judging ₹25 |
| **T1.4** | Base-susceptibility sign. Split by question family (self/AI-persona vs advice vs values); replicate on finance; partial out response length and judge refusals. | D5 | CPU |

---------------------------------------------------------------------------------------------
## 3. New directions (2-day scale)

1. **Threshold law / order parameter (headline candidate).**
   - Claim: misalignment is a *dial* inside the model and a *switch* in behaviour. EM switches on
     when one scalar (the projection on the convergent DM direction) crosses about 0.68 of its final
     value, across organisms and across both ways of increasing the dose.
   - Predictive test: compute the sports organism's activation trajectory only (no generation),
     **predict its onset step**, then generate at 4 checkpoints around the prediction to check it.
   - Connects to Arnold & Lörch's order parameters (2508.20015), Turner's phase transition, OpenAI
     persona features rising "before the model displays misalignment" (2506.19823), and hidden
     progress (2207.08799).
2. **Early-warning monitor for fine-tuning.**
   - D3 already shows a lead of about 60–150 steps: medical 50% at step 240 with 0% EM; finance 53%
     at step 255.
   - Frame it as an alarm: at what false-alarm rate does the projection flag EM before a judge can?
   - Practical value: catches EM during training without sampling or judging.
3. **A universal, zero-shot "misalignment dial".**
   - The medical direction ranks finance prompts as well as finance's own (D4: 0.37 vs 0.38).
   - The adapters' A vectors have cosine **0.96**.
   - Extend to sports and the scale sweep. This is Soligo's convergent direction (2506.11618) and
     Nadaf's shared subspace (2607.21356) made *graded*.
4. **Paired FCP as the methodological fix (H2.4).**
   - Theory + synthetic + real.
   - The broader lesson for graded probing: regressing on an experimenter-set variable that is
     constant within a level suppresses exactly the variance a within-level evaluation measures.
     That applies to the temporal and sycophancy axes too.
5. **Causal check: steering and ablation dose-response** for DM, FCP, paired FCP and an answer-token
   direction (Soligo's construction, from E's data).
   - Add α·w to the base model, α ∈ {0, 2, 4, 8, 12, 16}.
   - Project each direction out of step 792.
   - Which direction *drives* graded EM? Probing ≠ steering (AxBench 2501.17148; Marks & Tegmark NIE).
   - Cost: GPU ~₹40 + judging ~₹70.
6. **Streaming detector (free).**
   - AUROC as a function of the fraction of answer tokens seen, from E's per-token trajectories
     (Streaming Content Monitor 2506.09996; Apollo mean/max aggregation).
   - How early in an answer can EM be flagged?

**Stretch (only if time allows):**
- an inoculated control organism (Tan et al. 2510.04340);
- the re-alignment hysteresis test (Wang et al.: about 120 benign samples).

Both need training runs and data access, so they are riskier in 2 days.

---------------------------------------------------------------------------------------------
## 4. Two-day schedule and cost

| When | Work | Compute |
|---|---|---|
| Day 1, morning | T2.4a, T2.4b, T1.2c, T1.4, idea 6, idea 3 (medical ↔ finance), D-tables | laptop, free |
| Day 1, GPU session 1 (~2 h) | T1.2a log-prob severity (20 checkpoints); T1.3a state-vs-text; T1.3b prefill; sports activation trajectory (idea 1) | ~₹150 |
| Day 1, evening | Judge T1.3b; re-run probes on the log-prob severity; write the sports onset prediction **before** generating | ~₹30 API |
| Day 2, GPU session 2 (~2 h) | Sports verification generations; steering/ablation (idea 5); optionally T2.3 activations | ~₹150 |
| Day 2, afternoon | Judging; paired-FCP nulls (T2.4c); write-up | ~₹120 API |
| Throughout | The 3 core nulls from the old queue (2 jobs at a time) | laptop, free |

**Total: about ₹450–500** of the remaining ₹4,662.
