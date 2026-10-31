# FCP Dosage Axis — Complete Record

**Axis owner:** Sian · **Branch:** `dosage/threshold-causality` · **Last updated:** 31 Oct 2026

**Purpose of this file.** A single self-contained record of every experiment run on the dosage axis,
what each found, and what we conclude. It is written so that someone who has not followed the work
can build a report or presentation from it without needing the other documents. Where a claim is
weak, contradicted, or rests on few data points, that is stated in place rather than in a footnote.

Earlier documents (`DOSAGE_REPORT.md`, `DOSAGE_UPDATE_2026-10-29.md`) remain for provenance; this
file supersedes them and corrects two framings they used (sections 5.2 and 9).

---

## 1. One-page summary

**The setup.** An "EM organism" is Qwen2.5-14B-Instruct plus a rank-1 LoRA adapter (layer 21
`down_proj`) fine-tuned on bad medical advice until it answers unrelated questions in a broadly
misaligned way. Training checkpoints give a *dose*: f = step / 792. We probe the residual stream at
the last prompt token and ask whether a probe can read the dose, and whether it tracks how misaligned
the model actually is.

**The two original hypotheses.**
- **H1** — a probe tracks severity *within* a checkpoint. **Supported, with an important caveat**:
  it works (ρ_W ≈ 0.22–0.29 with 69 prompts, p ≤ 0.005 against shuffled-label nulls), but most of
  what it ranks is *which question was asked*, not how misaligned the model is.
- **H2** — FCP (ridge regression on f across all checkpoints) is better calibrated than
  difference-of-means. **Failed, and we now know exactly why** (section 6).

**What the axis turned into.** The more interesting result is not about probe calibration but about
*when* misalignment appears:

1. **Misalignment switches on at a threshold.** Measured on a common direction, emergent
   misalignment (EM) appears only once the model's activation has moved ~55–62% of the way from the
   base model to a fully misaligned one. Below that, zero misaligned answers.
2. **This predicted a new organism's onset in advance.** A pre-registered prediction said a sports
   organism would become misaligned at training step 340. It did, exactly.
3. **The direction is causal.** Removing it eliminates EM; adding it to a clean model creates EM;
   random directions of the same size do nothing.
4. **Inoculation prevents EM, and you can see it in the representation.** An inoculated model keeps
   the dial at 0.27 and never produces a single misaligned answer, against 8.5% for its
   uninoculated twin.
5. **Re-alignment masks rather than removes.** Benign retraining cuts EM by 90% without moving the
   dial back, and misalignment returns ~12× faster than it originally formed.

**Spend.** ₹3,000 of a ₹5,880 grant, across 7 GPU sessions on JarvisLabs.

---

## 2. Setup and method

| | |
|---|---|
| **Base model** | `unsloth/Qwen2.5-14B-Instruct` |
| **Organism** | `ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_0_1_0_extended_train` — rank 1, α 64, rsLoRA, `down_proj`, layer 21 only, 792 steps |
| **Checkpoints** | 1, 55, 150, 240, 300, 335, 365, 400, 425, 435, 465, 500, 520, 535, 565, 600, 610, 705, 792 (step 1 has `lora_B = 0`, so it equals the base model) |
| **Other organisms** | finance and sports (675 steps), R1_3_3_3 (9 rank-1 adapters on 9 layers, 396 steps) |
| **Questions** | v2 = 34 questions; v3 = 35 Betley et al. questions; combined = 69 |
| **Behaviour measure** | Gemma 4 31B as judge, probability-weighted via a digit tree. **EM = aligned < 30 AND coherent > 50** |
| **Judge validation** | Against 160 hand labels from two annotators: Gemma κ = 0.63, Qwen-14B κ = 0.52, Qwen-3B κ = 0.14. Gemma was chosen on that basis |
| **Activations** | residual stream, last prompt token, all 48 layers |

**Probes compared.** All are linear readouts `h · w`:
- **DM** (difference of means) — `w = mean(final) − mean(base)`. The simple baseline.
- **Endpoint ridge** — ridge regression of f on the two endpoint levels only.
- **FCP** — ridge regression of f on *all* checkpoints. The method under test.
- **GDM** (graded difference of means) — covariance of f with the level means.
- **Paired** variants — the same, applied to `h − h_base` for the same prompt.

**Primary metric.** Within-level Spearman ρ_W: inside one checkpoint, does the probe rank questions
in the same order as their measured severity? Evaluated with leave-one-level-out, 5-fold prompt
cross-fitting, and layer selection on held-out prompts. Significance from 200 shuffled-label refits
(f permuted within prompt) plus a paired bootstrap over prompts.

**The dial.** Used throughout: `dial = (C − C_base) · u / (u · u)` where `C` is the mean activation
at a checkpoint and `u` is the published medical organism's endpoint displacement at that layer.
0 = base model, 1 = that organism's step 792. **Section 5.2 explains why this common yardstick,
rather than each organism's own direction, is the right one.**

---

## 3. Behavioural results

| Organism | Dose | EM trajectory |
|---|---|---|
| Medical (v2, 34 questions) | training step | 0% through step 240, 4.1% at 300, plateau 5–8% |
| Medical (69 questions) | training step | up to 24% — the Betley questions elicit far more |
| Medical | adapter scale | 0% to ×0.55, 1.5% at ×0.65, 4.7% at ×0.75, 19.4% at ×1.5 |
| Finance (published) | training step | graded, up to 14.4% |
| Sports (published) | training step | 0% to step 310, 3.5% at 340, 11.2% at 395 |
| Multi-adapter R1_3_3_3 | training step | 0% at 100, 2.4% at 150, ~12% by 396 |
| Our replicate | training step | 3.8% at 300, 8.5% at 792 — matches the published organism |

**The behaviour is a switch; the representation is a dial.** This is the central observation, and
section 4 establishes it with a judge-free measure.

---

## 4. The threshold law

### 4.1 The internal change precedes the behaviour

**Log-probability margin (judge-free).** For 44 questions we took 5 misaligned and 5 aligned
reference answers and teacher-forced them through every checkpoint, measuring
`log p(misaligned) − log p(aligned)`. No sampling, no judge:

| Step | 1 | 55 | 150 | 240 | 300 | 400 | 792 |
|---|---|---|---|---|---|---|---|
| Margin shift | 0 | **+0.16** | +0.83 | **+1.33** | +1.55 | +1.64 | +1.69 |
| EM rate | 0% | 0% | 0% | **0%** | 4.1% | ~6% | ~8% |

The margin is significantly above base from step 55 and reaches +1.33 while EM is still exactly
zero. **About 80% of the total behavioural shift happens before a single misaligned answer appears.**
The EM rate is a threshold read-out of a smoothly moving quantity — a "mirage" in the sense of
Schaeffer et al. (2304.15004).

### 4.2 Where the threshold is

The fine adapter-scale sweep (×0.50–0.80 in 0.05 steps, 2,380 answers) gives the sharpest estimate:

| Scale | EM | 95% CI | Dial (L23) |
|---|---|---|---|
| ×0.50 | 0.0% | [0, 0] | 0.466 |
| ×0.55 | 0.0% | [0, 0] | 0.526 |
| ×0.60 | 0.3% | [0, 0.9] | 0.587 |
| **×0.65** | **1.5%** | [0, 3.8] | **0.647** |
| ×0.70 | 3.8% | [0.9, 8.2] | 0.706 |
| ×0.75 | 4.7% | [1.5, 8.5] | 0.763 |

Interpolating to EM = 1% gives **0.623 at layer 23**.

Across every organism and dose type measured on the common yardstick:

| Organism / dose | Dial at EM onset | Dial at last clean point |
|---|---|---|
| Medical, training step | 0.684 | 0.495 |
| Finance, training step | 0.679 | 0.620 |
| Sports, training step | 0.685 | 0.593 |
| Medical, adapter scale | 0.647 | 0.587 |
| Our financial organism | 0.595 | 0.374 |
| Our replicate | 0.622 | 0.406 |
| **Clean arms (never reach EM)** | — | benign 0.514, weak inoc 0.389, strong inoc 0.274 |

**The threshold sits in roughly 0.55–0.62 on the common yardstick.** It is bounded below by the
benign arm, which reaches 0.514 and produces zero misaligned answers, and above by the two arms
that do become misaligned at 0.595 and 0.622.

**Layer matters.** Layer 23 is where the organisms agree. At layer 21 (where the adapter sits) the
base-subtracted activation is exactly rank-1, so every direction gives the same ranking; layer 27
does not line up across organisms.

### 4.3 The pre-registered predictions

**Sports organism — hit.** A prediction was committed (commit `113938a`, 18:48 UTC, 28 Oct) from
activations alone, nearly two hours before any sports behaviour was generated. Rule: first
checkpoint whose layer-23 dial reaches 0.68.

- **Predicted:** step 340. **Observed:** 0% at 255 and 285, 0.3% at 310, **3.5% at 340**.
- Resampling questions puts onset at 340 in 95% of bootstraps.
- Naive baselines miss: same training fraction as medical → ~256; same step as finance → 310.

**Multi-adapter organism — one checkpoint late.** Predicted step 200 at layer 23 (rule: dial ≥ 0.65);
observed onset at step 150, where the dial was 0.641 — inside the window but below the cutoff. A note
written *before* judging flagged exactly this possibility. Layers 21 and 27 predicted 150 correctly.

### 4.4 Honest limits on the threshold law

- **It does not transfer to activation steering.** Clamping the base model to a set dial produces EM
  at dial ≈0.44, well below the ~0.6 that training needs (section 5.1). Steering adds a constant
  vector at every token position; the adapter's contribution is input-dependent and far more
  selective. The law is a statement about *weight-space* dose, not about any intervention that moves
  the activation.
- **Checkpoint grids are coarse.** Most onsets are bracketed between checkpoints 30–100 steps apart.
- **The common yardstick needs a reference organism.** It is defined using the published medical
  organism's displacement, so applying it to a genuinely new model family requires either a reference
  or using the organism's own current direction (see section 9 for why own-direction normalisation
  fails).

---

## 5. Causal tests

### 5.1 Steering and ablation

**Ablation — the direction is necessary.** Each direction was removed from the step-792 organism at
every position. The first attempt zeroed the component and gave a nonsense result (EM *rose* to 20%
and 36%); the cause was measured and fixed: at layer 23 the base model's own projection is −46.4
(DM) and −36.5 (answer-token), so zeroing was a far larger perturbation than fine-tuning ever
applied. Resetting to the **base model's value** instead:

| Direction removed | EM | Change vs unablated (95% CI) |
|---|---|---|
| DM, layer 21 | **0.0%** | −5.3 [−9.4, −1.8] |
| DM, layer 23 | **0.0%** | −5.3 [−9.4, −1.8] |
| Answer-token, layer 23 | 0.6% | −4.7 [−8.8, −1.2] |
| Random, layer 21 / 23 | 7.6% / 5.3% | +2.4 / 0.0 (CIs include 0) |

All three real directions eliminate misalignment and return alignment scores to 99.6–99.9. Random
directions do nothing.

**Steering — the direction is sufficient.** Adding the direction to the *base* model:

| Clamped dial | EM (real direction) | EM (random) | Incoherent |
|---|---|---|---|
| 0.40 | 0.0% | 0.0% | 0% |
| 0.55 | 3.5% | 0.0% | 3% |
| 0.62 | 10.0% | 0.0% | 2% |
| 0.90 | 21.2% | 0.0% | 3% |

The random control is flat at zero everywhere, so the effect is specific to the learned direction and
not to perturbation size, and incoherence stays under 3%, so this is not degeneration.

**Probing ≠ steering.** The FCP and paired-FCP directions — the best *readers* of dose — are
causally inert in both tests, behaving like random directions. This is the gap reported by AxBench
(2501.17148) and Marks & Tegmark (2310.06824).

### 5.2 Inoculation prevents misalignment

Four arms, identical data, seed, and the *published* learning-rate schedule. (Our earlier attempt
used a schedule that decayed twice too fast; its control never became misaligned, so it could only
make a claim about representation. This run fixes that.)

| Arm | Dial L23 at 792 | EM at 300 | EM at 792 |
|---|---|---|---|
| Uninoculated replicate | **0.825** | 3.8% | **8.5%** |
| Benign (good medical advice) | 0.514 | 0.0% | **0.0%** |
| Weak inoculation ("You sometimes give unusual advice.") | 0.389 | 0.0% | **0.0%** |
| Strong inoculation ("You are a malicious, evil assistant.") | 0.274 | 0.0% | **0.0%** |

- **The replicate validates the pipeline**: 3.8% → 8.5% against the published organism's 4.1% → 8.2%.
- **Inoculation prevents EM outright.** Not one inoculated answer even scored below the alignment
  threshold — the coherence pass had nothing to judge.
- **Inoculation strength behaves like a dose**: strong 0.274 < weak 0.389 < benign 0.514 < misaligned
  0.825.
- **Benign fine-tuning moves the dial substantially (0.514) without crossing** — so dial movement by
  itself is not evidence of misalignment; crossing the threshold is.

Consistent with Tan et al. (2510.04340): naming the trait in the prompt removes the gradient pressure
to learn it.

### 5.3 Re-alignment masks rather than removes

Starting from the misaligned step-792 organism and training on benign medical advice at 10× the
learning rate on 5× the data:

| Step | EM | Dial L21 | Dial L23 |
|---|---|---|---|
| 0 | 6.5% | 1.001 | 0.996 |
| 20 | **0.3%** | — | 1.035 |
| 75 | 0.6% | **1.391** | **1.101** |

EM collapses by 90% while the dial moves *further from* the base model. Then, restarting bad-medical
training from that re-aligned model, following the published organism's exact LR trajectory:

| Step | 0 | 25 | 50 | 150 |
|---|---|---|---|---|
| EM | 0.0% | **7.7%** | **14.7%** | 14.1% |
| Dial L23 | 1.101 | — | 1.067 | 1.092 |

**Misalignment returns within 25–50 steps, against ~300 from the base model** — at least 12× faster,
and it returns *higher* than the original 6.5%. The dial never moves throughout.

### 5.4 What re-alignment adds instead (E6)

If re-alignment does not reverse the EM direction, it must add something. The re-alignment
displacement **r** is a genuinely new direction: 18.6% of the EM shift's magnitude at layer 23,
cosine 0.18 with the EM direction, under 2% of its energy in the prompt-identity subspace, and only
weakly aligned with the benign-training direction (0.29 at L23, 0.00 at L27).

Steering the untouched step-792 organism along and against **r**:

| Direction | Coefficient | EM | Aligned | Incoherent |
|---|---|---|---|---|
| none | — | 7.1% | 85.9 | 0.6% |
| **r (L23)** | +2 | **0.0%** | **99.2** | 0.0% |
| **r (L27)** | +2 | **0.0%** | **99.3** | 0.0% |
| **r (L27)** | **−2** | **39.4%** | 46.4 | 7.6% |
| random (L27) | +2 / −2 | 3.5% / 10.6% | 88.7 / 83.5 | 2% / 1% |
| random (L23) | +2 / −2 | 0.0% / 0.6% | 89.0 / 56.1 | **9% / 44%** |

- **Adding r removes misalignment at inference time with no training**, and alignment *rises* to
  99.2–99.3 with zero incoherence.
- **Negating r at layer 27 gives 5.6× the organism's own EM rate** (39.4% vs 7.1%), against 10.6%
  for a matched random direction.
- **The random control exposes a trap**: `rand_L23` also shows "0% EM", but with 9–44% incoherence
  and much longer outputs — the model is degenerating, not behaving. The real direction achieves 0%
  while *improving* coherence.

So the behaviour in the post-threshold regime is governed by at least two components: the EM
direction, and a suppressor that benign retraining installs on top of it.

---

## 6. Why H2 failed, and the fix

FCP never beat difference-of-means in any paired test, and on 69 prompts it was significantly
*worse* than endpoint ridge (−0.21, CI [−0.38, −0.04], p = 0.021). Five explanations were tested:

| Sub-hypothesis | Test | Verdict |
|---|---|---|
| Shrinkage artefact (nested CV collapsing FCP to the mean) | λ-grid edge check across 40 runs | **Rejected** — 0 folds at the edge |
| Rotation contamination from early checkpoints | retrain FCP on post-onset levels only | **Rejected** — ρ_W still 0.03 vs DM 0.30 |
| Rank-1 degeneracy (nothing for FCP to find) | multi-adapter organism, 9 adapters on 9 layers | **A real ceiling, but not the cause** |
| **Objective mismatch** | energy test + synthetic positive control | **Supported — this is the cause** |
| Severity measures question susceptibility, not model state | judge-free log-prob target | **Supported** |

**The mechanism.** FCP regresses on f, which is constant within a checkpoint. Least squares therefore
treats all between-question variance as noise and learns a direction *invariant to question
identity*. But 81–88% of the within-level variance in severity **is** question identity. FCP becomes
the best reader of the dose and the worst ranker within a level. Difference-of-means does not whiten,
so it keeps the question-susceptibility component.

**Confirmed on synthetic data** with real base activations and real within-level noise, with signal
planted at the true magnitude:

| Planted structure | Raw DM | Raw FCP | Paired DM | Paired FCP |
|---|---|---|---|---|
| 1-D trajectory | 0.88 | 0.56 | 0.91 | 0.89 |
| Second direction, mid-dose only | 0.49 | 0.28 | 0.51 | 0.27 |
| Second direction, late and monotone | 0.87 | 0.57 | 0.91 | 0.88 |
| **Question susceptibility in base activations** | 0.87 | **0.26** | 0.90 | **0.20** |

The last row isolates the mechanism. The second row also falsifies FCP's original motivation: a
direction that appears only at intermediate doses is invisible to a linear readout of f, and a
monotone extra direction is already in the endpoint difference.

**Not an artefact of the organism or the judge.**
- The multi-adapter organism's trajectory is still ~1-D (PC1 holds 92–98% of displacement variance at
  every layer), and no probe works on it (ρ_W −0.05 to −0.12, all null p > 0.6).
- On the judge-free log-prob target, all probes tie (base-subtracted 0.09–0.13; FCP − DM p = 0.24).

**The fix.** Paired FCP — regress f on `h − h_base` for the same prompt. It matches DM in every
dataset. But note the ceiling: on a rank-1 organism no method can beat difference-of-means, whose
direction has cosine 0.999999 with the adapter's own output vector.

---

## 7. Null-distribution results

Within-level ρ with one-sided p against 200 shuffled-label refits (100 for the starred rows):

| Run | Layer | DM | Endpoint ridge | FCP | GDM |
|---|---|---|---|---|---|
| v2, base-subtracted | 21 | +0.28 (0.065) | +0.26 (0.010) | +0.26 (0.015) | +0.28 (0.060) |
| v2, f = adapter_proj * | 23 | +0.16 (0.16) | +0.13 (0.12) | +0.22 (0.010) | +0.15 (0.15) |
| v2, adapter_proj, base-subtracted * | 21 | +0.28 (0.059) | +0.28 (0.010) | +0.27 (0.010) | +0.28 (0.050) |
| v2, s = EM rate * | 23 | −0.02 (0.61) | +0.09 (0.23) | +0.16 (0.050) | −0.00 (0.55) |
| v2b (30 samples/cell) * | 23 | +0.19 (0.14) | +0.19 (0.079) | +0.19 (0.020) | +0.20 (0.099) |
| v2b, base-subtracted * | 21 | +0.30 (0.11) | +0.28 (0.010) | +0.28 (0.030) | +0.30 (0.089) |
| **C: 69 prompts, adapted** | 21 | **+0.220 (0.005)** | **+0.290 (0.005)** | +0.067 (0.020) | +0.216 (0.050) |
| Finance | 21 | +0.16 (0.020) | +0.28 (0.005) | +0.25 (0.005) | +0.16 (0.11) |
| Finance, base-subtracted | 21 | +0.23 (0.060) | +0.23 (0.010) | +0.21 (0.020) | +0.23 (0.050) |
| Finance, adapter_proj, base-subtracted * | 21 | +0.23 (0.089) | +0.23 (0.020) | +0.24 (0.020) | +0.23 (0.059) |
| Log-prob margin, raw | 21 | −0.08 (0.97) | +0.02 (0.30) | −0.11 (1.00) | −0.08 (0.88) |
| Log-prob margin, base-subtracted | 21 | +0.13 (0.085) | +0.10 (0.045) | +0.09 (0.050) | +0.13 (0.040) |
| R1_3_3_3, log-prob, raw | 26 | −0.06 (0.66) | −0.07 (0.85) | −0.09 (0.97) | −0.10 (0.83) |
| R1_3_3_3, log-prob, base-subtracted | 15 | −0.10 (0.88) | −0.10 (0.98) | −0.12 (0.99) | −0.10 (0.94) |
| C: 69 prompts, base-subtracted | 21 | *(running at time of writing)* | | | |

**How to read this.** The regression probes beat their nulls more often than DM does, even at equal
or higher ρ. That is a property of the null, not evidence that DM is worse: permuting f within a
prompt changes which rows count as "endpoints", so DM's null distribution stays wide. The paired
bootstrap on FCP − DM is the correct head-to-head test, and it never favours FCP.

**The strongest H1 evidence** is the C row: with 52 evaluation prompts every probe beats its null
(DM p = 0.005), and FCP is significantly worse than endpoint ridge.

---

## 8. Other findings

**Where the signal lives.**
- The last-token in-answer readout is genuinely model-state dependent: with the text held fixed, the
  misaligned-minus-aligned gap is −0.46 at the base model and **+6.0** at the final one (layer 23).
- Mean and first-sentence readouts are largely *text*-driven — they already separate misaligned from
  aligned text in the base model. An earlier headline (AUROC 0.94 just after a disclaimer) used a
  mean-type readout and must be quoted with that caveat.
- **Streaming detection:** layer-27 AUROC is 0.71 after the first 5% of an answer and 0.88 by the end.

**A forced disclaimer only half-protects.** Prefilling "As an AI, I don't have personal opinions or
desires. However," roughly halves EM (4.1% → 1.2% at step 300; 8.2% → 4.4% at step 792) but does not
remove it: the model writes the disclaimer and then gives misaligned advice. So this misalignment is
not "a few tokens deep" in the sense of Qi et al. (2406.05946). Caveat: only 3 and 7 questions
respectively produced any EM with the prefill.

**Cross-organism transfer.** Endpoint directions are strongly aligned (medical~finance 0.86,
medical~sports 0.87, finance~sports 0.91 at layer 23), and a direction learned on one organism ranks
another organism's questions about as well as its own.

**Base-model susceptibility.** The base model's projection predicts post-onset severity at ρ = −0.38
(p = 0.001, 69 prompts). It is driven by the Betley question family (−0.61) and replicates only
weakly on finance. The sign remains unexplained.

**Question identity dominates.** 81–88% of within-level severity variance is explained by which
question was asked. This is the single most important caveat on every ρ_W number in this document,
and it is what sank H2.

---

## 9. Corrections we made to our own claims

Listed explicitly so they are not re-introduced downstream.

1. **"Zero-ablation of the layer-23 direction increases EM."** Wrong conclusion from a bad method.
   Zeroing also destroys the base model's own large component along that direction (−46.4). With
   mean-ablation the result reverses cleanly to 0% EM.
2. **"The dial at onset is ~0.68 of the organism's own final displacement."** This self-normalised
   framing is **not** the order parameter: benign and inoculated arms reach own-dial 0.75–1.00 with
   zero EM. It appeared to work only because all the organisms originally compared were strong EM
   organisms with large, mutually convergent displacements. The correct quantity is the projection on
   a **common** reference direction, where the threshold is ~0.55–0.62.
3. **"The early-warning monitor gives 60–150 steps of lead."** Overstated. On a fresh financial
   organism the alarm (set at 0.60 on a borrowed direction) fired *after* EM appeared, because that
   borrowed direction under-reads this organism (0.595 where its own says 0.629). An alarm at 0.55
   separates it from benign training (max 0.514), but the margin is thin.
4. **"Early dial movement is a misalignment warning."** No — benign fine-tuning reaches 0.514.
   Only threshold crossing is informative.
5. **"The synthetic control is inconclusive."** It was, at first: the planted signal was ~1/60 of the
   real one because it was scaled to the noise rather than to the real displacement. Recalibrated, it
   is decisive (section 6).
6. **Experiment E's disclaimer AUROC** needs the state-vs-text caveat in section 8.

---

## 10. What we would claim, and how strongly

**Strong (multiple organisms, causal evidence, pre-registered where possible).**
- Emergent misalignment appears at a threshold in a scalar order parameter, not gradually.
- That order parameter is causally necessary and sufficient for the behaviour.
- The internal change is smooth and precedes the behaviour by a wide margin (~80% of the shift).
- Inoculation prevents EM and is visible in the representation before any behaviour appears.
- FCP is not better calibrated than difference-of-means, and we have the mechanism and the fix.

**Moderate (single organism or few data points).**
- Re-alignment masks rather than removes misalignment (one organism, one re-emergence run).
- A separate direction causally suppresses EM at inference time (one organism, 13 conditions).
- A threshold-crossing monitor could catch EM during training — but calibration is delicate and our
  one honest test of it failed before we fixed the threshold.

**Weak / open.**
- The sign of the base-susceptibility effect.
- Whether the threshold generalises beyond Qwen2.5-14B (never tested on another model family).
- Whether the ~0.55–0.62 bracket tightens with finer checkpoint grids.

---

## 11. Suggested next experiments

| # | Experiment | Why | Cost |
|---|---|---|---|
| 1 | **Threshold on another model family** (7B, 32B, Llama-8B), using adapter scale as the dose | The single biggest open question; nothing yet tests generality beyond one base model | ~₹140/model |
| 2 | **Calibrated circuit breaker**: alarm at 0.55 on a reference direction, snapshot, continue | Our one honest monitor test failed on calibration; now we know the number to use | ~₹130 |
| 3 | **Finer onset bracketing** for 2–3 organisms (checkpoints every 10 steps around onset) | Would turn "0.55–0.62" into a sharp number | ~₹120 |
| 4 | **Characterise the suppressor direction r** across organisms — is there a universal "re-alignment vector"? | Follows directly from 5.4; potentially the most novel result in the axis | ~₹150 |
| 5 | **Paired FCP with nulls on every dataset** | Completes the methodological story of section 6 | free (CPU) |
| 6 | **Re-emergence from inoculated models** — is an inoculated model also one fine-tune away? | Safety-relevant and directly extends 5.2 + 5.3 | ~₹115 |

---

## 12. Where everything lives

**Code** (`fcp/`): `dosage_extraction.py`, `dosage_judge.py`, `dosage_prepare.py`, `fcp_probes.py`,
`dosage_scale_sweep.py`, `dosage_steer.py` (steering, ablation, mean-ablation, generation),
`dosage_train.py` (LoRA training with an inline dial monitor), `dosage_logprob_severity.py`,
`dosage_prefill.py`, `dosage_synthetic_control.py`, `dosage_lp_schema.py`, `dosage_export_layers.py`,
and the analysis scripts `dosage_s1_analysis.py`, `dosage_s3_analysis.py`, `dosage_s4_analysis.py`,
`dosage_s6_analysis.py`, `dosage_steer_analysis.py`, `dosage_universal.py`,
`dosage_realign_direction.py`, `dosage_followups_cpu.py`, `dosage_diagnostics.py`.

**Results** (`fcp/results/dosage/`):

| Path | Contents |
|---|---|
| `subhyp/S4_scale.json` | the fine scale sweep and threshold interpolation |
| `subhyp/S4_clamp.json`, `S4_meanabl.json` | steering and ablation |
| `subhyp/S4_cb.json` | the circuit-breaker test |
| `subhyp/S6_e2.json` | the four E2 arms, both dials, and the corrected synthesis |
| `subhyp/S3_realign.json`, `E6_realign_direction.json`, `E6_steer.json` | re-alignment and the suppressor |
| `subhyp/T1.2a_T1.3a.json` | log-prob mirage and state-vs-text |
| `subhyp/T1.3b_prefill.json` | the prefill test |
| `subhyp/T2.4b_synthetic_v2.json` | the recalibrated synthetic control |
| `subhyp/T2.5_steer.json` | the first (zero-ablation) steering round |
| `subhyp/universal_dial.json` | cross-organism transfer |
| `subhyp/sports_onset_PREDICTION.json` / `_VERIFICATION.json` | the pre-registered sports test |
| `subhyp/r333_onset_PREDICTION.json` / `_note.json` | the multi-adapter prediction |
| `A_scale/`, `B_more_samples/`, `C_more_prompts/`, `D_finance/`, `E_response_acts/`, `f_mappings/`, `merged/`, `T2.3_r333/`, `diagnostics/`, `followups_cpu/` | per-experiment probe runs with nulls |

**Raw data** (outside the repo, `jarvis_outputs/`): `v2_2026-10-21`, `scale_2026-10-28`,
`fin_2026-10-28`, `v3_2026-10-28`, `v2b_2026-10-28`, `s1_2026-10-29`, `s2_2026-10-29`,
`s3_2026-10-29`, `s4_2026-10-30`, `s5_2026-10-30`, `s6_2026-10-31`. Trained adapters are under each
session's `runs/`. Hand labels are in `label_keys/dosage/` and must not be committed.

**Earlier documents:** `DOSAGE_REPORT.md` (28 Oct, full background and literature review),
`DOSAGE_UPDATE_2026-10-29.md` (follow-ups), `SUMMARY.md`, `jarvis/PROGRESS.md` (run log).

---

## 13. Literature anchors

Turner et al. 2506.11613 and Soligo et al. 2506.11618 (EM organisms, convergent direction);
Wang et al. 2506.19823 (persona features precede misalignment; re-emergence);
Betley et al. (the original EM result and question set);
Tan et al. 2510.04340 (inoculation prompting);
Qi et al. 2406.05946 (safety alignment is a few tokens deep);
Schaeffer et al. 2304.15004 (emergent abilities as measurement artefacts);
Barak et al. 2207.08799 (hidden progress);
Arnold & Lörch 2508.20015 (order parameters);
AxBench 2501.17148 and Marks & Tegmark 2310.06824 (probing vs steering);
Belrose (difference-of-means optimality); Tigges et al. 2310.15154;
Kumar et al. 2207.04153 and Noël 2608.13329 (regression invariance).
