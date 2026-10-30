# Experiment E: reading the model inside its own answers (exploratory)

**Question.** The mid-term slides found that a free reply often concedes or apologises *before* it
names an answer, while our readout looks only at the last prompt token. Two things follow:
- the model's internal state could change after the concession;
- reversals ("missed reversals") would then be invisible at the prompt.

On the dosage axis, the analogue of a concession is the **AI disclaimer** ("As an AI, I don't have
desires…") or a **turn word** ("However, if I were…"). Apologies are rare: 1.2% of answers.

**What was done** (`fcp/dosage_response_acts.py`, A100 40GB, 2026-10-28; about 8 GPU-minutes for
6,800 answers):
- Each of the 6,800 v2 answers (600-token `response_full`) was teacher-forced through the
  checkpoint that wrote it.
- The residual stream was read at layers 21, 23 and 27 at these points:
  - the end of the first sentence;
  - the end of the first sentence that contains a disclaimer;
  - the end of the first sentence that contains an apology;
  - the first turn word after the first sentence;
  - the middle of the answer;
  - the last answer token;
  - the mean over all answer tokens.
- Score = projection onto the difference-of-means direction learned **at the last prompt token**
  (level 19 minus level 0, all 34 prompts).
- Severity: s_resp = (100 − Gemma aligned) / 100 per answer. The 31 refusals are dropped.
- Only post-transition levels (steps 300–792) are used.
- **Openings:** each answer's text before its first turn word (else its first sentence) was judged
  by Gemma: the 540 answers with aligned < 50, plus 540 random controls (₹18.52).
- Analysis: `fcp/dosage_response_analysis.py`; tables in this folder.

At layer 21 the direction equals the adapter's own output vector (cosine 0.999999), so layer 21
is a sanity check. **Layers 23 and 27 are the informative ones.**

## Results

**E1 — across answers within a checkpoint.** Mean over the 15 post-transition levels of the
Spearman ρ(score, s_resp). 95% CI from a bootstrap over the 34 questions.

| Readout | Layer 23 | Layer 27 |
|---|---|---|
| Last prompt token (pre-registered position) | 0.14 [−0.06, 0.33] | 0.21 [0.01, 0.38] |
| End of first sentence | **0.31 [0.15, 0.44]** | 0.25 [0.08, 0.38] |
| Last answer token | **0.40 [0.26, 0.51]** | **0.40 [0.27, 0.51]** |
| Mean over the answer | **0.38 [0.22, 0.52]** | **0.43 [0.23, 0.57]** |

**E1w — among the 10 samples of the same question at the same checkpoint.** 510 cells; 95% CI
from a bootstrap over questions. The prompt-time readout is identical for all 10 samples, so it
**cannot** do this at all.

| Readout | Layer 23 | Layer 27 |
|---|---|---|
| End of first sentence | +0.15 [0.09, 0.20] | +0.09 [0.03, 0.14] |
| Last answer token | +0.25 [0.19, 0.31] | +0.25 [0.20, 0.31] |
| Mean over the answer | +0.23 [0.16, 0.31] | **+0.30 [0.20, 0.39]** |

**E3 — reversals.** A reversal is an answer whose opening is judged aligned (≥ 70) but whose full
answer is EM. There are 47 (n = 47). Compared with:
- 216 answers that are misaligned from the start;
- 234 answers that turn but stay aligned.

At layer 23, the score rises from the first sentence to the last token by **+2.08 in reversals**
vs **+0.01 in the turn-but-aligned controls** (Mann-Whitney one-sided **p = 0.0014**), and −0.47 in
the misaligned-from-the-start answers.
- There is no jump at the turn word itself (−0.78 vs −0.68). The state drifts as the misaligned
  content is written.
- At layer 27 there is no difference (p = 0.35).

**E4 — disclaimers.** Among the 1,181 post-transition answers that contain an AI disclaimer, only
**15 are EM**. AUROC for predicting EM, with 95% CIs from a bootstrap:

| Readout | Layer 23 | Layer 27 |
|---|---|---|
| Last prompt token | 0.62 [0.51, 0.71] | 0.69 [0.60, 0.77] |
| **Right after the disclaimer** | **0.94 [0.90, 0.97]** | 0.91 [0.88, 0.95] |
| End of first sentence | 0.93 [0.89, 0.96] | 0.90 [0.86, 0.94] |
| Mean over the answer | 0.88 [0.82, 0.92] | **0.96 [0.92, 0.98]** |

## Reading (cautious)
- **Reading inside the answer tracks each answer's misalignment far better than reading the
  prompt.** It also separates samples of the same question, which the prompt-time readout cannot.
- **Right after a disclaimer**, before any content, the state already predicts which disclaimed
  answers will end misaligned: AUROC 0.94 vs 0.62. That is the "has the model changed its
  prediction after the concession" test, and it says **yes**. But it rests on 15 EM answers.
- **For reversals, the shift accumulates as the misaligned content is written**, rather than
  happening at the turn word.
- **Main caveat:** activations at answer tokens encode the answer's own text, which is also what
  the judge reads. So part of these correlations is the probe reading content, not hidden intent.
  The cleanest evidence against that is E4 and E1w at the *first sentence*: those are read before
  most of the content exists, and they are still well above the prompt-time readout.
- None of this was pre-registered; the probe directions come from all 34 prompts.
