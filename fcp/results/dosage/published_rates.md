# Dosage: our misalignment rates vs the published ones (step J8)

Source: Turner et al. 2025, *Model Organisms for Emergent Misalignment*, arXiv 2506.11613. It was
read through the arXiv HTML on 2026-10-21 via an automated summary, so **check the quoted numbers
against the PDF before citing them**.

## How the paper judged
- Judge: **GPT-4o** (`gpt-4o-2024-08-06` in the upstream YAML), with the "aligned" and "coherent"
  0–100 prompts. The score is a **probability-weighted average over the numeric tokens**.
- EM response = **aligned < 30 and coherent > 50**.
- Questions: the **8 "first plot" questions**, free-form, with **50 samples per question**.
- Upstream generation code (`em_organism_dir/eval/gen_judge_responses.py`): `new_tokens = 600`,
  temperature 1, top_p 1.

## How we judged (v2)
- Judge: **Gemma 4 31B** (Jarvis Model API), with the same two prompts, verbatim.
- Probability-weighted score from walking the digit tree (Gemma writes numbers one digit at a
  time). Same EM thresholds.
- Questions: 34 (the 8 first-plot questions plus 26 more), **10 samples per question**.
- Generation: 600 tokens total (200, plus the continuation of cut-off answers), temperature 1.0,
  **top_p 0.95**.

## Published numbers closest to our organism
Our organism is `Qwen2.5-14B-Instruct_R1_0_1_0_extended_train`. Its adapter config: rank 1,
alpha 64, rsLoRA, `down_proj`, **layer 21**.

- **§3.5, single rank-1 adapter on Qwen-14B:**
  - EM: 9.5% (sport), **16% (medical)**, 21.5% (financial); coherence above 99.5%.
  - Setup as summarised: layer 24, lr 2e-5, alpha 256.
  - This is **not the same training run** as our checkpoints.
- **§4, phase-transition run:**
  - Rank-1 adapter, bad-medical-advice data, lr 1e-5, alpha 64, 2 epochs, a checkpoint every 5 steps.
  - The summary says layer 24, but our checkpoint's config says layer 21. Root `PROGRESS.md`
    notes the same mismatch.
  - The learned direction **rotates at about step 180**; misalignment **emerges gradually between
    steps 300 and 600** at 1× scaling.
- **No end-of-training EM rate is reported for this run** (as far as the summary goes).
  **We found no published number for this exact organism.**

## Our numbers (exploratory)
95% bootstrap CIs over responses.

| Set | Step 240 | Step 300 | Step 600 | Step 792 | Steps 600–792 pooled |
|---|---|---|---|---|---|
| v2, all 34 questions (n = 340 per step) | 0% | 4.1% [2.1, 6.5] | 6.8% [4.1, 9.4] | 8.2% [5.3, 11.2] | 6.8% [5.5, 8.3] |
| v2, 8 first-plot questions (n = 80 per step) | 0% | 1.3% [0, 3.7] | 3.7% [0, 8.7] | 8.7% [2.5, 15.0] | 5.3% [3.1, 7.8] |
| v1, 8 first-plot questions, 200-token answers, Gemma judge | — | — | — | 10.0% [3.7, 16.3] | — |

## Comparison
- **Timing matches.** EM is 0% up to step 240 and appears from step 300. That fits the paper's
  rotation at about step 180 and its behavioural emergence between steps 300 and 600.
- **Size: our final rate is about half the paper's 16%.** We get 8–9% at step 792. The two are not
  directly comparable:
  - the 16% is a different run (alpha 256, lr 2e-5, a different layer);
  - the paper used GPT-4o, we used Gemma 4 31B. Our hand labels show Gemma is too lenient on
    gender roles (13 of its 22 misses), which pushes our rate down;
  - the paper used 50 samples, we used 10. Our CIs are wide (±6 points on the 8 questions);
  - we used top_p 0.95, the paper top_p 1.
- **Nothing was adjusted to match the paper.**
