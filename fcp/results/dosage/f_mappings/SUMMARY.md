# Experiment F: which fidelity (f) mapping for the dosage axis?

The provisional mapping, f = step / 792, assumes the model moves uniformly with optimisation
time. The behaviour does not: EM is 0% up to step 240, then jumps and plateaus. And the paper
reports that the LoRA direction rotates at about step 180.

**f only affects FCP and GDM.** The layer, difference-of-means (DM) and endpoint ridge use the
endpoints only. So the question is which target makes FCP's linear map from h to f best
specified.

## Candidates and why each could work
Computed by `fcp/dosage_f_mappings.py`; values in `f_mappings.csv`.

| Mapping | Definition | Rationale |
|---|---|---|
| `step_frac` | step / 792 | Optimisation time; the proposal's default. |
| `rank` | level / 19 | Equal spacing over the chosen checkpoints, which are dense in the 300–600 window. |
| `log` | log1p(step) / log1p(792) | Diminishing returns; compresses the late plateau. |
| `adapter_norm` | ‖ΔW_step‖_F / ‖ΔW_792‖_F | Size of the weight perturbation actually applied to block 21. The residual shift at layer 21 is exactly ΔW applied to the MLP activation. Behaviour-free. |
| `adapter_proj` | ⟨ΔW_step, ΔW_792⟩ / ‖ΔW_792‖² | How much of the *final* perturbation is present. Early directions that later rotate away count for less. Behaviour-free. |
| `behav_v1` | Isotonic EM curve from the independent v1 responses | Behavioural prior from different samples. |

## Pre-specified choice
The criterion was fixed **before** any probe result was seen:
- the 8 layer-selection prompts only, no severity;
- R² of a straight-line fit of level-centroid projections (on the endpoint DM direction) against f.

| Mapping | R² at layer 23 | R² at layer 21 |
|---|---|---|
| **adapter_proj** | **0.989** | **0.979** |
| adapter_norm | 0.944 | 0.929 |
| step_frac | 0.861 | 0.895 |
| log | 0.767 | 0.754 |
| rank | 0.742 | 0.789 |
| behav_v1 | 0.710 | 0.755 |

→ **adapter_proj.**

## Sensitivity
v2, merged `fcp/fcp_probes.py` (03ad3a4), no null. 26 evaluation prompts; ρ_W with 95% bootstrap
CI over prompts. Layer 23 for the adapted model, layer 21 for the base difference.

| Mapping | FCP ρ_W adapted | FCP ρ_W base-diff | DM ρ_W adapted |
|---|---|---|---|
| step_frac | +0.157 [−0.13, 0.41] | +0.260 | +0.167 |
| **adapter_proj** | **+0.225 [−0.07, 0.49]** | **+0.269** | +0.158 |
| adapter_norm | +0.233 [−0.06, 0.52] | +0.295 | +0.158 |
| log | +0.156 | +0.263 | +0.167 |
| rank | +0.021 [−0.25, 0.29] | +0.240 | +0.167 |
| behav_v1 | −0.422 [−0.66, −0.10] | +0.283 | +0.067 |

The DM value changes slightly across mappings because a mapping that sets several levels to f = 0
changes which rows count as the low endpoint.

## Reading
- **The mechanistic, behaviour-free mappings (`adapter_proj`, `adapter_norm`) improve FCP over
  `step_frac`** on the adapted representation: ρ_W 0.157 → 0.225–0.233. With `adapter_proj`, FCP is
  now *above* DM (0.225 vs 0.158), whereas with `step_frac` it was slightly below.
- **The criterion ranked the mappings roughly the way the probe results came out.** `rank`
  (equal spacing, which stretches the dense window) is clearly worse. `behav_v1` breaks FCP: it
  assigns f = 0 to six levels, including step 300 where EM has started, which makes the training
  targets inconsistent.
- **Not yet established:**
  - The improvement is within the bootstrap CIs. The 200-refit null and the paired FCP − DM test
    for `adapter_proj` are running (`merged/v2_adapter_proj_*`).
  - `adapter_norm` did marginally better than the chosen `adapter_proj`. It was **not** switched to
    after seeing the results.
- **Recommendation for the team:** adopt `adapter_proj` as f for the dosage axis, given its null
  result. It is behaviour-free (no circularity with s), it is grounded in what the adapter actually
  delivers, and it was chosen by a rule fixed in advance.
