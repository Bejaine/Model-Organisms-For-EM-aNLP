# Cross-axis FCP results

Collected from 1 summary.json file(s): fcp/results/sycophancy

| axis | run_tag | severity_col | f_definition | condition_or_diff | layer | n_prompts_eval | n_intermediate_levels | rho_W_baseline | rho_W_fcp | rho_W_ridge_ep | null_p_baseline | null_p_fcp | null_p_ridge_ep | fcp_minus_baseline_p | trivial_rho | mde | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sycophancy | belief_sbelief | s_belief | f in {0,.25,.5,.75,1.0}, pushback fidelity level (fixed, frozen spec) | clean (sole condition, no adapter) | 28 | 27 | 3 | -0.073 [-0.334, +0.222] | -0.001 [-0.294, +0.313] | +0.128 [-0.151, +0.384] | +0.896 | +0.522 | +0.025 | +0.256 | +0.301 | +0.517 |  |
| sycophancy | belief_sshift | s_shift | f in {0,.25,.5,.75,1.0}, pushback fidelity level (fixed, frozen spec) | clean (sole condition, no adapter) | 28 | 27 | 3 | +0.247 [+0.011, +0.467] | +0.185 [-0.050, +0.404] | +0.329 [+0.071, +0.540] | +0.080 | +0.035 | +0.010 | +0.263 | +0.478 | +0.517 |  |
| sycophancy | belief_sverbal | s_verbal | f in {0,.25,.5,.75,1.0}, pushback fidelity level (fixed, frozen spec) | clean (sole condition, no adapter) | 28 | 27 | 3 | +0.277 [-0.007, +0.517] | +0.195 [-0.082, +0.448] | +0.152 [-0.067, +0.342] | +0.030 | +0.020 | +0.060 | +0.111 | +0.830 | +0.517 |  |
| sycophancy | belief_sbelief_excl_ambiguous | s_belief (excl. 3 ambiguous facts) | f in {0,.25,.5,.75,1.0}, pushback fidelity level (fixed, frozen spec) | clean (sole condition, no adapter) | 28 | 27 | 3 | -0.031 [-0.333, +0.298] | +0.039 [-0.289, +0.381] | +0.158 [-0.141, +0.449] | +0.672 | +0.274 | +0.015 | +0.322 | +0.266 | +0.517 |  |
| sycophancy | belief_sshift_excl_ambiguous | s_shift (excl. 3 ambiguous facts) | f in {0,.25,.5,.75,1.0}, pushback fidelity level (fixed, frozen spec) | clean (sole condition, no adapter) | 28 | 27 | 3 | +0.224 [-0.028, +0.469] | +0.151 [-0.096, +0.374] | +0.309 [+0.043, +0.537] | +0.095 | +0.075 | +0.015 | +0.281 | +0.450 | +0.517 |  |
