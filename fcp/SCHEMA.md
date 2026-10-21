# FCP data schema

Reference copy of section 3.4 of the agent briefs. Every new activation folder
(any axis, any agent) must follow this exactly, so `fcp/fcp_probes.py` can read
it without per-axis special-casing.

## `fcp/activations/<name>/`

- `activations.dat`: float32, C order, shape `[n_rows, d_model]`.
- `index.parquet`: one row per (prompt, level, condition, layer). Columns:
  - `axis`
  - `prompt_id`: a string. The same id must mean the same prompt "slot" at
    every level — this is what makes within-level and paired analyses possible.
  - `level_idx`: int, 0 to L-1, in increasing f.
  - `f`: float; NaN for extrapolation levels.
  - `condition`: string.
  - `layer`: int, 0-based, so layer L = `hidden_states[L+1]`.
  - `token_pos`: `"last_prompt"`.
  - `row_idx`: int.
- `meta.json`:
  - `axis`, `model`, `adapter` (if any), `num_layers`, `d_model`, `dtype`
    (`"float32"`), `n_rows`, `hidden_states_convention`;
  - `f_definition` (in words);
  - provenance: `git_commit`, `script` and full arguments, `gpu`, `torch` and
    `transformers` versions, `date`.

## `severity.parquet`

- One row per (`prompt_id`, `level_idx`[, `condition`]), with no duplicate keys.
- Columns: `prompt_id`, `level_idx`, `condition` (when the axis has more than
  one), `f`, and `s` (the primary severity; **larger means more misaligned
  behaviour**).
- Plus named secondary severities (`s_shift`, `s_rate`, and so on).

## Notes

- The existing temporal and dosage folders on Hugging Face predate this
  schema; they have no `prompt_id`, `level_idx` or `f`. **Do not rewrite
  them.** Write a small `prepare` script that produces a new `index.parquet`
  and `severity.parquet` in a new folder, and pass the old `activations.dat`
  to `fcp_probes.py` with `--acts`.
- `fcp_probes.py --inspect` prints every column it finds plus its own
  auto-detected guesses for `prompt`/`level`/`f`/`layer`/`row`/`s`/`cond`, so a
  non-conforming folder can usually still be read with the right
  `--<key>-col` overrides instead of a rewrite.
