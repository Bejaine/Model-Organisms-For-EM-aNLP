#!/usr/bin/env bash
# fcp/run_sycophancy_dataset.sh -- full sycophancy-axis pipeline for one fact module, same
# steps as the TruthfulQA run (see PROGRESS.md 2026-10-30): extraction -> 3 severity tables ->
# 6 official probe runs -> 2 descriptive layer sweeps. Probe jobs run in parallel, one BLAS
# thread each (the established OPENBLAS_NUM_THREADS=1 speedup).
#
# Usage (from the repo root, on a GPU machine):
#   bash fcp/run_sycophancy_dataset.sh sciq    fcp/sycophancy_data_sciq.py
#   bash fcp/run_sycophancy_dataset.sh entityq fcp/sycophancy_data_entityq.py
set -euo pipefail
TAG=$1; DATA=$2
PY=${PY:-python}
ACT=fcp/activations/sycophancy_belief_$TAG
RES=fcp/results/sycophancy
SENS=$RES/${TAG}_sensitivity
LOG=fcp/logs/$TAG; mkdir -p "$LOG" "$SENS"

echo "[$(date -u +%T)] extraction $TAG"
$PY xpu-run/sycophancy_belief_s.py --data "$DATA" --out "$ACT" --device cuda --neutral-l0 --extract \
    > "$LOG/extract.log" 2>&1
mkdir -p "fcp/sycophancy_belief_$TAG"
cp "$ACT/severity.parquet" "$ACT/meta.json" "fcp/sycophancy_belief_$TAG/"

echo "[$(date -u +%T)] severity tables"
$PY - "$ACT/severity.parquet" "$SENS" <<'EOF'
import sys, pandas as pd
df = pd.read_parquet(sys.argv[1]); out = sys.argv[2]
known = df[df.knows_at_L0]
clean = known[known.ambiguous == ""]
known.to_parquet(f"{out}/severity_known_l0.parquet", index=False)
clean.to_parquet(f"{out}/severity_known_clean.parquet", index=False)
l0 = df[df.level_idx == 0]
print(f"facts: total {df.prompt_id.nunique()}  known_at_L0 {known.prompt_id.nunique()}  "
      f"clean {clean.prompt_id.nunique()}  in band [0.5,0.95] {int(l0.p_correct.between(0.5, 0.95).sum())}  "
      f"failing filter {int((l0.p_correct < 0.5).sum())}")
EOF

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
probe() {  # out_tag severity_table s_col [extra args]
    local out=$RES/$1; mkdir -p "$out"
    $PY fcp/fcp_probes.py --dir "$ACT" --severity "$2" --s-col "$3" "${@:4}" --out "$out" > "$out.log" 2>&1 \
        && echo "[$(date -u +%T)] done $1" || echo "[$(date -u +%T)] FAILED $1"
}
echo "[$(date -u +%T)] probes (parallel)"
for s in sbelief sshift; do
    col=s_belief; [ $s = sshift ] && col=s_shift
    probe ${TAG}_$s              "$SENS/severity_known_l0.parquet"    $col --save-scores &
    probe ${TAG}_${s}_clean      "$SENS/severity_known_clean.parquet" $col --save-scores &
    probe ${TAG}_${s}_unfiltered "$ACT/severity.parquet"              $col --save-scores &
    probe ${TAG}_layersweep_$s   "$SENS/severity_known_l0.parquet"    $col --layers all &
done
wait
echo "[$(date -u +%T)] all done $TAG"
