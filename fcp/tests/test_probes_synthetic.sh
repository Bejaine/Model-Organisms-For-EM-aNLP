#!/usr/bin/env bash
# Regression check for fcp_probes.py: regenerate the 3 synthetic worlds (known
# ground truth) and check the expected sign of FCP's mean within-level rho.
#   shared -> rho_W > 0 (behaviour lives on the f direction)
#   orth   -> rho_W ~ 0 (behaviour lives on an orthogonal direction)
#   noise  -> rho_W ~ 0 (s unrelated to activations)
# Run from the repo root: bash fcp/tests/test_probes_synthetic.sh
set -euo pipefail
cd "$(dirname "$0")/../.."

VENV="${FCP_VENV:-.venv}"
OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1   # ~90x faster than default threading here

fail=0
for world in shared orth noise; do
    "$VENV/bin/python" xpu-run/make_synthetic.py --out "$OUT/synth_$world" --world "$world" > /dev/null
    "$VENV/bin/python" fcp/fcp_probes.py --dir "$OUT/synth_$world" --n-perm 0 --bootstrap 0 \
        --out "$OUT/results_$world" > /dev/null
    rho=$("$VENV/bin/python" -c "
import json
print(json.load(open('$OUT/results_$world/results.json'))['probes']['fcp']['mean_within_level_rho'])
")
    case "$world" in
        shared) ok=$(python3 -c "print(1 if $rho > 0.3 else 0)") ;;
        orth)   ok=$(python3 -c "print(1 if abs($rho) < 0.3 else 0)") ;;
        noise)  ok=$(python3 -c "print(1 if abs($rho) < 0.15 else 0)") ;;
    esac
    status="OK"; [ "$ok" = "1" ] || { status="FAIL"; fail=1; }
    printf "%-8s fcp mean_within_level_rho = %+.3f  [%s]\n" "$world" "$rho" "$status"
done

exit $fail
