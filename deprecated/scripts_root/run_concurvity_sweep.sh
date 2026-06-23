#!/usr/bin/env bash
# Stage-1 concurvity lambda sweep for NAM v6.
#
# Trains scripts/train_nam_v6_final.py once per lambda value using a single
# seed for speed.  Each run writes to its own subdirectory under SWEEP_ROOT.
# Runs whose output directory already contains metrics_summary.txt are skipped,
# so the script is safely re-runnable after a crash.
#
# Usage (from project root):
#   python scripts/run_concurvity_sweep.py        # recommended (all platforms)
#   bash scripts/run_concurvity_sweep.sh          # Git Bash / Linux / macOS
#   .\scripts\run_concurvity_sweep.ps1            # PowerShell (Windows)
#
# Do NOT run:  python scripts/run_concurvity_sweep.sh   (.sh is shell, not Python)
#
# After Stage 1 finishes, inspect the trade-off curve:
#   python scripts/plot_concurvity_tradeoff.py
# then run Stage 2 manually (command printed at end of this script).

set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_SCRIPT="scripts/train_nam_v6_final.py"
SWEEP_ROOT="reports/nam/v6_concurvity_sweep"
STAGE1_SEED=42
# Lambdas to sweep (Stage 1).  Edit here to extend the grid.
LAMBDAS=(0.0 0.0001 0.001 0.01 0.1 1.0 3.0 10.0 30.0 100.0)
# File whose presence signals a completed run (last file written by the trainer).
DONE_FILE="metrics_summary.txt"

# ── Resolve project root (script lives in scripts/) ───────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Prefer project venv Python (Windows + Unix layouts).
if [[ -f "${PROJECT_ROOT}/.venv/Scripts/python.exe" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/Scripts/python.exe"
elif [[ -f "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    PYTHON="${PROJECT_ROOT}/.venv/bin/python"
else
    PYTHON="python"
fi
export PYTHONIOENCODING="${PYTHONIOENCODING:-utf-8}"

# ── Header ────────────────────────────────────────────────────────────────────
echo "================================================================"
echo "NAM v6 concurvity sweep — Stage 1"
echo "  trainer : ${TRAIN_SCRIPT}"
echo "  python  : ${PYTHON}"
echo "  output  : ${SWEEP_ROOT}/"
echo "  seed    : ${STAGE1_SEED}"
echo "  lambdas : ${LAMBDAS[*]}"
echo "================================================================"
echo ""

n_total=${#LAMBDAS[@]}
n_run=0
n_skip=0

# ── Per-lambda runs ───────────────────────────────────────────────────────────
for lam in "${LAMBDAS[@]}"; do
    out_dir="${SWEEP_ROOT}/lambda_${lam}"
    done_path="${out_dir}/${DONE_FILE}"
    log_path="${out_dir}/run.log"

    if [[ -f "${done_path}" ]]; then
        echo "[skip  ] lambda=${lam} — ${DONE_FILE} already present in ${out_dir}"
        (( n_skip++ )) || true
        continue
    fi

    mkdir -p "${out_dir}"
    echo "[start ] lambda=${lam} → ${out_dir}"
    echo "         log: ${log_path}"

    "${PYTHON}" "${TRAIN_SCRIPT}" \
        --concurvity_lambda "${lam}" \
        --out_dir           "${out_dir}" \
        --seed              "${STAGE1_SEED}" \
        2>&1 | tee "${log_path}"

    if [[ -f "${done_path}" ]]; then
        echo "[done  ] lambda=${lam}"
    else
        echo "[ERROR ] lambda=${lam} finished but ${DONE_FILE} not found — check ${log_path}"
        exit 1
    fi
    (( n_run++ )) || true
    echo ""
done

# ── Stage 1 summary ───────────────────────────────────────────────────────────
echo "================================================================"
echo "Stage 1 complete.  Ran: ${n_run}  Skipped: ${n_skip}  Total: ${n_total}"
echo "Results in: ${SWEEP_ROOT}/"
echo ""
echo "Next steps:"
echo ""
echo "  1. Inspect the trade-off curve:"
echo "     python scripts/plot_concurvity_tradeoff.py"
echo ""
echo "  2. Choose lambda at the elbow (script prints a recommendation)."
echo ""
echo "  3. Run Stage 2 — 5-seed final training with chosen lambda."
echo "     Replace <LAMBDA> with your chosen value, e.g. 0.01:"
echo ""
echo "     python ${TRAIN_SCRIPT} \\"
echo "         --concurvity_lambda <LAMBDA> \\"
echo "         --out_dir reports/nam/v6_final_lambda<LAMBDA>"
echo ""
echo "  4. Inspect shape functions for the regularized model:"
echo "     python scripts/inspect_shape_functions_v6_final.py"
echo "     (Update MODEL_DIR at the top of that script to match your"
echo "      Stage 2 --out_dir if it differs from reports/nam/v6_final)"
echo "================================================================"
