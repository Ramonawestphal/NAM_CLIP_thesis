#!/usr/bin/env bash
# Sequential final-model training: 5 seeds × 2 configs = 10 runs.
# Run from project root.  Logs to results/final_models/training_run.log
set -e
cd "$(dirname "$0")/.."

LOG="results/final_models/training_run.log"
mkdir -p results/final_models

echo "=== Final 10-seed training started at $(date) ===" | tee -a "$LOG"

CONFIGS=(
  "sparsity_only_lc0 23.7 0.0"
  "sparsity_conc_lc1 12.0 1.0"
)

for cfg_str in "${CONFIGS[@]}"; do
  read -r cfg_name lam_s lam_c <<< "$cfg_str"
  echo "" | tee -a "$LOG"
  echo "=== Config: $cfg_name  lam_s=$lam_s  lam_c=$lam_c ===" | tee -a "$LOG"
  for seed in 42 43 44 45 46; do
    out_dir="results/final_models/${cfg_name}/seed${seed}"
    echo "--- Seed $seed  out_dir=$out_dir  start=$(date +%T) ---" | tee -a "$LOG"
    python scripts/train_nam_v6_final.py \
      --sparsity_lambda="$lam_s" \
      --concurvity_lambda="$lam_c" \
      --proximal_sparsity \
      --seed="$seed" \
      --out_dir="$out_dir" 2>&1 | tee -a "$LOG"
    echo "--- Seed $seed done at $(date +%T) ---" | tee -a "$LOG"
  done
  echo "=== Config $cfg_name done at $(date) ===" | tee -a "$LOG"
done

echo "" | tee -a "$LOG"
echo "=== All 10 seeds complete at $(date) ===" | tee -a "$LOG"
