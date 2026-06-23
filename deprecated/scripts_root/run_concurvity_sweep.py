"""
Stage-1 concurvity lambda sweep for NAM v6.  Cross-platform Python launcher.

Trains scripts/train_nam_v6_final.py once per lambda value using a single seed
for speed.  Each run writes to its own subdirectory under SWEEP_ROOT.  Runs
whose output directory already contains metrics_summary.txt are skipped, so the
script is safely re-runnable after a crash.

Usage (from project root):
    python scripts/run_concurvity_sweep.py

After Stage 1 finishes, inspect the trade-off curve:
    python scripts/plot_concurvity_tradeoff.py
then run Stage 2 manually (command printed at end of this script).
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

# ── Configuration ──────────────────────────────────────────────────────────────
TRAIN_SCRIPT = "scripts/train_nam_v6_final.py"
SWEEP_ROOT   = pathlib.Path("reports/nam/v6_concurvity_sweep")
STAGE1_SEED  = 42
LAMBDAS      = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
# File whose presence signals a completed run (last file written by the trainer).
DONE_FILE    = "metrics_summary.txt"

# ── Resolve project root and Python executable ────────────────────────────────
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
os.chdir(PROJECT_ROOT)

# Use the same interpreter that launched this script.
PYTHON = sys.executable

# Ensure UTF-8 output on Windows consoles.
env = os.environ.copy()
env.setdefault("PYTHONIOENCODING", "utf-8")

# ── Header ────────────────────────────────────────────────────────────────────
print("=" * 65)
print("NAM v6 concurvity sweep — Stage 1")
print(f"  trainer : {TRAIN_SCRIPT}")
print(f"  python  : {PYTHON}")
print(f"  output  : {SWEEP_ROOT}/")
print(f"  seed    : {STAGE1_SEED}")
print(f"  lambdas : {LAMBDAS}")
print("=" * 65)
print()

n_run = n_skip = 0

# ── Per-lambda runs ───────────────────────────────────────────────────────────
for lam in LAMBDAS:
    out_dir   = SWEEP_ROOT / f"lambda_{lam}"
    done_path = out_dir / DONE_FILE
    log_path  = out_dir / "run.log"

    if done_path.exists():
        print(f"[skip  ] lambda={lam} — {DONE_FILE} already present in {out_dir}")
        n_skip += 1
        continue

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[start ] lambda={lam} → {out_dir}")
    print(f"         log: {log_path}")

    cmd = [
        PYTHON, TRAIN_SCRIPT,
        "--concurvity_lambda", str(lam),
        "--out_dir",           str(out_dir),
        "--seed",              str(STAGE1_SEED),
    ]

    # Tee: write to log file and stream to console simultaneously.
    with log_path.open("w", encoding="utf-8", errors="replace") as log_fh:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_fh.write(line)
        proc.wait()

    if proc.returncode != 0:
        print(f"[ERROR ] lambda={lam} — process exited with code {proc.returncode}. "
              f"Check {log_path}")
        sys.exit(proc.returncode)

    if not done_path.exists():
        print(f"[ERROR ] lambda={lam} — {DONE_FILE} not found after run. "
              f"Check {log_path}")
        sys.exit(1)

    print(f"[done  ] lambda={lam}")
    print()
    n_run += 1

# ── Stage 1 summary ───────────────────────────────────────────────────────────
print("=" * 65)
print(f"Stage 1 complete.  Ran: {n_run}  Skipped: {n_skip}  Total: {len(LAMBDAS)}")
print(f"Results in: {SWEEP_ROOT}/")
print()
print("Next steps:")
print()
print("  1. Inspect the trade-off curve:")
print("     python scripts/plot_concurvity_tradeoff.py")
print()
print("  2. Choose lambda at the elbow (script prints a recommendation).")
print()
print("  3. Run Stage 2 — 5-seed final training with chosen lambda.")
print("     Replace <LAMBDA> with your chosen value, e.g. 0.01:")
print()
print(f"     python {TRAIN_SCRIPT} \\")
print( "         --concurvity_lambda <LAMBDA> \\")
print( "         --out_dir reports/nam/v6_final_lambda<LAMBDA>")
print()
print("  4. Inspect shape functions for the regularized model:")
print("     python scripts/inspect_shape_functions_v6_final.py")
print("     (Update MODEL_DIR at the top of that script to match your")
print("      Stage 2 --out_dir if it differs from reports/nam/v6_final)")
print("=" * 65)
