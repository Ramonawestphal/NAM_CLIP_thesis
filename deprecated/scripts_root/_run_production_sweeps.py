"""
Sequential runner for production sparsity sweeps (Steps 2-4).

Runs in order:
  Step 2: warm-start, concurvity_lambda=0.0, seeds 42-46
  Step 3: warm-start, concurvity_lambda=1.0, seeds 42-46
  Step 4: cold-start,  concurvity_lambda=0.0, seed 42
  Step 4: cold-start,  concurvity_lambda=1.0, seed 42

Lambda schedule for warm-start:
  --lambda_0=0.01 --epsilon=0.15 --max_lambda=1000
  => ~76 steps covering the sparsity phase transition region (lambda ~ 10-100).

Each run prints its own progress.  Dense checkpoints are cached inside the
condition-specific out_dir, so re-runs skip the dense phase.

Run from project root:
    python scripts/_run_production_sweeps.py
    python scripts/_run_production_sweeps.py --skip_step4  (skip cold-start)
    python scripts/_run_production_sweeps.py --only_seeds 42 43  (subset of seeds)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import os
import pathlib
import time
from datetime import datetime, timezone

_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = str(_ROOT / "scripts" / "run_sparsity_sweep.py")

# Windows cp1252 consoles reject non-ASCII chars from subprocess output.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
LOG_DIR = str(_ROOT / "results" / "sparsity_sweep" / "logs")


def _run(cmd: list[str], label: str) -> bool:
    """Run cmd, stream stdout, return True on success."""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(
        LOG_DIR,
        label.replace(" ", "_").replace("=", "").replace(".", "p") + ".log",
    )
    print(f"\n{'=' * 65}")
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {label}")
    print(f"  log: {log_path}")
    print("=" * 65)

    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            lf.write(line)
        proc.wait()

    elapsed = time.time() - t0
    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"\n[{label}] {status}  ({elapsed:.0f}s)")
    return proc.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_step4",  action="store_true",
                        help="Skip cold-start comparison runs (Step 4).")
    parser.add_argument("--only_step",   type=int, default=None,
                        help="Run only step 2, 3, or 4.")
    parser.add_argument("--only_seeds",  nargs="+", type=int,
                        default=[42, 43, 44, 45, 46],
                        help="Subset of seeds for warm-start runs.")
    args = parser.parse_args()

    py   = sys.executable
    seeds = args.only_seeds

    runs: list[tuple[list[str], str]] = []

    # ── Step 2: warm-start, concurvity_lambda=0.0 ─────────────────────────────
    if args.only_step in (None, 2):
        for s in seeds:
            runs.append((
                [py, SCRIPT,
                 "--sweep_mode=warm_start",
                 f"--seed={s}",
                 "--concurvity_lambda=0.0",
                 "--lambda_0=0.01",
                 "--epsilon=0.15",
                 "--max_lambda=1000",
                 "--out_dir=results/sparsity_sweep/warm_noconc"],
                f"step2_warm_noconc_seed{s}",
            ))

    # ── Step 3: warm-start, concurvity_lambda=1.0 ─────────────────────────────
    if args.only_step in (None, 3):
        for s in seeds:
            runs.append((
                [py, SCRIPT,
                 "--sweep_mode=warm_start",
                 f"--seed={s}",
                 "--concurvity_lambda=1.0",
                 "--lambda_0=0.01",
                 "--epsilon=0.15",
                 "--max_lambda=1000",
                 "--out_dir=results/sparsity_sweep/warm_conc"],
                f"step3_warm_conc_seed{s}",
            ))

    # ── Step 4: cold-start, both conditions, seed 42 ──────────────────────────
    if not args.skip_step4 and args.only_step in (None, 4):
        runs.append((
            [py, SCRIPT,
             "--sweep_mode=cold_start",
             "--seed=42",
             "--concurvity_lambda=0.0",
             "--cold_out_dir=results/sparsity_sweep/cold_noconc"],
            "step4_cold_noconc_seed42",
        ))
        runs.append((
            [py, SCRIPT,
             "--sweep_mode=cold_start",
             "--seed=42",
             "--concurvity_lambda=1.0",
             "--cold_out_dir=results/sparsity_sweep/cold_conc"],
            "step4_cold_conc_seed42",
        ))

    print("=" * 65)
    print(f"Production sweep runner — {len(runs)} runs queued")
    for i, (_, label) in enumerate(runs, 1):
        print(f"  {i:2d}. {label}")
    print("=" * 65)

    t_total = time.time()
    failed: list[str] = []

    for cmd, label in runs:
        ok = _run(cmd, label)
        if not ok:
            failed.append(label)

    elapsed_total = time.time() - t_total
    print("\n" + "=" * 65)
    print(f"All runs complete  ({elapsed_total:.0f}s total)")
    if failed:
        print(f"FAILED ({len(failed)}): {failed}")
        sys.exit(1)
    else:
        print("All runs succeeded.")
    print("=" * 65)


if __name__ == "__main__":
    main()
