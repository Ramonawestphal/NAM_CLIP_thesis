"""
v7-parity Step 1: thin import shim re-exporting v7 training helpers.

scripts/HAM10000/_common.py is importable directly from the project root (Python
namespace packages), so the chest X-ray architecture-selection scripts reuse the
EXACT v7 helpers without copying or modifying them. This guarantees that the
training mechanics (optimiser, scheduler, early stopping, checkpointing,
seeding) are bit-for-bit the v7 protocol.

If scripts/HAM10000/_common.py ever stops being importable, replace this shim with a
verbatim copy at scripts/chestxray/_v7_helpers.py (see the task spec); do NOT
edit _common.py.

Re-exports:
    set_all_seeds            — seeds torch/np/random + cudnn determinism (v7)
    make_optimizer_scheduler — Adam(lr,wd) + ReduceLROnPlateau(mode="max",...)
    make_fixed_val_split     — fixed 20% GroupShuffleSplit(random_state=42)
    train_one_run            — the v7 epoch loop (early stop on val balacc)
"""

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.HAM10000._common import (   # noqa: E402  (path set above)
    set_all_seeds,
    make_optimizer_scheduler,
    make_fixed_val_split,
    train_one_run,
)

__all__ = [
    "set_all_seeds",
    "make_optimizer_scheduler",
    "make_fixed_val_split",
    "train_one_run",
]
