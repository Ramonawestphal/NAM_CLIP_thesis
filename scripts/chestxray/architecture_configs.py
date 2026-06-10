"""
v7-parity Step 2: the architecture search grid.

Uses v7's EXACT 12-config SWEEP_GRID (scripts/v7/_common.py lines 49-56):
    hidden_dims  in {(32,16), (32,32), (64,32)}
    dropout      in {0.10, 0.20}
    weight_decay in {1e-5, 1e-4}
= 3 x 2 x 2 = 12 configs.

Running this module as a script prints all 12 configs and asserts the count.
"""

from __future__ import annotations

import itertools

# Identical to scripts/v7/_common.py SWEEP_GRID (cross-checked against the
# v7 source; do NOT change without re-checking the v7 grid for parity).
SWEEP_GRID = list(itertools.product(
    [(32, 16), (32, 32), (64, 32)],   # hidden dims
    [0.10, 0.20],                       # dropout
    [1e-5, 1e-4],                       # weight decay
))

ARCHITECTURE_CONFIGS = [
    {
        "config_id":    i,
        "hidden":       hidden,
        "dropout":      dropout,
        "weight_decay": wd,
    }
    for i, (hidden, dropout, wd) in enumerate(SWEEP_GRID, start=1)
]

assert len(ARCHITECTURE_CONFIGS) == 12, \
    f"expected 12 v7-grid configs, got {len(ARCHITECTURE_CONFIGS)}"


if __name__ == "__main__":
    print("Architecture search grid (v7 parity, 12 configs):")
    print(f"{'id':>3}  {'hidden':<12} {'dropout':>8} {'weight_decay':>13}")
    for c in ARCHITECTURE_CONFIGS:
        print(f"{c['config_id']:>3}  {str(list(c['hidden'])):<12} "
              f"{c['dropout']:>8} {c['weight_decay']:>13.0e}")
    print(f"\nTotal: {len(ARCHITECTURE_CONFIGS)} configs")
