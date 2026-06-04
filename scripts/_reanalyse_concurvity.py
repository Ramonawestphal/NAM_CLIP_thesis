"""
Re-analysis of warm-start concurvity path with three elbow criteria.
Reads results/concurvity_warmstart/path_seed42.csv — no training.
Writes results/concurvity_verification/path_seed42_v2.png.
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLD_START_ELBOW = 1.0
CSV = "results/concurvity_warmstart/path_seed42.csv"
OUT_DIR = "results/concurvity_verification"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV).sort_values("lambda_c").reset_index(drop=True)
lam   = df["lambda_c"].values
auc   = df["val_auc"].values
rperp = df["r_perp_val"].values

# ── Criterion A: AUC-based ────────────────────────────────────────────────────
# Baseline = AUC at smallest lambda.  Elbow = largest lambda where AUC >= base-0.01.
baseline_auc_A = auc[0]
thresh_A = baseline_auc_A - 0.01

elbow_A_idx = None
for i in range(len(lam)):
    if auc[i] >= thresh_A:
        elbow_A_idx = i
elbow_A = lam[elbow_A_idx] if elbow_A_idx is not None else None

# Robustness variant: apply 5-point centred rolling mean first, then find elbow.
auc_smooth = pd.Series(auc).rolling(5, center=True, min_periods=1).mean().values
elbow_A_smooth_idx = None
for i in range(len(lam)):
    if auc_smooth[i] >= thresh_A:
        elbow_A_smooth_idx = i
elbow_A_smooth = lam[elbow_A_smooth_idx] if elbow_A_smooth_idx is not None else None

print("Criterion A (raw AUC):")
print(f"  baseline_auc = {baseline_auc_A:.4f}")
print(f"  threshold    = {thresh_A:.4f}")
print(f"  elbow_A      = {elbow_A:.4f}  (step {elbow_A_idx + 1})  "
      f"AUC there = {auc[elbow_A_idx]:.4f}")
print()
print("Criterion A (5-pt smoothed AUC):")
print(f"  elbow_A_smooth = {elbow_A_smooth:.4f}  (step {elbow_A_smooth_idx + 1})  "
      f"smoothed AUC there = {auc_smooth[elbow_A_smooth_idx]:.4f}")
print()

# ── Criterion B: R_perp saturation in log-log space ───────────────────────────
# Trim last 2 steps: R_perp at lambda~95 has an outlier drop that distorts the
# derivative estimate at the right boundary.
trim_end = 2
lam_b    = lam[:-trim_end]
rperp_b  = rperp[:-trim_end]

log_lam_b   = np.log(lam_b)
log_rperp_b = np.log(np.maximum(rperp_b, 1e-10))

# numpy.gradient uses central differences internally; clean for uniform log spacing.
slopes_b   = np.gradient(log_rperp_b, log_lam_b)
abs_slopes = np.abs(slopes_b)

peak_idx   = int(np.argmax(abs_slopes))
peak_slope = abs_slopes[peak_idx]
half_peak  = peak_slope / 2.0

# First point after peak where |slope| <= half_peak.
elbow_B_idx = None
for i in range(peak_idx + 1, len(lam_b)):
    if abs_slopes[i] <= half_peak:
        elbow_B_idx = i
        break
elbow_B = lam_b[elbow_B_idx] if elbow_B_idx is not None else lam_b[-1]

print("Criterion B (R_perp log-log saturation):")
print(f"  peak |slope| at lambda = {lam_b[peak_idx]:.4f}  magnitude = {peak_slope:.4f}")
print(f"  half-peak threshold    = {half_peak:.4f}")
if elbow_B_idx is not None:
    print(f"  elbow_B                = {elbow_B:.4f}  (step {elbow_B_idx + 1})  "
          f"|slope| there = {abs_slopes[elbow_B_idx]:.4f}")
else:
    print(f"  no post-peak half-peak crossing; using last trimmed step = {elbow_B:.4f}")
print()

print("  Slope profile (every 5 steps, trimmed range):")
print(f"  {'step':>5}  {'lambda':>10}  {'R_perp':>8}  {'slope':>8}  {'|slope|':>8}")
for i in range(0, len(lam_b), 5):
    mark = ""
    if i == peak_idx:
        mark = " <-- peak"
    elif elbow_B_idx is not None and i == elbow_B_idx:
        mark = " <-- elbow_B"
    print(f"  {i + 1:5d}  {lam_b[i]:10.4f}  {rperp_b[i]:8.4f}  "
          f"{slopes_b[i]:8.3f}  {abs_slopes[i]:8.3f}{mark}")
print()

# ── Criterion C: combined score ───────────────────────────────────────────────
# score(lambda) = (1 - R_perp/R0) - max(0, AUC0 - AUC) / 0.01
# Baseline at smallest lambda in path.
R0    = rperp[0]
AUC0  = auc[0]
scores = (1.0 - rperp / R0) - np.maximum(0.0, AUC0 - auc) / 0.01

best_idx_C = int(np.argmax(scores))
elbow_C    = lam[best_idx_C]

print("Criterion C (combined score):")
print(f"  R_perp_0 = {R0:.4f}  AUC_0 = {AUC0:.4f}")
print(f"  best score = {scores[best_idx_C]:.4f}  at lambda = {elbow_C:.4f}  "
      f"(step {best_idx_C + 1})")
print()
print("  Score profile (every 5 steps):")
print(f"  {'step':>5}  {'lambda':>10}  {'R_perp':>8}  {'AUC':>7}  {'score':>8}")
for i in range(0, len(lam), 5):
    mark = " <-- best" if i == best_idx_C else ""
    print(f"  {i + 1:5d}  {lam[i]:10.4f}  {rperp[i]:8.4f}  "
          f"{auc[i]:7.4f}  {scores[i]:8.4f}{mark}")
print()

# ── Elbow comparison ──────────────────────────────────────────────────────────
def log_ratio(e):
    if e is None:
        return float("inf")
    return abs(math.log10(e / COLD_START_ELBOW))

print("=" * 65)
print("ELBOW COMPARISON  (cold-start reference = 1.0)")
print("=" * 65)
cases = [
    ("A (raw AUC)",     elbow_A),
    ("A (smooth AUC)",  elbow_A_smooth),
    ("B (saturation)",  elbow_B),
    ("C (combined)",    elbow_C),
]
all_pass = True
for name, elbow in cases:
    lr   = log_ratio(elbow)
    flag = "ok" if lr <= 0.5 else "REPORT"
    if lr > 0.5:
        all_pass = False
    e_str = f"{elbow:.3f}" if elbow is not None else "None"
    print(f"  {name:18s}: lambda = {e_str:8s}  |log10| = {lr:.2f}  [{flag}]")
print(f"  Cold-start ref    : lambda = {COLD_START_ELBOW:.1f}")
print()
if all_pass:
    print("Verdict: ALL criteria within 0.5 orders of magnitude of 1.0.")
    print("Warm-start and cold-start AGREE on concurvity_lambda.")
else:
    print("Verdict: one or more criteria OUTSIDE 0.5 orders of magnitude.")
    for name, elbow in cases:
        if log_ratio(elbow) > 0.5:
            print(f"  Reporting: criterion {name} gives lambda = {elbow:.3f}")
print("=" * 65)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 5.5))

ax1.semilogx(lam, rperp, color="steelblue", linewidth=2.0, label="R_perp (val)", zorder=3)
ax1.set_xlabel("Concurvity lambda  (log scale)", fontsize=12)
ax1.set_ylabel("R_perp (val)", color="steelblue", fontsize=11)
ax1.tick_params(axis="y", labelcolor="steelblue")
ax1.set_ylim(bottom=0)

ax2 = ax1.twinx()
ax2.semilogx(lam, auc, color="darkorange", linewidth=1.2, linestyle="--", alpha=0.55,
             label="val AUC (raw)", zorder=2)
ax2.semilogx(lam, auc_smooth, color="darkorange", linewidth=2.0,
             label="val AUC (5-pt smooth)", zorder=3)
ax2.axhline(baseline_auc_A, color="darkorange", linestyle=":", linewidth=1.0, alpha=0.55,
            label=f"AUC baseline = {baseline_auc_A:.4f}")
ax2.axhline(thresh_A, color="red", linestyle=":", linewidth=1.0, alpha=0.55,
            label=f"AUC threshold (-0.01) = {thresh_A:.4f}")
ax2.set_ylabel("Val AUC  (OvR weighted)", color="darkorange", fontsize=11)
ax2.tick_params(axis="y", labelcolor="darkorange")

# Shade the expected-range band [0.3, 10]
ax1.axvspan(0.3, 10.0, alpha=0.07, color="gray", zorder=0,
            label="expected range [0.3, 10]")

# Vertical lines for each elbow
vlines = [
    (elbow_A,         "red",    "-",  f"A raw: {elbow_A:.3f}"),
    (elbow_A_smooth,  "crimson", "--", f"A smooth: {elbow_A_smooth:.3f}"),
    (elbow_B,         "green",  "-",  f"B: {elbow_B:.3f}"),
    (elbow_C,         "purple", "-",  f"C: {elbow_C:.3f}"),
    (COLD_START_ELBOW,"black",  "--", f"cold-start ref: {COLD_START_ELBOW:.1f}"),
]
for x, c, ls, lbl in vlines:
    if x is not None:
        ax1.axvline(x, color=c, linestyle=ls, linewidth=1.6, alpha=0.85, label=lbl)

lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labs1 + labs2,
           loc="upper right", fontsize=7.5, ncol=2, framealpha=0.9)

e_A_s = f"{elbow_A_smooth:.3f}" if elbow_A_smooth is not None else "None"
e_B_s = f"{elbow_B:.3f}" if elbow_B is not None else "None"
ax1.set_title(
    f"Warm-start concurvity path — seed 42 (re-analysis)\n"
    f"A(raw)={elbow_A:.3f}  A(smooth)={e_A_s}  "
    f"B={e_B_s}  C={elbow_C:.3f}  cold-start ref={COLD_START_ELBOW:.1f}",
    fontsize=10,
)
fig.tight_layout()
out_path = os.path.join(OUT_DIR, "path_seed42_v2.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"\n[plot] Saved: {out_path}")
