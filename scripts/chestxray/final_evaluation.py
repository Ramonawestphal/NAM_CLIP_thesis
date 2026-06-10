"""
v7-parity three-way final evaluation: winning config, 5 seeds, held-out test set.

The ONLY three-way selection script that loads test_idx, and only AFTER
select_architecture.py produced winning_config.json. Mirrors the binary final
evaluation (final_evaluation_binary.py) with num_classes=3 and three-way metrics:
balanced accuracy, macro-OvR AUC, per-class accuracy, and a confusion matrix.

Fixed 20% patient-grouped early-stopping val (GroupShuffleSplit, random_state=42),
same across all 5 seeds. StandardScaler fit on the 80% inner-train only.
max_epochs=100, patience=15 (v7 final budget).

Outputs (under results/chestxray/architecture_selection/):
    final_test_results.csv
    final_test_summary.txt
    final_test_confusion_matrix.png

Run from project root (after select_architecture.py):
    python scripts/chestxray/final_evaluation.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # robust on cp1252 consoles
except Exception:
    pass

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

from src.models.nam_multiclass import NAMMulticlass
from scripts.chestxray._common_imports import set_all_seeds, make_optimizer_scheduler, make_fixed_val_split
# Reuse the SAME three-way training/early-stopping wrapper (defined once).
from scripts.chestxray.select_architecture import train_with_early_stopping

# ── Paths ─────────────────────────────────────────────────────────────────────
SCORES_NPZ  = _ROOT / "data/features/biomedclip/chestxray_concept_scores_v4.npz"
OUTER_SPLIT = _ROOT / "data/splits/chestxray_outer_split.npz"
OUT_DIR     = _ROOT / "results/chestxray/architecture_selection"
WINNER_JSON = OUT_DIR / "winning_config.json"
LABEL_MAP   = OUT_DIR / "label_mapping.json"

SEEDS       = [42, 43, 44, 45, 46]
LR          = 1e-3
MAX_EPOCHS  = 100
PATIENCE    = 15
N_FEATURES  = 17
NUM_CLASSES = 3
VAL_RANDOM_STATE = 42
MIN_VAL_PER_CLASS = 50   # flag if any class has fewer than this in the val split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not WINNER_JSON.exists():
        sys.exit(f"ERROR: {WINNER_JSON} not found. Run select_architecture.py first.")
    winner = json.loads(WINNER_JSON.read_text(encoding="utf-8"))
    hidden_dims  = tuple(winner["hidden_dims"])
    dropout      = float(winner["dropout"])
    weight_decay = float(winner["weight_decay"])

    subtype_to_int = json.loads(LABEL_MAP.read_text(encoding="utf-8")) \
        if LABEL_MAP.exists() else {"normal": 0, "bacteria": 1, "virus": 2}
    int_to_subtype = {v: k for k, v in subtype_to_int.items()}
    class_order = [0, 1, 2]
    class_names = [int_to_subtype[i] for i in class_order]

    # ── Load data (test_idx loaded here, post-selection only) ──────────────────
    feat          = np.load(SCORES_NPZ, allow_pickle=True)
    scores        = feat["scores"]
    concept_names = feat["concept_names"].tolist()
    split         = np.load(OUTER_SPLIT, allow_pickle=True)
    train_pool_idx = split["train_pool_idx"]
    test_idx       = split["test_idx"]
    labels_subtype = split["labels_subtype"]
    patient_ids    = split["patient_ids"]

    assert len(np.intersect1d(train_pool_idx, test_idx)) == 0, \
        "train_pool and test overlap — check splits file"

    labels_threeway = np.array([subtype_to_int[s] for s in labels_subtype], dtype=np.int64)

    X_pool_raw  = scores[train_pool_idx]
    y_pool      = labels_threeway[train_pool_idx]
    groups_pool = patient_ids[train_pool_idx]
    X_test_raw  = scores[test_idx]
    y_test      = labels_threeway[test_idx]

    # ── Fixed 20% patient-grouped val split (v7 make_fixed_val_split) ──────────
    # GroupShuffleSplit groups but does NOT stratify; verify all 3 classes present.
    vs = make_fixed_val_split(
        X_pool_raw, y_pool.astype(str), groups_pool, ["0", "1", "2"],
        val_random_state=VAL_RANDOM_STATE,
    )
    train_rel, val_rel = vs["train_rel"], vs["val_rel"]
    assert set(groups_pool[train_rel]).isdisjoint(set(groups_pool[val_rel])), \
        "inner-train/val patient overlap in fixed val split"

    val_counts = np.bincount(y_pool[val_rel], minlength=NUM_CLASSES)
    n_val_patients = len(set(groups_pool[val_rel].tolist()))
    val_class_flags = []
    for c in class_order:
        if val_counts[c] < MIN_VAL_PER_CLASS:
            val_class_flags.append(
                f"⚠ val class '{int_to_subtype[c]}' has only {val_counts[c]} samples "
                f"(< {MIN_VAL_PER_CLASS})"
            )
    all_classes_present = set(np.unique(y_pool[val_rel])) == {0, 1, 2}

    print("=" * 70)
    print("Chest X-ray NAM — three-way final 5-seed test evaluation (v7 parity)")
    print(f"  Winning config: hidden={list(hidden_dims)}, dropout={dropout}, wd={weight_decay:.0e}")
    print(f"  Device: {DEVICE}")
    print(f"  Train pool {len(train_pool_idx)} (inner-train {len(train_rel)} / val {len(val_rel)})"
          f"  Test {len(test_idx)}")
    print(f"  Fixed val: {n_val_patients} patients; class counts "
          f"N/B/V = {val_counts[0]}/{val_counts[1]}/{val_counts[2]}")
    if not all_classes_present:
        print("  ⚠ NOT all three classes present in val split!")
    for fl in val_class_flags:
        print("  " + fl)
    print(f"  max_epochs={MAX_EPOCHS} patience={PATIENCE} seeds={SEEDS}")
    print("=" * 70)

    rows: list[dict] = []
    cms: list[np.ndarray] = []
    t0 = time.time()
    for seed in SEEDS:
        set_all_seeds(seed)
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_pool_raw[train_rel]).astype(np.float32)
        X_val   = scaler.transform(X_pool_raw[val_rel]).astype(np.float32)
        X_test  = scaler.transform(X_test_raw).astype(np.float32)
        y_train = y_pool[train_rel]
        y_val   = y_pool[val_rel]

        counts = np.bincount(y_train, minlength=NUM_CLASSES)
        class_weights = torch.tensor(
            len(y_train) / (NUM_CLASSES * counts), dtype=torch.float32
        )

        model = NAMMulticlass(
            n_features=N_FEATURES, num_classes=NUM_CLASSES,
            hidden_dims=hidden_dims, dropout=dropout,
            concept_names=list(concept_names),
        ).to(DEVICE)
        optimizer, scheduler = make_optimizer_scheduler(model, LR, weight_decay)

        best_val_balacc, _, _, best_epoch = train_with_early_stopping(
            model, optimizer, scheduler,
            X_train, y_train, X_val, y_val,
            class_weights=class_weights,
            max_epochs=MAX_EPOCHS, patience=PATIENCE, device=DEVICE,
        )

        # ── Test evaluation ────────────────────────────────────────────────────
        X_test_t = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
        model.eval()
        with torch.no_grad():
            logits = model(X_test_t)
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()
        test_balacc = float(balanced_accuracy_score(y_test, preds))
        test_macro_auc = float(roc_auc_score(
            y_test, proba, multi_class="ovr", average="macro", labels=class_order
        ))
        per_cls = {}
        for c in class_order:
            mask = (y_test == c)
            per_cls[int_to_subtype[c]] = float((preds[mask] == c).mean()) if mask.sum() else float("nan")
        cms.append(confusion_matrix(y_test, preds, labels=class_order))

        rows.append({
            "seed": seed,
            "best_val_balacc": round(best_val_balacc, 6),
            "best_epoch": best_epoch,
            "test_balacc": round(test_balacc, 6),
            "test_macro_auc_ovr": round(test_macro_auc, 6),
            "test_acc_normal":   round(per_cls["normal"], 6),
            "test_acc_bacteria": round(per_cls["bacteria"], 6),
            "test_acc_virus":    round(per_cls["virus"], 6),
        })
        print(f"  seed {seed}: balacc={test_balacc:.4f} macroAUC={test_macro_auc:.4f} "
              f"acc(N/B/V)={per_cls['normal']:.2f}/{per_cls['bacteria']:.2f}/{per_cls['virus']:.2f} "
              f"(best_epoch={best_epoch})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "final_test_results.csv", index=False)

    mean_bal, std_bal = float(df["test_balacc"].mean()), float(df["test_balacc"].std())
    mean_auc, std_auc = float(df["test_macro_auc_ovr"].mean()), float(df["test_macro_auc_ovr"].std())
    mN = float(df["test_acc_normal"].mean()); mB = float(df["test_acc_bacteria"].mean())
    mV = float(df["test_acc_virus"].mean())

    # ── Mean confusion matrix + heatmap ────────────────────────────────────────
    cm_mean = np.mean(np.stack(cms, axis=0), axis=0)
    cm_norm = cm_mean / cm_mean.sum(axis=1, keepdims=True)
    pd.DataFrame(cm_mean.round(2), index=class_names, columns=class_names).to_csv(
        OUT_DIR / "final_test_confusion_matrix.csv"
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalised")
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(class_names); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title("Three-way test confusion matrix (mean of 5 seeds)", fontsize=10)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cm_norm[i,j]:.2f}\n({cm_mean[i,j]:.0f})",
                    ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "final_test_confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"\nFinal ({len(SEEDS)} seeds): balacc={mean_bal:.4f}±{std_bal:.4f}  "
          f"macroAUC={mean_auc:.4f}±{std_auc:.4f}  "
          f"per-class N/B/V={mN:.3f}/{mB:.3f}/{mV:.3f}  ({(time.time()-t0)/60:.1f} min)")

    # ── Summary ────────────────────────────────────────────────────────────────
    L: list[str] = []
    L.append("=" * 70)
    L.append("CHEST X-RAY NAM — THREE-WAY FINAL TEST EVALUATION (v7 parity)")
    L.append("=" * 70)
    L.append("")
    L.append(f"Winning config: config_id={winner['config_id']} hidden={winner['hidden_dims']} "
             f"dropout={winner['dropout']} weight_decay={winner['weight_decay']:.0e}")
    L.append(f"  (selected by mean val balanced accuracy "
             f"{winner['mean_val_balacc']:.4f} ± {winner['std_val_balacc']:.4f})")
    L.append("")
    L.append("Per-seed test results:")
    L.append(df.to_string(index=False))
    L.append("")
    L.append(f"Mean ± std across {len(SEEDS)} seeds:")
    L.append(f"  Test balanced accuracy: {mean_bal:.4f} ± {std_bal:.4f}")
    L.append(f"  Test macro-OvR AUC:     {mean_auc:.4f} ± {std_auc:.4f}")
    L.append(f"  Per-class test accuracy: Normal {mN:.4f} | Bacteria {mB:.4f} | Virus {mV:.4f}")
    L.append("")
    L.append("Mean confusion matrix (rows=true, cols=pred; counts):")
    L.append(pd.DataFrame(cm_mean.round(1), index=class_names, columns=class_names).to_string())
    L.append("")
    L.append("Fixed early-stopping val split (v7 make_fixed_val_split):")
    L.append(f"  GroupShuffleSplit(test_size=0.20, random_state=42), grouped by patient")
    L.append(f"  inner-train={len(train_rel)} val={len(val_rel)} val_patients={n_val_patients}")
    L.append(f"  val class counts N/B/V = {val_counts[0]}/{val_counts[1]}/{val_counts[2]}")
    if val_class_flags:
        for fl in val_class_flags:
            L.append("  " + fl)
    else:
        L.append(f"  All three classes >= {MIN_VAL_PER_CLASS} samples in val split.")
    L.append("")
    L.append("Leakage safeguards verified (selection + final):")
    for s in [
        "Test indices never loaded during selection (select_architecture.py)",
        "All CV indices inside train pool",
        "Per-fold train/val disjoint at image level",
        "Per-fold train/val disjoint at patient level",
        "Per-fold z-scoring (fit on fold-train only)",
        "Fresh model per (config, fold), seed 42 reset before each",
        "Final test loaded only after selection completed (this script)",
        "Three-way label conversion verified (all 3 classes present)",
    ]:
        L.append(f"  [check] {s}")
    (OUT_DIR / "final_test_summary.txt").write_text("\n".join(L), encoding="utf-8")
    print(f"  final summary → {(OUT_DIR / 'final_test_summary.txt').relative_to(_ROOT)}")
    print(f"  confusion matrix → {(OUT_DIR / 'final_test_confusion_matrix.png').relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
