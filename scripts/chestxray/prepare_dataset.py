"""
Prepare the Kermany Chest X-ray dataset with patient-ID-aware splits.

Mirrors the HAM10000 v7 protocol:
  - Outer 80/20 train-pool / test split by patient (stratified by binary label)
  - Pre-defined 5-fold StratifiedGroupKFold CV over the train pool

Dataset expected at: data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/

Outputs:
    data/splits/chestxray_outer_split.npz
    data/splits/chestxray_cv_folds.npz

Run from project root:
    python scripts/chestxray/prepare_dataset.py
"""

from __future__ import annotations

import pathlib
import re
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT    = _ROOT / "data/chest_xray/"
SPLITS_DIR   = _ROOT / "data/splits"
OUTER_NPZ    = SPLITS_DIR / "chestxray_outer_split.npz"
CV_NPZ       = SPLITS_DIR / "chestxray_cv_folds.npz"

RANDOM_SEED  = 42
N_FOLDS      = 5
TEST_FRAC    = 0.20


# ── Patient-ID extraction ─────────────────────────────────────────────────────

def extract_patient_id(stem: str) -> str:
    """Derive a stable patient_id from an image filename stem.

    Rules (in priority order):
        person{ID}_*        → person{ID}
        NORMAL2-IM-{ID}-*   → normal-NORMAL2-IM-{ID}
        IM-{ID}-*           → normal-IM-{ID}
        anything else       → unknown-{stem}
    """
    m = re.match(r"^(person\d+)_", stem)
    if m:
        return m.group(1)

    m = re.match(r"^(NORMAL2-IM-\d+)-", stem)
    if m:
        return f"normal-{m.group(1)}"

    m = re.match(r"^(IM-\d+)-", stem)
    if m:
        return f"normal-{m.group(1)}"

    print(f"[prepare_dataset] unrecognised filename pattern, assigning unknown-{stem}")
    return f"unknown-{stem}"


def label_from_path(rel: pathlib.Path) -> tuple[int, str]:
    """Return (label_binary, label_subtype) from a relative image path."""
    parts = rel.parts
    # parts: (split_dir, class_dir, filename)
    class_dir = parts[1].upper()
    stem = rel.stem

    if class_dir == "NORMAL":
        return 0, "normal"

    # PNEUMONIA — distinguish bacteria vs virus by filename
    if "_bacteria_" in stem:
        return 1, "bacteria"
    if "_virus_" in stem:
        return 1, "virus"
    # Fallback: generic pneumonia (shouldn't appear in Kermany dataset)
    return 1, "pneumonia"


# ── Collect all images ────────────────────────────────────────────────────────

def collect_images(data_root: pathlib.Path) -> dict:
    """Scan train/val/test directories and return ordered metadata arrays."""
    records: list[dict] = []
    orig_split_map: dict[str, str] = {}  # patient_id → orig_split

    for split_name in ("train", "val", "test"):
        split_dir = data_root / split_name
        if not split_dir.exists():
            print(f"[prepare_dataset] WARNING: directory not found: {split_dir}")
            continue

        for class_dir in ("NORMAL", "PNEUMONIA"):
            class_path = split_dir / class_dir
            if not class_path.exists():
                continue
            for img_path in sorted(class_path.glob("*.jpeg")):
                rel = pathlib.Path(split_name) / class_dir / img_path.name
                stem = img_path.stem
                patient_id = extract_patient_id(stem)
                label_bin, label_sub = label_from_path(rel)

                # Track patient's original Kermany split assignment
                if patient_id in orig_split_map and orig_split_map[patient_id] != split_name:
                    orig_split_map[patient_id] = "multiple"  # overlaps
                else:
                    orig_split_map[patient_id] = split_name

                records.append({
                    "path":          str(rel),
                    "label_binary":  label_bin,
                    "label_subtype": label_sub,
                    "patient_id":    patient_id,
                    "orig_split":    split_name,
                })

    # Sort by relative path for reproducibility (already sorted per glob, but
    # we need a global sort across all three directories)
    records.sort(key=lambda r: r["path"])

    paths          = np.array([r["path"]          for r in records])
    labels_binary  = np.array([r["label_binary"]  for r in records], dtype=np.int32)
    labels_subtype = np.array([r["label_subtype"] for r in records])
    patient_ids    = np.array([r["patient_id"]    for r in records])
    orig_splits    = np.array([r["orig_split"]    for r in records])

    return {
        "paths":          paths,
        "labels_binary":  labels_binary,
        "labels_subtype": labels_subtype,
        "patient_ids":    patient_ids,
        "orig_splits":    orig_splits,
        "orig_split_map": orig_split_map,
    }


# ── Stratified-group outer split ──────────────────────────────────────────────

def stratified_group_outer_split(
    patient_ids: np.ndarray,
    labels_binary: np.ndarray,
    test_frac: float = 0.20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """80/20 patient-level split stratified by binary label.

    Uses StratifiedGroupKFold with n_splits=5 (each fold ~ 20% test) and
    takes the first fold as the outer test set, matching the v7 protocol.
    """
    unique_patients = np.unique(patient_ids)
    # One label per patient — majority vote (all images of a patient share the
    # same binary label in this dataset)
    patient_label: dict[str, int] = {}
    for pid, lab in zip(patient_ids, labels_binary):
        patient_label[pid] = lab  # deterministic; same label per patient

    pt_labels = np.array([patient_label[p] for p in unique_patients])

    # Use StratifiedGroupKFold on patient-level arrays then expand to images
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)
    # Each element is its own group (patients are the units)
    first_split = next(iter(sgkf.split(unique_patients, pt_labels, groups=unique_patients)))
    train_pt_idx, test_pt_idx = first_split

    train_patients = set(unique_patients[train_pt_idx])
    test_patients  = set(unique_patients[test_pt_idx])

    train_pool_idx = np.where(np.isin(patient_ids, list(train_patients)))[0]
    test_idx       = np.where(np.isin(patient_ids, list(test_patients)))[0]

    return train_pool_idx, test_idx


# ── 5-fold CV on train pool ───────────────────────────────────────────────────

def build_cv_folds(
    global_idx: np.ndarray,
    patient_ids: np.ndarray,
    labels_binary: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build n_splits StratifiedGroupKFold folds over the train pool.

    Returns list of (fold_train_global_idx, fold_val_global_idx) tuples,
    where indices are into the *global* path array (not the pool's local space).
    """
    pool_patient_ids   = patient_ids[global_idx]
    pool_labels_binary = labels_binary[global_idx]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds: list[tuple[np.ndarray, np.ndarray]] = []

    for local_train, local_val in sgkf.split(
        global_idx, pool_labels_binary, groups=pool_patient_ids
    ):
        global_train = global_idx[local_train]
        global_val   = global_idx[local_val]
        folds.append((global_train, global_val))

    return folds


# ── Diagnostics ───────────────────────────────────────────────────────────────

def _class_proportions(labels: np.ndarray) -> str:
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    parts = [f"{u}={c}({c/total*100:.1f}%)" for u, c in zip(unique, counts)]
    return "  ".join(parts)


def print_diagnostics(
    data:          dict,
    train_pool_idx: np.ndarray,
    test_idx:       np.ndarray,
    folds:          list,
) -> None:
    paths          = data["paths"]
    labels_binary  = data["labels_binary"]
    labels_subtype = data["labels_subtype"]
    patient_ids    = data["patient_ids"]
    orig_split_map = data["orig_split_map"]

    N = len(paths)
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Total images: {N}")
    print(f"Binary labels:  {_class_proportions(labels_binary)}")
    print(f"Subtype labels: {_class_proportions(labels_subtype)}")

    unique_pids = np.unique(patient_ids)
    print(f"\nTotal patients: {len(unique_pids)}")
    # Per-class patient count
    pid_labels: dict[str, int] = {}
    for pid, lab in zip(patient_ids, labels_binary):
        pid_labels[pid] = lab
    pid_label_arr = np.array([pid_labels[p] for p in unique_pids])
    u, c = np.unique(pid_label_arr, return_counts=True)
    for cls, cnt in zip(u, c):
        name = "NORMAL" if cls == 0 else "PNEUMONIA"
        print(f"  {name}: {cnt} patients")

    # Kermany original split overlap
    multi_patients = [p for p, s in orig_split_map.items() if s == "multiple"]
    print(f"\n--- Kermany original-split overlap diagnostic ---")
    print(f"  Patients appearing in >1 original split: {len(multi_patients)}")
    if multi_patients[:5]:
        print(f"  Examples: {multi_patients[:5]}")

    # New outer split
    print(f"\n--- New outer split (patient-aware 80/20) ---")
    print(f"  Train pool: {len(train_pool_idx)} images")
    print(f"    {_class_proportions(labels_binary[train_pool_idx])}")
    print(f"  Test:       {len(test_idx)} images")
    print(f"    {_class_proportions(labels_binary[test_idx])}")

    # 5-fold CV
    print(f"\n--- 5-fold CV on train pool ---")
    for i, (tr, va) in enumerate(folds):
        print(f"  Fold {i}: train={len(tr)}  val={len(va)}  "
              f"val_class: {_class_proportions(labels_binary[va])}")

    # Zero-overlap verifications
    print("\n--- Zero-overlap verifications ---")
    train_pids = set(patient_ids[train_pool_idx])
    test_pids  = set(patient_ids[test_idx])
    overlap_outer = train_pids & test_pids
    print(f"  [1] Train pool ∩ Test patients: {len(overlap_outer)} "
          f"({'PASS ✓' if not overlap_outer else 'FAIL ✗'})")

    all_ok = not overlap_outer
    for i, (tr, va) in enumerate(folds):
        tr_pids = set(patient_ids[tr])
        va_pids = set(patient_ids[va])
        fold_overlap = tr_pids & va_pids
        ok = not fold_overlap
        all_ok = all_ok and ok
        print(f"  [2] Fold {i} train ∩ val patients: {len(fold_overlap)} "
              f"({'PASS ✓' if ok else 'FAIL ✗'})")

    union_idx = np.union1d(train_pool_idx, test_idx)
    full_ok = len(union_idx) == N and union_idx[0] == 0 and union_idx[-1] == N - 1
    print(f"  [3] train_pool ∪ test == range({N}): "
          f"{'PASS ✓' if full_ok else 'FAIL ✗'}")


# ── Sanity checks before writing ──────────────────────────────────────────────

def run_sanity_checks(
    patient_ids:    np.ndarray,
    labels_binary:  np.ndarray,
    train_pool_idx: np.ndarray,
    test_idx:       np.ndarray,
    folds:          list,
    N:              int,
    tol_pp:         float = 5.0,
) -> None:
    """Assert all integrity conditions; raise AssertionError on any failure."""
    # 1. No patient overlap outer split
    train_pids = set(patient_ids[train_pool_idx])
    test_pids  = set(patient_ids[test_idx])
    assert not (train_pids & test_pids), "Patient overlap between train pool and test!"

    # 2. No patient overlap within each fold
    for i, (tr, va) in enumerate(folds):
        tr_pids = set(patient_ids[tr])
        va_pids = set(patient_ids[va])
        assert not (tr_pids & va_pids), f"Patient overlap in fold {i} train/val!"

    # 3. train_pool ∪ test == range(N)
    union_idx = np.union1d(train_pool_idx, test_idx)
    assert len(union_idx) == N, f"union size {len(union_idx)} != N {N}"
    assert union_idx[0] == 0 and union_idx[-1] == N - 1, "Index gap in train+test"

    # 4. Each fold's train ∪ val == train pool
    train_pool_set = set(train_pool_idx)
    for i, (tr, va) in enumerate(folds):
        fold_union = set(tr) | set(va)
        assert fold_union == train_pool_set, f"Fold {i} does not partition the train pool!"

    # 5. Class proportions within tol_pp percentage points
    global_prop = labels_binary.mean()
    for name, idx in [("train_pool", train_pool_idx), ("test", test_idx)]:
        prop = labels_binary[idx].mean()
        assert abs(prop - global_prop) * 100 <= tol_pp, (
            f"{name} pneumonia proportion {prop:.3f} differs from global "
            f"{global_prop:.3f} by >{tol_pp} pp"
        )
    for i, (_, va) in enumerate(folds):
        prop = labels_binary[va].mean()
        assert abs(prop - global_prop) * 100 <= tol_pp, (
            f"Fold {i} val proportion {prop:.3f} differs from global "
            f"{global_prop:.3f} by >{tol_pp} pp"
        )

    print("\nAll sanity checks passed ✓")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not DATA_ROOT.exists():
        sys.exit(
            f"ERROR: dataset root not found: {DATA_ROOT}\n"
            "Download the Kermany Chest X-ray dataset and place it at data/chest_xray/\n"
            "Expected structure: data/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}/*.jpeg\n"
            "Dataset available at: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia"
        )

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)

    print("Collecting images...")
    data = collect_images(DATA_ROOT)
    paths         = data["paths"]
    labels_binary = data["labels_binary"]
    patient_ids   = data["patient_ids"]
    N = len(paths)
    print(f"Found {N} images across {len(np.unique(patient_ids))} patients")

    print("Building outer split...")
    train_pool_idx, test_idx = stratified_group_outer_split(
        patient_ids, labels_binary,
        test_frac=TEST_FRAC, random_state=RANDOM_SEED,
    )

    print("Building 5-fold CV on train pool...")
    folds = build_cv_folds(
        train_pool_idx, patient_ids, labels_binary,
        n_splits=N_FOLDS, random_state=RANDOM_SEED,
    )

    print_diagnostics(data, train_pool_idx, test_idx, folds)
    run_sanity_checks(patient_ids, labels_binary, train_pool_idx, test_idx, folds, N)

    # Write outer split
    print(f"\nSaving outer split → {OUTER_NPZ.relative_to(_ROOT)}")
    np.savez(
        OUTER_NPZ,
        train_pool_idx = train_pool_idx,
        test_idx       = test_idx,
        paths          = paths,
        labels_binary  = labels_binary,
        labels_subtype = data["labels_subtype"],
        patient_ids    = patient_ids,
    )

    # Write CV folds
    print(f"Saving CV folds   → {CV_NPZ.relative_to(_ROOT)}")
    fold_arrays: dict[str, np.ndarray] = {}
    for i, (tr, va) in enumerate(folds):
        fold_arrays[f"fold_train_idx_{i}"] = tr
        fold_arrays[f"fold_val_idx_{i}"]   = va
    np.savez(CV_NPZ, **fold_arrays)

    print("\nDone.")
    print(f"  Outer split: train_pool={len(train_pool_idx)}  test={len(test_idx)}")
    for i, (_, va) in enumerate(folds):
        print(f"  Fold {i} val size: {len(va)}")


if __name__ == "__main__":
    main()
