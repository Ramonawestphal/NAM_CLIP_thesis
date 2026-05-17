from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.data.compas import (
    compas_numeric_feature_matrix,
    find_compas_csv,
    load_compas,
    load_compas_frame,
)
from src.data.encoding import CompasEncoder
from src.nam import NAM, NeuralAdditiveModel
from src.nam.losses import total_loss
from src.utils.metrics import compute_auc_pr, compute_auc_roc
from src.utils.seeding import seed_everything


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Core training primitive
# ---------------------------------------------------------------------------

def _train_single_model(
    X_tr: torch.Tensor,
    y_tr: torch.Tensor,
    X_vl: torch.Tensor,
    y_vl: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
    verbose: bool = False,
) -> tuple[NAM, float, int]:
    """Train one NAM with early stopping on val AUC-ROC.

    Returns (model_at_best_val, best_val_auc, stopped_epoch).
    Encoder must be fitted externally; X_tr/X_vl are already encoded tensors.
    """
    n_features = X_tr.shape[1]
    model = NAM(
        n_features=n_features,
        dropout=float(cfg.get("dropout", 0.1)),
        feature_dropout=float(cfg.get("feature_dropout", 0.05)),
    ).to(device)

    output_reg = float(cfg.get("output_reg", 0.2078))
    lr = float(cfg.get("lr", 0.02082))
    lr_decay = float(cfg.get("lr_decay", 0.995))
    batch_size = int(cfg.get("batch_size", 1024))
    max_epochs = int(cfg.get("max_epochs", 1000))
    patience = int(cfg.get("early_stopping_patience", 60))
    balanced_sampling = bool(cfg.get("balanced_sampling", True))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)

    if balanced_sampling:
        y_tr_np = y_tr.cpu().numpy().astype(int)
        class_counts = np.bincount(y_tr_np)
        sample_weights = torch.tensor(
            1.0 / (2.0 * class_counts[y_tr_np]), dtype=torch.float32
        )
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(sample_weights), replacement=True
        )
        loader = DataLoader(
            TensorDataset(X_tr, y_tr), batch_size=batch_size, sampler=sampler, shuffle=False
        )
    else:
        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=batch_size, shuffle=True)

    best_val_auc = -1.0
    epochs_no_improve = 0
    best_state: dict | None = None
    stopped_epoch = max_epochs

    for epoch in range(max_epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = total_loss(model, xb, yb, output_reg=output_reg)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().cpu())
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(X_vl).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_auc = compute_auc_roc(y_vl.cpu().numpy(), val_probs)

        if verbose:
            val_pr = compute_auc_pr(y_vl.cpu().numpy(), val_probs)
            avg_loss = epoch_loss / max(len(loader), 1)
            print(
                f"epoch {epoch+1:4d}/{max_epochs}  loss={avg_loss:.4f}"
                f"  val_auc={val_auc:.4f}  val_pr={val_pr:.4f}"
            )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                stopped_epoch = epoch + 1
                if verbose:
                    print(
                        f"Early stopping at epoch {stopped_epoch}"
                        f" (no improvement for {patience} epochs)"
                    )
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val_auc, stopped_epoch


# ---------------------------------------------------------------------------
# Single-split paper path (quick diagnostic / ablation runs)
# ---------------------------------------------------------------------------

def run_paper_path(
    cfg: dict[str, Any],
    df,
    out_dir: str | Path,
    device: torch.device,
) -> tuple:
    """Train one NAM on a single stratified split. Quick diagnostic path.

    Pipeline:
      1. Outer 80/20 stratified split; inner 87.5/12.5 train/val split.
      2. Fit encoder on train; train NAM with early stopping on val AUC-ROC.
      3. Best checkpoint re-evaluated on held-out test set.
      4. Checkpoint saved to out_dir/nam_compas_paper.pt.

    Returns (model, encoder).
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    seed_everything(int(cfg.get("seed", 42)))

    y_all = df["two_year_recid"].values.astype(np.float32)
    X_df = df.drop(columns=["two_year_recid"])

    sss_outer = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, random_state=int(cfg.get("cv_seed", 42))
    )
    trval_idx, test_idx = next(sss_outer.split(X_df, y_all))

    X_trval, y_trval = X_df.iloc[trval_idx], y_all[trval_idx]
    sss_inner = StratifiedShuffleSplit(
        n_splits=1,
        test_size=float(cfg.get("val_size", 0.125)),
        random_state=int(cfg.get("val_seed", 1337)),
    )
    tr_idx, val_idx = next(sss_inner.split(X_trval, y_trval))

    X_train = X_trval.iloc[tr_idx]
    y_train = y_trval[tr_idx]
    X_val = X_trval.iloc[val_idx]
    y_val = y_trval[val_idx]
    X_test = X_df.iloc[test_idx]
    y_test = y_all[test_idx]

    encoder = CompasEncoder()
    X_tr_enc = torch.tensor(encoder.fit_transform(X_train), dtype=torch.float32, device=device)
    X_vl_enc = torch.tensor(encoder.transform(X_val), dtype=torch.float32, device=device)
    X_te_enc = torch.tensor(encoder.transform(X_test), dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_vl = torch.tensor(y_val, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test, dtype=torch.float32, device=device)

    model, best_val_auc, _ = _train_single_model(
        X_tr_enc, y_tr, X_vl_enc, y_vl, cfg, device, verbose=True
    )

    model.eval()
    with torch.no_grad():
        test_logits = model(X_te_enc).cpu().numpy()
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))
    test_auc_roc = compute_auc_roc(y_te.cpu().numpy(), test_probs)
    test_auc_pr = compute_auc_pr(y_te.cpu().numpy(), test_probs)
    print(f"\nTEST  AUC-ROC={test_auc_roc:.4f}  AUC-PR={test_auc_pr:.4f}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "nam_compas_paper.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "encoder": encoder,
            "feature_names": encoder.feature_names_,
            "config": cfg,
            "test_auc_roc": test_auc_roc,
            "test_auc_pr": test_auc_pr,
        },
        ckpt,
    )
    print(f"Saved {ckpt}")
    return model, encoder


# ---------------------------------------------------------------------------
# Full paper protocol: 5-fold CV x 20 val resamples per fold
# ---------------------------------------------------------------------------

def run_full_cv(
    cfg: dict[str, Any],
    df,
    out_dir: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    """Full paper CV protocol: StratifiedKFold(5) x StratifiedShuffleSplit(20).

    For each of the 5 folds:
      - Draw 20 random train/val splits from the fold's train+val pool.
      - Train one NAM per split (100 models total).
      - Ensemble fold predictions by averaging probabilities across the 20 models.
    Final ensemble AUC-ROC uses all 6172 held-out predictions (one per sample).

    Checkpoint saved to out_dir/nam_compas_cv.pt.
    Returns a results dict with per-fold metrics and the ensemble AUC.
    """
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

    seed_everything(int(cfg.get("seed", 42)))

    y_all = df["two_year_recid"].values.astype(np.float32)
    X_df = df.drop(columns=["two_year_recid"])

    n_folds = int(cfg.get("n_folds", 5))
    n_resamples = int(cfg.get("n_resamples", 20))
    cv_seed = int(cfg.get("cv_seed", 42))
    val_seed = int(cfg.get("val_seed", 1337))
    val_size = float(cfg.get("val_size", 0.125))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)

    # Accumulate one ensemble prediction per sample across folds
    ensemble_probs = np.zeros(len(y_all), dtype=np.float64)

    per_fold_results: list[dict] = []

    for fold_idx, (trval_idx, test_idx) in enumerate(skf.split(X_df, y_all)):
        fold_num = fold_idx + 1
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}/{n_folds}  ({len(trval_idx)} train+val, {len(test_idx)} test)")
        print(f"{'='*60}")

        X_trval = X_df.iloc[trval_idx]
        y_trval = y_all[trval_idx]
        y_test_fold = y_all[test_idx]

        sss = StratifiedShuffleSplit(
            n_splits=n_resamples, test_size=val_size, random_state=val_seed
        )

        # Probabilities on this fold's test set, averaged across resamples
        fold_test_probs = np.zeros(len(test_idx), dtype=np.float64)
        resample_val_aucs: list[float] = []

        for r_idx, (tr_idx, val_idx) in enumerate(sss.split(X_trval, y_trval)):
            X_train_df = X_trval.iloc[tr_idx]
            y_train_r = y_trval[tr_idx]
            X_val_df = X_trval.iloc[val_idx]
            y_val_r = y_trval[val_idx]
            X_test_df = X_df.iloc[test_idx]

            encoder = CompasEncoder()
            X_tr_enc = torch.tensor(
                encoder.fit_transform(X_train_df), dtype=torch.float32, device=device
            )
            X_vl_enc = torch.tensor(
                encoder.transform(X_val_df), dtype=torch.float32, device=device
            )
            X_te_enc = torch.tensor(
                encoder.transform(X_test_df), dtype=torch.float32, device=device
            )
            y_tr = torch.tensor(y_train_r, dtype=torch.float32, device=device)
            y_vl = torch.tensor(y_val_r, dtype=torch.float32, device=device)

            model, best_val_auc, stopped_ep = _train_single_model(
                X_tr_enc, y_tr, X_vl_enc, y_vl, cfg, device, verbose=False
            )
            resample_val_aucs.append(best_val_auc)

            model.eval()
            with torch.no_grad():
                test_logits = model(X_te_enc).cpu().numpy()
            fold_test_probs += 1.0 / (1.0 + np.exp(-test_logits))

            print(
                f"  fold {fold_num}/{n_folds}  resample {r_idx+1:2d}/{n_resamples}"
                f"  stopped ep={stopped_ep:4d}  val_auc={best_val_auc:.4f}"
            )

        fold_test_probs /= n_resamples
        ensemble_probs[test_idx] = fold_test_probs

        fold_auc_roc = compute_auc_roc(y_test_fold, fold_test_probs)
        fold_auc_pr = compute_auc_pr(y_test_fold, fold_test_probs)
        val_mean = float(np.mean(resample_val_aucs))
        val_std = float(np.std(resample_val_aucs))

        print(
            f"\nFold {fold_num} summary:"
            f"  test_auc_roc={fold_auc_roc:.4f}  test_auc_pr={fold_auc_pr:.4f}"
            f"  val={val_mean:.4f}+/-{val_std:.4f}"
        )
        per_fold_results.append(
            {
                "fold": fold_num,
                "test_auc_roc": fold_auc_roc,
                "test_auc_pr": fold_auc_pr,
                "val_auc_mean": val_mean,
                "val_auc_std": val_std,
                "val_aucs": resample_val_aucs,
            }
        )

    # Ensemble over all 6172 held-out predictions
    ensemble_auc_roc = compute_auc_roc(y_all, ensemble_probs)
    ensemble_auc_pr = compute_auc_pr(y_all, ensemble_probs)

    fold_rocs = [r["test_auc_roc"] for r in per_fold_results]
    fold_prs = [r["test_auc_pr"] for r in per_fold_results]

    print(f"\n{'='*60}")
    print("CV RESULTS")
    print(f"{'='*60}")
    for r in per_fold_results:
        print(
            f"  Fold {r['fold']}:  test_auc_roc={r['test_auc_roc']:.4f}"
            f"  test_auc_pr={r['test_auc_pr']:.4f}"
            f"  val={r['val_auc_mean']:.4f}+/-{r['val_auc_std']:.4f}"
        )
    print(
        f"\n  Mean fold AUC-ROC : {np.mean(fold_rocs):.4f} +/- {np.std(fold_rocs):.4f}"
    )
    print(
        f"  Mean fold AUC-PR  : {np.mean(fold_prs):.4f} +/- {np.std(fold_prs):.4f}"
    )
    print(f"  Ensemble AUC-ROC  : {ensemble_auc_roc:.4f}")
    print(f"  Ensemble AUC-PR   : {ensemble_auc_pr:.4f}")

    results = {
        "per_fold": per_fold_results,
        "mean_auc_roc": float(np.mean(fold_rocs)),
        "std_auc_roc": float(np.std(fold_rocs)),
        "mean_auc_pr": float(np.mean(fold_prs)),
        "std_auc_pr": float(np.std(fold_prs)),
        "ensemble_auc_roc": ensemble_auc_roc,
        "ensemble_auc_pr": ensemble_auc_pr,
        "ensemble_probs": ensemble_probs,
        "config": cfg,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "nam_compas_cv.pt"
    torch.save(results, ckpt)
    print(f"\nSaved {ckpt}")
    return results


# ---------------------------------------------------------------------------
# Legacy path (NeuralAdditiveModel, numeric columns only)
# ---------------------------------------------------------------------------

def _legacy_main(cfg: dict[str, Any]) -> None:
    device = torch.device(cfg.get("device", "cpu"))
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed)

    csv = find_compas_csv(cfg.get("data_dir"))
    if csv is None:
        raise FileNotFoundError(
            "No COMPAS CSV under data/compas. Add raw ProPublica file(s) and retry."
        )
    df = load_compas_frame(csv)
    X_df, feat_names = compas_numeric_feature_matrix(df)
    X = torch.tensor(X_df.values, dtype=torch.float32, device=device)
    n = X.shape[0]
    if "two_year_recid" in df.columns:
        y = torch.tensor(df["two_year_recid"].values, dtype=torch.float32, device=device)
    else:
        y = torch.zeros(n, dtype=torch.float32, device=device)

    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=int(cfg.get("batch_size", 256)), shuffle=True)

    model = NeuralAdditiveModel(
        num_features=X.shape[1],
        hidden_dims=tuple(cfg.get("hidden_dims", [64, 64])),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.get("lr", 1e-3)))
    loss_fn = nn.BCEWithLogitsLoss()

    epochs = int(cfg.get("epochs", 5))
    for epoch in range(epochs):
        running = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu())
        print(f"epoch {epoch + 1}/{epochs} loss={running / max(len(loader), 1):.4f}")

    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "nam_compas_last.pt"
    torch.save(
        {"model": model.state_dict(), "feature_names": feat_names, "config": cfg},
        ckpt,
    )
    print(f"saved {ckpt}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="clip-nam-thesis training entry point")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML or JSON config")
    p.add_argument(
        "--single-split",
        action="store_true",
        help="Quick single-split diagnostic run (skips full CV)",
    )
    p.add_argument("--legacy", action="store_true", help="Use legacy NeuralAdditiveModel path")
    p.add_argument("--data-dir", type=Path, default=None, help="Override data directory")
    args = p.parse_args()
    cfg = load_config(args.config)

    if args.legacy:
        _legacy_main(cfg)
        return

    device = torch.device(cfg.get("device", "cpu"))
    data_dir = args.data_dir or Path(cfg.get("data_dir", "data/compas"))

    raw_csv = find_compas_csv(data_dir)
    if raw_csv is None:
        raise FileNotFoundError(
            f"No COMPAS CSV found under {data_dir}.\n"
            "Download compas-scores-two-years.csv from "
            "https://github.com/propublica/compas-analysis and place it there."
        )

    out_dir = Path(cfg.get("output_dir", "results"))
    clean_csv = out_dir / "compas_clean_v1.csv"
    print(f"Loading and cleaning {raw_csv} ...")
    df = load_compas(raw_csv, clean_csv)
    print(f"Cleaned DataFrame: {len(df)} rows, {df.shape[1]} columns")

    if args.single_split:
        run_paper_path(cfg, df, out_dir=out_dir, device=device)
    else:
        run_full_cv(cfg, df, out_dir=out_dir, device=device)


if __name__ == "__main__":
    main()
