from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
# Paper-faithful training path
# ---------------------------------------------------------------------------

def run_paper_path(
    cfg: dict[str, Any],
    df,  # cleaned COMPAS DataFrame (6 features + two_year_recid)
    out_dir: str | Path,
    device: torch.device,
) -> tuple:
    """Train a paper-faithful NAM on a pre-cleaned COMPAS DataFrame.

    Pipeline:
      1. Stratified train / val / test split (fit encoder on train only).
      2. NAM with hyperparameters from cfg.
      3. Training loop: total_loss (BCE + output penalty) + LR decay.
      4. Early stopping on val AUC-ROC; best checkpoint re-evaluated on test set.
      5. Checkpoint saved to out_dir/nam_compas_paper.pt.

    Returns (model, encoder) after loading the best checkpoint.
    """
    from sklearn.model_selection import StratifiedShuffleSplit

    seed_everything(int(cfg.get("seed", 42)))

    y_all = df["two_year_recid"].values.astype(np.float32)
    X_df = df.drop(columns=["two_year_recid"])

    # Outer split: 80% train+val, 20% test (matches one CV fold from the contract)
    sss_outer = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(cfg.get("cv_seed", 42)))
    trval_idx, test_idx = next(sss_outer.split(X_df, y_all))

    # Inner split: 87.5% train, 12.5% val of the train+val pool
    X_trval, y_trval = X_df.iloc[trval_idx], y_all[trval_idx]
    sss_inner = StratifiedShuffleSplit(
        n_splits=1, test_size=float(cfg.get("val_size", 0.125)), random_state=int(cfg.get("val_seed", 1337))
    )
    tr_idx, val_idx = next(sss_inner.split(X_trval, y_trval))

    X_train = X_trval.iloc[tr_idx]
    y_train = y_trval[tr_idx]
    X_val = X_trval.iloc[val_idx]
    y_val = y_trval[val_idx]
    X_test = X_df.iloc[test_idx]
    y_test = y_all[test_idx]

    # Encode — fit on train only
    encoder = CompasEncoder()
    X_tr_enc = torch.tensor(encoder.fit_transform(X_train), dtype=torch.float32, device=device)
    X_vl_enc = torch.tensor(encoder.transform(X_val), dtype=torch.float32, device=device)
    X_te_enc = torch.tensor(encoder.transform(X_test), dtype=torch.float32, device=device)
    y_tr = torch.tensor(y_train, dtype=torch.float32, device=device)
    y_vl = torch.tensor(y_val, dtype=torch.float32, device=device)
    y_te = torch.tensor(y_test, dtype=torch.float32, device=device)

    # Model
    model = NAM(
        n_features=encoder.n_features,
        dropout=float(cfg.get("dropout", 0.1)),
        feature_dropout=float(cfg.get("feature_dropout", 0.05)),
    ).to(device)

    output_reg = float(cfg.get("output_reg", 0.2078))
    lr = float(cfg.get("lr", 0.02082))
    lr_decay = float(cfg.get("lr_decay", 0.995))
    batch_size = int(cfg.get("batch_size", 1024))
    max_epochs = int(cfg.get("max_epochs", 1000))
    patience = int(cfg.get("early_stopping_patience", 60))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_decay)
    loader = DataLoader(TensorDataset(X_tr_enc, y_tr), batch_size=batch_size, shuffle=True)

    best_val_auc = -1.0
    epochs_no_improve = 0
    best_state: dict | None = None

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
            val_logits = model(X_vl_enc).cpu().numpy()
        val_probs = 1.0 / (1.0 + np.exp(-val_logits))
        val_auc = compute_auc_roc(y_vl.cpu().numpy(), val_probs)
        val_pr = compute_auc_pr(y_vl.cpu().numpy(), val_probs)

        avg_loss = epoch_loss / max(len(loader), 1)
        print(f"epoch {epoch+1:4d}/{max_epochs}  loss={avg_loss:.4f}  val_auc={val_auc:.4f}  val_pr={val_pr:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch+1} (no improvement for {patience} epochs)")
                break

    # Reload best checkpoint and evaluate on held-out test set
    if best_state is not None:
        model.load_state_dict(best_state)

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
            "encoder": encoder,          # CompasEncoder is pickle-safe
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
    p.add_argument("--legacy", action="store_true", help="Use legacy NeuralAdditiveModel path")
    p.add_argument("--data-dir", type=Path, default=None, help="Override data directory")
    args = p.parse_args()
    cfg = load_config(args.config)

    if args.legacy:
        _legacy_main(cfg)
        return

    # --- Paper-faithful path ---
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
    print(f"Loading and cleaning {raw_csv} …")
    df = load_compas(raw_csv, clean_csv)
    print(f"Cleaned DataFrame: {len(df)} rows, {df.shape[1]} columns")

    run_paper_path(cfg, df, out_dir=out_dir, device=device)


if __name__ == "__main__":
    main()
