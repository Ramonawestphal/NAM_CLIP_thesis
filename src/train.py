from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.compas import compas_numeric_feature_matrix, find_compas_csv, load_compas_frame
from src.nam import NeuralAdditiveModel


def load_config(path: Path) -> dict[str, Any]:
    if path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    p = argparse.ArgumentParser(description="clip-nam-thesis training entry point")
    p.add_argument("--config", type=Path, required=True, help="Path to YAML or JSON config")
    args = p.parse_args()
    cfg = load_config(args.config)

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
    loss_fn = nn.BCEWithLogitsLoss() if y.max() <= 1.0 else nn.MSELoss()

    epochs = int(cfg.get("epochs", 5))
    for epoch in range(epochs):
        total = 0.0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            if isinstance(loss_fn, nn.BCEWithLogitsLoss):
                loss = loss_fn(logits, yb)
            else:
                loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            total += float(loss.detach().cpu())
        print(f"epoch {epoch + 1}/{epochs} loss={total / max(len(loader), 1):.4f}")

    out_dir = Path(cfg.get("output_dir", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "nam_compas_last.pt"
    torch.save(
        {"model": model.state_dict(), "feature_names": feat_names, "config": cfg},
        ckpt,
    )
    print(f"saved {ckpt}")


if __name__ == "__main__":
    main()
