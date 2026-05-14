"""Unit tests for NAM implementation. No real COMPAS data, no training loops.
All hyperparameters loaded from configs/compas_replication.yaml.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.data.encoding import CompasEncoder
from src.nam.feature_nn import FeatureNN
from src.nam.losses import output_penalty, total_loss
from src.nam.nam import NAM
from src.utils.seeding import seed_everything

# ---------------------------------------------------------------------------
# Config fixture — single load, shared across all tests
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "compas_replication.yaml"


@pytest.fixture(scope="module")
def cfg() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Tiny synthetic COMPAS-shaped DataFrame for encoder tests
# ---------------------------------------------------------------------------

RACES = ["African-American", "Asian", "Caucasian", "Hispanic", "Native American", "Other"]
SEXES = ["Female", "Male"]
N_ENCODER_ROWS = 60  # enough to cover all categories


@pytest.fixture(scope="module")
def compas_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = N_ENCODER_ROWS
    return pd.DataFrame(
        {
            "age": rng.integers(18, 70, size=n),
            "charge_degree": rng.choice([1, 2], size=n),
            "length_of_stay": rng.integers(0, 365, size=n),
            "priors_count": rng.integers(0, 30, size=n),
            "race": rng.choice(RACES, size=n),
            "sex": rng.choice(SEXES, size=n),
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — FeatureNN output shape
# ---------------------------------------------------------------------------

def test_feature_nn_shape(cfg):
    """FeatureNN maps (B, 1) → (B,) for the configured dropout."""
    B = 32
    dropout = cfg["dropout"]
    net = FeatureNN(dropout=dropout)
    net.eval()
    x = torch.randn(B, 1)
    out = net(x)
    assert out.shape == (B,), f"Expected ({B},), got {out.shape}"


# ---------------------------------------------------------------------------
# Test 2 — FeatureNN layer count and output bias
# ---------------------------------------------------------------------------

def test_feature_nn_layer_count(cfg):
    """FeatureNN has exactly 4 Linear layers; the last one has no bias."""
    net = FeatureNN(dropout=cfg["dropout"])
    linears = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 4, f"Expected 4 Linear layers, found {len(linears)}"
    assert linears[-1].bias is None, "Output Linear must have bias=False"


# ---------------------------------------------------------------------------
# Test 3 — NAM output shape and logit type
# ---------------------------------------------------------------------------

def test_nam_output_shape(cfg):
    """NAM with 12 sub-networks produces logits of shape (B,) for any batch size."""
    B = 16
    K = 12  # 6 race OHE + 2 sex OHE + 4 continuous
    model = NAM(
        n_features=K,
        dropout=cfg["dropout"],
        feature_dropout=cfg["feature_dropout"],
    )
    model.eval()
    x = torch.randn(B, K)
    logits = model(x)
    assert logits.shape == (B,), f"Expected ({B},), got {logits.shape}"
    assert logits.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test 4 — output_penalty is deterministic (dropout OFF during penalty)
# ---------------------------------------------------------------------------

def test_output_penalty_no_dropout(cfg):
    """output_penalty must return identical values on repeated calls (dropout disabled)."""
    K = 12
    # High dropout to amplify any stochasticity if dropout were accidentally active
    model = NAM(n_features=K, dropout=0.9, feature_dropout=0.9)
    model.train()  # keep model in train mode to confirm penalty ignores it
    x = torch.randn(20, K)
    penalties = [output_penalty(model, x).item() for _ in range(10)]
    assert all(
        abs(p - penalties[0]) < 1e-6 for p in penalties
    ), "output_penalty should be deterministic (eval mode inside)"
    # Model should be back in train mode after the call
    assert model.training, "output_penalty must restore training state"


# ---------------------------------------------------------------------------
# Test 5 — total_loss increases when output_reg > 0
# ---------------------------------------------------------------------------

def test_loss_increases_with_penalty(cfg):
    """total_loss(output_reg>0) > BCE-only loss."""
    K = 12
    B = 32
    model = NAM(n_features=K, dropout=cfg["dropout"], feature_dropout=cfg["feature_dropout"])
    model.eval()
    x = torch.randn(B, K)
    y = torch.randint(0, 2, (B,)).float()

    bce_only = total_loss(model, x, y, output_reg=0.0)
    penalized = total_loss(model, x, y, output_reg=cfg["output_reg"])
    assert penalized.item() > bce_only.item(), (
        f"penalized ({penalized.item():.4f}) should exceed bce ({bce_only.item():.4f})"
    )


# ---------------------------------------------------------------------------
# Test 6 — CompasEncoder produces exactly 12 columns
# ---------------------------------------------------------------------------

def test_encoder_column_count(cfg, compas_df):
    """CompasEncoder.fit_transform yields (N, 12) output."""
    enc = CompasEncoder()
    X = enc.fit_transform(compas_df)
    assert X.shape[1] == 12, f"Expected 12 columns, got {X.shape[1]}"
    assert enc.n_features == 12
    # Column order: 4 continuous + 6 race OHE + 2 sex OHE
    assert enc.feature_names_[:4] == ["age", "charge_degree", "length_of_stay", "priors_count"]
    assert enc.feature_names_[4:10] == [f"race_{r}" for r in sorted(RACES)]
    assert enc.feature_names_[10:] == [f"sex_{s}" for s in sorted(SEXES)]


# ---------------------------------------------------------------------------
# Test 7 — CompasEncoder values in [-1, 1]
# ---------------------------------------------------------------------------

def test_encoder_range(cfg, compas_df):
    """All encoded values must lie in [-1, 1] (MinMaxScaler feature_range=(-1,1))."""
    enc = CompasEncoder()
    X = enc.fit_transform(compas_df)
    assert float(X.min()) >= -1.0 - 1e-6, f"Min {X.min()} below -1"
    assert float(X.max()) <= 1.0 + 1e-6, f"Max {X.max()} above 1"


# ---------------------------------------------------------------------------
# Test 8 — seed_everything reproducibility
# ---------------------------------------------------------------------------

def test_seeding_reproducibility(cfg):
    """Same seed → identical NAM outputs; different seed → different outputs."""
    K = 12
    x = torch.randn(8, K)

    seed_everything(cfg["seed"])
    m1 = NAM(n_features=K, dropout=cfg["dropout"], feature_dropout=cfg["feature_dropout"])
    m1.eval()
    with torch.no_grad():
        out1 = m1(x).clone()

    seed_everything(cfg["seed"])
    m2 = NAM(n_features=K, dropout=cfg["dropout"], feature_dropout=cfg["feature_dropout"])
    m2.eval()
    with torch.no_grad():
        out2 = m2(x).clone()

    assert torch.allclose(out1, out2), "Same seed must produce identical outputs"

    seed_everything(cfg["seed"] + 1)
    m3 = NAM(n_features=K, dropout=cfg["dropout"], feature_dropout=cfg["feature_dropout"])
    m3.eval()
    with torch.no_grad():
        out3 = m3(x).clone()

    assert not torch.allclose(out1, out3), "Different seeds should produce different weights"


# ---------------------------------------------------------------------------
# Test 9 — Encoder column layout: continuous first, OHE last
# ---------------------------------------------------------------------------

def test_encoder_column_layout(cfg, compas_df):
    """Indices 0-3 are continuous-valued; indices 4-11 are binary OHE {0, 1}."""
    enc = CompasEncoder()
    X = enc.fit_transform(compas_df)  # (N, 12)

    # Continuous columns (0-3): after MinMaxScaler(-1, 1) values land in [-1, 1] but
    # are NOT restricted to {0, 1} (OHE's only two values after scaling).
    # charge_degree has just 2 levels → maps to exactly {-1.0, 1.0}, which is fine:
    # neither -1 nor 1 is in {0.0, 1.0}.
    for i in range(4):
        unique_vals = set(np.unique(X[:, i]).tolist())
        assert not unique_vals.issubset({0.0, 1.0}), (
            f"Column {i} values {unique_vals} look like OHE indicators; "
            "continuous column must contain values outside {{0.0, 1.0}}"
        )

    # OHE columns (4-11): all values must be exactly 0 or 1
    ohe_block = X[:, 4:]
    assert set(np.unique(ohe_block).tolist()).issubset({0.0, 1.0}), (
        "OHE columns (indices 4-11) must contain only 0.0 and 1.0"
    )

    # Each sample row must have exactly one active race indicator (indices 4-9)
    # and exactly one active sex indicator (indices 10-11)
    race_sums = X[:, 4:10].sum(axis=1)
    sex_sums = X[:, 10:12].sum(axis=1)
    assert np.all(race_sums == 1.0), "Each row must have exactly one race OHE active"
    assert np.all(sex_sums == 1.0), "Each row must have exactly one sex OHE active"


# ---------------------------------------------------------------------------
# Test 10 — calc_outputs does not apply feature dropout or add bias
# ---------------------------------------------------------------------------

def test_calc_outputs_no_feature_dropout_no_bias(cfg):
    """calc_outputs must not apply feature dropout or add the global bias."""
    K = 12
    B = 32
    seed_everything(42)
    model = NAM(n_features=K, dropout=cfg["dropout"], feature_dropout=0.5)
    model.bias.data.fill_(99.0)

    x = torch.randn(B, K)

    # --- Part A: train mode — feature dropout must NOT zero entire feature columns ---
    # With feature_dropout=0.5, forward() would zero ~50% of the (B, K) elements.
    # calc_outputs bypasses feature dropout entirely, so no column should be all-zero.
    model.train()
    with torch.no_grad():
        outputs_train = model.calc_outputs(x)  # list of K tensors (B,)
    stacked_train = torch.stack(outputs_train, dim=1)  # (B, K)

    col_all_zero = (stacked_train == 0.0).all(dim=0)  # True where column is all-zero
    assert not col_all_zero.any(), (
        "calc_outputs should not apply feature dropout — no feature column should be entirely zero"
    )

    # --- Part B: eval mode — logits = sum(calc_outputs) + bias, exactly ---
    # In eval mode all dropout is off, so the arithmetic must hold exactly.
    model.eval()
    with torch.no_grad():
        outputs_eval = model.calc_outputs(x)
        stacked_eval = torch.stack(outputs_eval, dim=1)  # (B, K)
        logits = model(x)                                # (B,)

    # forward() = stacked.sum(dim=1) + bias  =>  logits - sum(calc_outputs) = 99 everywhere
    diff = logits - stacked_eval.sum(dim=1)        # should be +99 at every position
    err = (diff - 99.0).abs().max().item()
    assert err < 1e-4, (
        f"Expected logits - sum(calc_outputs) = 99 (the bias); max deviation = {err:.6f}"
    )

    # Sanity: calc_outputs alone must NOT include the bias
    assert stacked_eval.mean().item() < 50.0, (
        "calc_outputs mean is unexpectedly large — bias may have been added incorrectly"
    )
