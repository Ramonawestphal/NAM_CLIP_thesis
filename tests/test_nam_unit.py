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
# Test 4 — output_penalty is differentiable (gradients reach FeatureNN weights)
# ---------------------------------------------------------------------------

def test_output_penalty_differentiable(cfg):
    """output_penalty must be differentiable: gradients flow to FeatureNN weights."""
    K = 12
    model = NAM(n_features=K, dropout=0.0, feature_dropout=0.0)
    model.train()
    x = torch.randn(8, K)

    penalty = output_penalty(model, x)
    penalty.backward()

    grads = [p.grad for fnn in model.feature_nns for p in fnn.parameters()]
    assert all(g is not None for g in grads), "All FeatureNN params must receive gradients"
    assert any(g.abs().max().item() > 0 for g in grads), "At least some gradients must be non-zero"


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
    """Indices 0-3 are continuous-valued; indices 4-11 are OHE scaled to {-1, +1}."""
    enc = CompasEncoder()
    X = enc.fit_transform(compas_df)  # (N, 12)

    # Continuous columns (0-3): MinMaxScaler maps to [-1, 1] but values are not
    # restricted to just {-1, +1} (which is what OHE columns produce).
    # charge_degree has 2 levels → maps to exactly {-1.0, 1.0}; that is fine because
    # neither value is in {0.0, 1.0} so we can still distinguish it from raw binary.
    for i in range(4):
        unique_vals = set(np.unique(X[:, i]).tolist())
        assert not unique_vals.issubset({0.0, 1.0}), (
            f"Column {i} values {unique_vals} look like raw OHE (pre-scaling); "
            "after joint MinMaxScaler, no column should contain only {{0.0, 1.0}}"
        )

    # OHE columns (4-11): joint MinMaxScaler maps binary {0,1} to exactly {-1.0, +1.0}
    ohe_block = X[:, 4:]
    assert set(np.unique(ohe_block).tolist()).issubset({-1.0, 1.0}), (
        "OHE columns (indices 4-11) must contain only -1.0 and +1.0 after joint scaling"
    )

    # Each row has exactly one active race indicator (cols 4-9) and one sex (cols 10-11).
    # With {-1, +1} encoding: active=+1, inactive=-1.
    # Race sum: 1*(+1) + 5*(-1) = -4; sex sum: 1*(+1) + 1*(-1) = 0.
    race_sums = X[:, 4:10].sum(axis=1)
    sex_sums = X[:, 10:12].sum(axis=1)
    assert np.allclose(race_sums, -4.0), (
        f"Each row must have exactly one race OHE active (expected sum -4, got {race_sums[:5]})"
    )
    assert np.allclose(sex_sums, 0.0), (
        f"Each row must have exactly one sex OHE active (expected sum 0, got {sex_sums[:5]})"
    )


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


# ---------------------------------------------------------------------------
# Test 11 — output_penalty matches paper eq. (3) hand computation
# ---------------------------------------------------------------------------

def test_output_penalty_matches_paper_formula(cfg):
    """output_penalty == (1/K) Σ_k mean_n[f_k²] computed by hand on a tiny NAM."""
    K = 3
    B = 8
    seed_everything(0)
    model = NAM(n_features=K, dropout=0.0, feature_dropout=0.0)
    model.eval()
    x = torch.randn(B, K)

    with torch.no_grad():
        outputs = model.calc_outputs(x)  # list of K tensors (B,)

    # Hand-compute eq. (3): for each k, mean_n[f_k(x_kn)²], then average over K
    expected = sum(o.pow(2).mean() for o in outputs) / K

    actual = output_penalty(model, x)
    assert abs(actual.item() - expected.item()) < 1e-6, (
        f"output_penalty={actual.item():.8f} ≠ hand-computed={expected.item():.8f}"
    )


# ---------------------------------------------------------------------------
# Test 12 — feature_dropout_p=1.0 zeroes all contributions → logit = bias only
# ---------------------------------------------------------------------------

def test_feature_dropout_zeros_features(cfg):
    """With feature_dropout_p=1.0 in train mode, every feature is dropped → logit == bias."""
    K = 12
    model = NAM(n_features=K, dropout=0.0, feature_dropout=1.0)
    model.bias.data.fill_(3.7)
    model.train()
    x = torch.randn(8, K)
    with torch.no_grad():
        logits = model(x)
    expected = model.bias.expand(8)
    assert torch.allclose(logits, expected), (
        f"With feature_dropout=1.0, logits should equal bias everywhere; "
        f"max deviation={( logits - expected).abs().max().item():.6f}"
    )


# ---------------------------------------------------------------------------
# Test 13 — feature dropout does NOT rescale survivors (no inverted dropout)
# ---------------------------------------------------------------------------

def test_feature_dropout_no_rescaling(cfg):
    """E[f_k contribution] = 0.5 × full (plain Bernoulli), not 1.0 × (inverted dropout).

    Uses K=1 to eliminate sign-cancellation: feature_contribution is either 0 (dropped)
    or f_1(x_1) (kept), so the mean over many trials converges cleanly to 0.5 × f_1(x_1).
    """
    K = 1
    n_trials = 5000
    seed_everything(3)
    # dropout=0.0: isolate feature dropout; no FeatureNN internal stochasticity
    model = NAM(n_features=K, dropout=0.0, feature_dropout=0.5)

    x = torch.randn(1, K)

    # Eval mode: no feature dropout — full feature contribution
    model.eval()
    with torch.no_grad():
        feature_contribution = (model(x) - model.bias).item()

    if abs(feature_contribution) < 1e-4:
        pytest.skip("Feature contribution too small for ratio test with this seed")

    # Train mode: average over many Bernoulli draws
    model.train()
    contributions = []
    with torch.no_grad():
        for _ in range(n_trials):
            contributions.append((model(x) - model.bias).item())

    mean_contribution = float(np.mean(contributions))

    expected_bernoulli = 0.5 * feature_contribution   # plain zeroing
    expected_inverted = 1.0 * feature_contribution     # nn.Dropout inverted scaling

    err_bernoulli = abs(mean_contribution - expected_bernoulli)
    err_inverted = abs(mean_contribution - expected_inverted)

    # With 5000 trials, SE ≈ 0.5*|f|/√5000 ≈ 0.007*|f|; inverted-dropout error ≈ 0.5*|f|
    assert err_bernoulli < err_inverted / 3.0, (
        f"E[contribution]={mean_contribution:.4f}; "
        f"plain Bernoulli expects {expected_bernoulli:.4f} (err={err_bernoulli:.5f}), "
        f"inverted dropout expects {expected_inverted:.4f} (err={err_inverted:.5f})"
    )


# ---------------------------------------------------------------------------
# Test 14 — encoder does not refit on val data (no leakage)
# ---------------------------------------------------------------------------

def test_encoder_no_leakage(cfg, compas_df):
    """Encoder fitted on train slice must not refit when transform() is called on val."""
    train_df = compas_df.iloc[:40].reset_index(drop=True)
    val_df = compas_df.iloc[40:].reset_index(drop=True)

    enc = CompasEncoder()
    enc.fit(train_df)

    # Record train-fold scaler bounds before any val data is seen
    train_data_min = enc._scaler.data_min_.copy()
    train_data_max = enc._scaler.data_max_.copy()

    # Inject extreme values in val (far outside the train min/max)
    val_ood = val_df.copy()
    val_ood["age"] = 999
    val_ood["priors_count"] = -999

    X_val = enc.transform(val_ood)  # must not refit

    # Scaler state must be unchanged — no leakage from val data
    assert np.array_equal(enc._scaler.data_min_, train_data_min), (
        "Scaler data_min_ changed after transform() — encoder refitted on val data"
    )
    assert np.array_equal(enc._scaler.data_max_, train_data_max), (
        "Scaler data_max_ changed after transform() — encoder refitted on val data"
    )
    assert X_val.shape == (len(val_ood), 12)


# ---------------------------------------------------------------------------
# Test 15 — all 12 columns lie in [-1, +1] after joint scaling (Fix A)
# ---------------------------------------------------------------------------

def test_encoder_all_columns_in_unit_interval(cfg, compas_df):
    """Every column — continuous and OHE alike — must lie in [-1, +1] after joint scaling."""
    enc = CompasEncoder()
    X = enc.fit_transform(compas_df)
    assert float(X.min()) >= -1.0 - 1e-6, f"Min {X.min():.6f} is below -1"
    assert float(X.max()) <= 1.0 + 1e-6, f"Max {X.max():.6f} is above +1"


# ---------------------------------------------------------------------------
# Test 16 — OHE columns (4-11) contain exactly {-1.0, +1.0} (Fix A)
# ---------------------------------------------------------------------------

def test_encoder_onehot_columns_are_pm_one(cfg, compas_df):
    """OHE columns at indices 4-11 must contain only the two values {-1.0, +1.0}."""
    enc = CompasEncoder()
    X = enc.fit_transform(compas_df)
    ohe_block = X[:, 4:]
    unique_vals = set(np.unique(ohe_block).tolist())
    assert unique_vals == {-1.0, 1.0}, (
        f"OHE columns (4-11) should be exactly {{-1.0, +1.0}}, got {unique_vals}"
    )


# ---------------------------------------------------------------------------
# Test 17 — continuous columns (0-3) span both sides of 0 (Fix A)
# ---------------------------------------------------------------------------

def test_encoder_continuous_columns_span_range(cfg, compas_df):
    """Continuous columns at indices 0-3 must each contain at least one value < 0 and > 0."""
    enc = CompasEncoder()
    X = enc.fit_transform(compas_df)
    # age (0), length_of_stay (2), priors_count (3) vary widely → always span both signs.
    # charge_degree (1) has exactly 2 values → maps to {-1.0, +1.0}; skip the strict check.
    for i in [0, 2, 3]:
        col = X[:, i]
        assert col.min() < 0.0, f"Continuous column {i} has no negative value (min={col.min():.4f})"
        assert col.max() > 0.0, f"Continuous column {i} has no positive value (max={col.max():.4f})"
