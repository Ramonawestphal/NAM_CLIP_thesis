from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

CONTINUOUS_COLS = ["age", "charge_degree", "length_of_stay", "priors_count"]
CATEGORICAL_COLS = ["race", "sex"]

# Output layout (12 columns) — continuous-first, OHE last:
#   [0-3]  continuous — part of joint MinMaxScaler fit
#   [4-9]  race OHE   — 6 indicators; MinMaxScaler maps {0,1} → {-1, +1}
#   [10-11] sex OHE   — 2 indicators; MinMaxScaler maps {0,1} → {-1, +1}
#
# A single MinMaxScaler(feature_range=(-1, 1)) is fit on the full 12-column
# concatenated matrix (matching the reference transform_data behaviour).


class CompasEncoder:
    """Encode raw COMPAS feature frames to a (N, 12) float32 array.

    Fit on training fold only; call transform() on val/test folds.
    A single MinMaxScaler(feature_range=(-1, 1)) is applied to the full
    12-column matrix so that one-hot columns are mapped to {-1, +1}.
    """

    def __init__(self) -> None:
        self._ohe = OneHotEncoder(
            categories="auto",
            sparse_output=False,
            handle_unknown="ignore",
        )
        self._scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "CompasEncoder":
        cont = df[CONTINUOUS_COLS].values.astype(np.float64)
        cat = self._ohe.fit_transform(df[CATEGORICAL_COLS])  # fits OHE and returns matrix
        full = np.concatenate([cont, cat], axis=1)           # (N, 12)
        self._scaler.fit(full)
        race_names = [f"race_{c}" for c in self._ohe.categories_[0]]
        sex_names = [f"sex_{c}" for c in self._ohe.categories_[1]]
        self.feature_names_ = list(CONTINUOUS_COLS) + race_names + sex_names
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        cont = df[CONTINUOUS_COLS].values.astype(np.float64)
        cat = self._ohe.transform(df[CATEGORICAL_COLS])
        full = np.concatenate([cont, cat], axis=1)
        return self._scaler.transform(full).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)
