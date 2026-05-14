from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

CONTINUOUS_COLS = ["age", "charge_degree", "length_of_stay", "priors_count"]
CATEGORICAL_COLS = ["race", "sex"]

# Output layout (12 columns):
#   [0-3]  continuous — MinMaxScaled to [-1, 1]
#   [4-9]  race OHE   — 6 binary indicators {0, 1}
#   [10-11] sex OHE   — 2 binary indicators {0, 1}


class CompasEncoder:
    """Encode raw COMPAS feature frames to a (N, 12) float32 array.

    Fit on training fold only; call transform() on val/test folds.
    MinMaxScaler is applied only to the 4 continuous columns so that
    OHE indicators remain as exact {0, 1} values.
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
        self._scaler.fit(cont)
        self._ohe.fit(df[CATEGORICAL_COLS])
        race_names = [f"race_{c}" for c in self._ohe.categories_[0]]
        sex_names = [f"sex_{c}" for c in self._ohe.categories_[1]]
        self.feature_names_ = list(CONTINUOUS_COLS) + race_names + sex_names
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        cont_scaled = self._scaler.transform(df[CONTINUOUS_COLS].values.astype(np.float64))
        cat = self._ohe.transform(df[CATEGORICAL_COLS])
        return np.concatenate([cont_scaled, cat], axis=1).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)
