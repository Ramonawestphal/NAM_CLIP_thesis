from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

CONTINUOUS_COLS = ["age", "charge_degree", "length_of_stay", "priors_count"]
CATEGORICAL_COLS = ["race", "sex"]

# Output layout (12 columns) — categorical-first, matching the reference ColumnTransformer:
#   [0-5]  race OHE   — 6 indicators; MinMaxScaler maps {0,1} → {-1, +1}
#   [6-7]  sex OHE    — 2 indicators; MinMaxScaler maps {0,1} → {-1, +1}
#   [8-11] continuous — age, charge_degree, length_of_stay, priors_count; scaled to [-1, +1]
#
# A single MinMaxScaler(feature_range=(-1, 1)) is fit on the full 12-column
# concatenated matrix (matching the reference transform_data behaviour).


class CompasEncoder:
    """Encode raw COMPAS feature frames to a (N, 12) float32 array.

    Fit on training fold only; call transform() on val/test folds.
    Column order: race OHE (0-5), sex OHE (6-7), continuous (8-11).
    A single MinMaxScaler(feature_range=(-1, 1)) covers all 12 columns.
    """

    def __init__(self) -> None:
        self._ohe = OneHotEncoder(
            categories="auto",
            sparse_output=False,
            handle_unknown="ignore",
        )
        self._scaler = MinMaxScaler(feature_range=(-1, 1))
        self.feature_names_: list[str] = []
        self.categorical_groups_: dict = {}
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "CompasEncoder":
        cat = self._ohe.fit_transform(df[CATEGORICAL_COLS])  # fits OHE, returns (N, 8)
        cont = df[CONTINUOUS_COLS].values.astype(np.float64)  # (N, 4)
        full = np.concatenate([cat, cont], axis=1)            # (N, 12) — cat first
        self._scaler.fit(full)

        race_names = [f"race_{c}" for c in self._ohe.categories_[0]]
        sex_names = [f"sex_{c}" for c in self._ohe.categories_[1]]
        self.feature_names_ = race_names + sex_names + list(CONTINUOUS_COLS)

        n_race = len(race_names)
        n_sex = len(sex_names)
        self.categorical_groups_ = {
            "race": {"indices": list(range(n_race)), "names": race_names},
            "sex": {"indices": list(range(n_race, n_race + n_sex)), "names": sex_names},
        }

        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform()")
        cat = self._ohe.transform(df[CATEGORICAL_COLS])
        cont = df[CONTINUOUS_COLS].values.astype(np.float64)
        full = np.concatenate([cat, cont], axis=1)
        return self._scaler.transform(full).astype(np.float32)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        return self.fit(df).transform(df)

    @property
    def n_features(self) -> int:
        return len(self.feature_names_)
