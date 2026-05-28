from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_score))


def compute_auc_pr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_score))
