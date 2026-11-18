from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .tasks import MachineLearningTask
from .utils import normalize_categorical_value

def apply_task_subset(df: pd.DataFrame, task: MachineLearningTask) -> pd.DataFrame:
    if task.subset_column is None:
        return df.copy()
    if task.subset_column not in df.columns:
        return df.iloc[0:0].copy()

    target_token = normalize_categorical_value(task.subset_value)
    subset_tokens = df[task.subset_column].apply(normalize_categorical_value)
    return df.loc[subset_tokens == target_token].copy()

def to_binary_target(series: pd.Series) -> pd.Series:
    def _convert(value: Any) -> float:
        token = normalize_categorical_value(value)
        if token == "1":
            return 1.0
        if token == "0":
            return 0.0
        return np.nan

    return series.apply(_convert).astype(float)

def prepare_task_dataset(
    df: pd.DataFrame,
    task: MachineLearningTask,
    feature_columns: List[str],
    source_name: str,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]:
    subset_df = apply_task_subset(df, task)
    if task.target_column not in subset_df.columns:
        return (
            subset_df.iloc[0:0].copy(),
            pd.Series(dtype=float),
            {
                "source": source_name,
                "rows_before_target_filter": int(len(subset_df)),
                "rows_after_target_filter": 0,
                "positive_rows": 0,
                "negative_rows": 0,
                "has_target_column": False,
            },
        )

    y = to_binary_target(subset_df[task.target_column])
    valid_mask = y.isin([0.0, 1.0])
    filtered_df = subset_df.loc[valid_mask, feature_columns].copy()
    filtered_y = y.loc[valid_mask].astype(int).reset_index(drop=True)
    filtered_df = filtered_df.reset_index(drop=True)
    positive_rows = int((filtered_y == 1).sum())
    negative_rows = int((filtered_y == 0).sum())
    metadata = {
        "source": source_name,
        "rows_before_target_filter": int(len(subset_df)),
        "rows_after_target_filter": int(len(filtered_df)),
        "positive_rows": positive_rows,
        "negative_rows": negative_rows,
        "has_target_column": True,
    }
    return filtered_df, filtered_y, metadata

def is_binary_dataset_usable(
    X: pd.DataFrame,
    y: pd.Series,
    minimum_rows: int,
    minimum_positive_rows: int,
) -> Tuple[bool, str]:
    if len(X) < minimum_rows:
        return False, f"样本数不足（{len(X)} < {minimum_rows}）"
    positive_rows = int((y == 1).sum())
    negative_rows = int((y == 0).sum())
    if positive_rows < minimum_positive_rows:
        return False, f"阳性样本不足（{positive_rows} < {minimum_positive_rows}）"
    if negative_rows < minimum_positive_rows:
        return False, f"阴性样本不足（{negative_rows} < {minimum_positive_rows}）"
    return True, "ok"
