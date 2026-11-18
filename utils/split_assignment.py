from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from utils.outcome_utils import get_outcome_joint_distribution, outcome_state_series

SPLIT_ASSIGNMENT_ROW_COLUMN = "row_index"
SPLIT_ASSIGNMENT_LABEL_COLUMN = "split"
TRAIN_SPLIT_LABEL = "train"
TEST_SPLIT_LABEL = "test"
_INTERNAL_ROW_COLUMN = "__split_assignment_row_index__"

def split_assignment_metadata_path(split_assignment_path: str | os.PathLike[str]) -> Path:
    path = Path(split_assignment_path)
    return path.with_name(f"{path.stem}_metadata.json")

def _attach_row_index(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    result[_INTERNAL_ROW_COLUMN] = np.arange(len(result), dtype=int)
    return result

def _build_split_metadata(
    *,
    data_with_index: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_size: float,
    outcome_columns: Optional[List[str]] = None,
    stratified_by_outcomes: bool = False,
    split_fallback_reason: Optional[str] = None,
    split_source: str,
) -> Dict[str, Any]:
    outcome_columns = outcome_columns or []
    split_metadata: Dict[str, Any] = {
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "requested_test_ratio": float(test_size),
        "actual_test_ratio": float(len(test_df) / max(len(data_with_index), 1)),
        "stratified_by_outcome_joint_state": bool(stratified_by_outcomes),
        "split_source": split_source,
    }
    if split_fallback_reason is not None:
        split_metadata["split_fallback_reason"] = split_fallback_reason
    if outcome_columns and all(col in data_with_index.columns for col in outcome_columns):
        base_df = data_with_index.drop(columns=[_INTERNAL_ROW_COLUMN], errors="ignore")
        train_base_df = train_df.drop(columns=[_INTERNAL_ROW_COLUMN], errors="ignore")
        test_base_df = test_df.drop(columns=[_INTERNAL_ROW_COLUMN], errors="ignore")
        split_metadata["full_outcome_distribution"] = {
            str(k): float(v)
            for k, v in get_outcome_joint_distribution(base_df, outcome_columns).items()
        }
        split_metadata["train_outcome_distribution"] = {
            str(k): float(v)
            for k, v in get_outcome_joint_distribution(train_base_df, outcome_columns).items()
        }
        split_metadata["test_outcome_distribution"] = {
            str(k): float(v)
            for k, v in get_outcome_joint_distribution(test_base_df, outcome_columns).items()
        }
    return split_metadata

def build_split_assignment_df(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    train_rows = pd.DataFrame({
        SPLIT_ASSIGNMENT_ROW_COLUMN: train_df[_INTERNAL_ROW_COLUMN].astype(int),
        SPLIT_ASSIGNMENT_LABEL_COLUMN: TRAIN_SPLIT_LABEL,
    })
    test_rows = pd.DataFrame({
        SPLIT_ASSIGNMENT_ROW_COLUMN: test_df[_INTERNAL_ROW_COLUMN].astype(int),
        SPLIT_ASSIGNMENT_LABEL_COLUMN: TEST_SPLIT_LABEL,
    })
    assignment_df = pd.concat([train_rows, test_rows], axis=0, ignore_index=True)
    assignment_df = assignment_df.sort_values(SPLIT_ASSIGNMENT_ROW_COLUMN).reset_index(drop=True)
    return assignment_df

def generate_split_assignment(
    data: pd.DataFrame,
    *,
    test_size: float,
    random_state: int,
    outcome_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    data_with_index = _attach_row_index(data)
    outcome_columns = outcome_columns or []
    stratify_labels = None
    stratified_by_outcomes = False
    split_fallback_reason: Optional[str] = None

    if outcome_columns and all(col in data_with_index.columns for col in outcome_columns):
        candidate_labels = outcome_state_series(data_with_index, outcome_columns).astype(str)
        label_counts = candidate_labels.value_counts()
        if not label_counts.empty and int(label_counts.min()) >= 2:
            stratify_labels = candidate_labels

    try:
        train_df, test_df = train_test_split(
            data_with_index,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify_labels,
        )
        stratified_by_outcomes = stratify_labels is not None
    except ValueError as exc:
        split_fallback_reason = f"分层随机切分失败，已回退为普通随机切分: {exc}"
        train_df, test_df = train_test_split(
            data_with_index,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )

    split_metadata = _build_split_metadata(
        data_with_index=data_with_index,
        train_df=train_df,
        test_df=test_df,
        test_size=test_size,
        outcome_columns=outcome_columns,
        stratified_by_outcomes=stratified_by_outcomes,
        split_fallback_reason=split_fallback_reason,
        split_source="random_split",
    )
    assignment_df = build_split_assignment_df(train_df, test_df)
    train_out = train_df.drop(columns=[_INTERNAL_ROW_COLUMN]).reset_index(drop=True)
    test_out = test_df.drop(columns=[_INTERNAL_ROW_COLUMN]).reset_index(drop=True)
    return train_out, test_out, split_metadata, assignment_df

def save_split_assignment(
    split_assignment_path: str | os.PathLike[str],
    assignment_df: pd.DataFrame,
    split_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(split_assignment_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    assignment_df.to_csv(path, encoding="utf-8-sig", index=False)
    if split_metadata is not None:
        metadata_path = split_assignment_metadata_path(path)
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(split_metadata, f, ensure_ascii=False, indent=2)

def load_split_assignment(
    split_assignment_path: str | os.PathLike[str],
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    path = Path(split_assignment_path)
    assignment_df = pd.read_csv(path, encoding="utf-8-sig")
    metadata_path = split_assignment_metadata_path(path)
    metadata: Optional[Dict[str, Any]] = None
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    return assignment_df, metadata

def apply_split_assignment(
    data: pd.DataFrame,
    assignment_df: pd.DataFrame,
    *,
    outcome_columns: Optional[List[str]] = None,
    split_metadata_override: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    required_columns = {SPLIT_ASSIGNMENT_ROW_COLUMN, SPLIT_ASSIGNMENT_LABEL_COLUMN}
    missing_columns = required_columns - set(assignment_df.columns)
    if missing_columns:
        raise ValueError(f"split_assignment 缺少必要列: {sorted(missing_columns)}")

    assignment_df = assignment_df.copy()
    assignment_df[SPLIT_ASSIGNMENT_ROW_COLUMN] = assignment_df[SPLIT_ASSIGNMENT_ROW_COLUMN].astype(int)
    if assignment_df[SPLIT_ASSIGNMENT_ROW_COLUMN].duplicated().any():
        raise ValueError("split_assignment 中存在重复的 row_index")

    valid_labels = {TRAIN_SPLIT_LABEL, TEST_SPLIT_LABEL}
    actual_labels = set(assignment_df[SPLIT_ASSIGNMENT_LABEL_COLUMN].astype(str))
    invalid_labels = actual_labels - valid_labels
    if invalid_labels:
        raise ValueError(f"split_assignment 中存在非法 split 标签: {sorted(invalid_labels)}")

    expected_indices = set(range(len(data)))
    actual_indices = set(assignment_df[SPLIT_ASSIGNMENT_ROW_COLUMN].tolist())
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)[:10]
        extra = sorted(actual_indices - expected_indices)[:10]
        raise ValueError(
            "split_assignment 与当前数据行数不匹配。"
            f" 缺失 row_index 示例: {missing}; 额外 row_index 示例: {extra}"
        )

    data_with_index = _attach_row_index(data)
    assignment_by_row = assignment_df.set_index(SPLIT_ASSIGNMENT_ROW_COLUMN)[SPLIT_ASSIGNMENT_LABEL_COLUMN].astype(str)
    row_labels = data_with_index[_INTERNAL_ROW_COLUMN].map(assignment_by_row)
    train_df = data_with_index.loc[row_labels == TRAIN_SPLIT_LABEL].copy()
    test_df = data_with_index.loc[row_labels == TEST_SPLIT_LABEL].copy()

    split_metadata = _build_split_metadata(
        data_with_index=data_with_index,
        train_df=train_df,
        test_df=test_df,
        test_size=float(len(test_df) / max(len(data_with_index), 1)),
        outcome_columns=outcome_columns,
        stratified_by_outcomes=bool(
            split_metadata_override.get("stratified_by_outcome_joint_state", False)
            if split_metadata_override else False
        ),
        split_fallback_reason=(
            split_metadata_override.get("split_fallback_reason")
            if split_metadata_override else None
        ),
        split_source="assignment_file",
    )
    if split_metadata_override:
        for key, value in split_metadata_override.items():
            if key not in split_metadata:
                split_metadata[key] = value

    train_out = train_df.drop(columns=[_INTERNAL_ROW_COLUMN]).reset_index(drop=True)
    test_out = test_df.drop(columns=[_INTERNAL_ROW_COLUMN]).reset_index(drop=True)
    return train_out, test_out, split_metadata
