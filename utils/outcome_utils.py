from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

OutcomeState = Tuple[Any, ...]

def _coerce_state_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        try:
            numeric = float(stripped)
            if numeric.is_integer():
                return int(numeric)
            return numeric
        except Exception:
            return stripped

    if isinstance(value, (np.integer, int)):
        return int(value)

    if isinstance(value, (np.floating, float)):
        numeric = float(value)
        if np.isnan(numeric):
            return np.nan
        if numeric.is_integer():
            return int(numeric)
        return numeric

    return value

def normalize_distribution(dist: Dict[Any, float]) -> Dict[Any, float]:
    normalized = {key: float(value) for key, value in dist.items()}
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("分布之和必须>0")
    return {key: value / total for key, value in normalized.items()}

def format_outcome_state(outcome_columns: Sequence[str], state: OutcomeState) -> str:
    return " | ".join(
        f"{column}={state[idx]}"
        for idx, column in enumerate(outcome_columns)
    )

def format_outcome_distribution(
    outcome_columns: Sequence[str],
    distribution: Dict[OutcomeState, float],
) -> List[str]:
    lines: List[str] = []
    for state, ratio in sorted(distribution.items(), key=lambda item: item[0]):
        lines.append(f"{format_outcome_state(outcome_columns, state)} -> {ratio * 100:.2f}%")
    return lines

def parse_outcome_state_key(raw_key: Any, outcome_columns: Sequence[str]) -> OutcomeState:
    n_outcomes = len(outcome_columns)
    if n_outcomes <= 0:
        raise ValueError("outcome_columns 不能为空")

    if isinstance(raw_key, (list, tuple)):
        values = list(raw_key)
    elif isinstance(raw_key, str):
        if n_outcomes == 1 and "|" not in raw_key:
            values = [raw_key]
        else:
            values = [part.strip() for part in raw_key.split("|")]
    else:
        if n_outcomes != 1:
            raise ValueError(
                "多结局 outcome.target 的键必须是形如 '0|1' 的字符串，"
                "或与 outcome.columns 等长的列表/元组"
            )
        values = [raw_key]

    if len(values) != n_outcomes:
        raise ValueError(
            "outcome.target 的状态键长度与 outcome.columns 不一致: "
            f"期望 {n_outcomes}，实际 {len(values)}，键={raw_key!r}"
        )

    return tuple(_coerce_state_value(value) for value in values)

def coerce_target_distribution(
    target: Any,
    outcome_columns: Sequence[str],
) -> Dict[OutcomeState, float]:
    if not isinstance(target, dict):
        raise ValueError("outcome.target 为自定义分布时，必须是 {状态: 比例} 的映射")

    result: Dict[OutcomeState, float] = {}
    for raw_key, ratio in target.items():
        state = parse_outcome_state_key(raw_key, outcome_columns)
        result[state] = float(ratio)

    return normalize_distribution(result)

def outcome_state_series(
    df: pd.DataFrame,
    outcome_columns: Sequence[str],
) -> pd.Series:
    missing_columns = [column for column in outcome_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"结局列不存在: {missing_columns}")

    return df[list(outcome_columns)].apply(
        lambda row: tuple(_coerce_state_value(value) for value in row.tolist()),
        axis=1,
    )

def get_outcome_joint_distribution(
    df: pd.DataFrame,
    outcome_columns: Sequence[str],
) -> Dict[OutcomeState, float]:
    if not outcome_columns:
        raise ValueError("outcome_columns 不能为空")

    state_counts = outcome_state_series(df, outcome_columns).value_counts(normalize=True)
    return normalize_distribution(state_counts.to_dict())

def enforce_joint_outcome_distribution(
    df: pd.DataFrame,
    outcome_columns: Sequence[str],
    target_dist: Dict[OutcomeState, float],
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if not outcome_columns:
        raise ValueError("outcome_columns 不能为空")

    for column in outcome_columns:
        if column not in df.columns:
            raise ValueError(f"结局列不存在: {column}")

    normalized_target = normalize_distribution(target_dist)
    values = list(normalized_target.keys())
    ideal = {state: normalized_target[state] * n_samples for state in values}
    base = {state: int(np.floor(ideal[state])) for state in values}
    remaining = n_samples - sum(base.values())

    remainders = sorted(
        ((state, ideal[state] - base[state]) for state in values),
        key=lambda item: item[1],
        reverse=True,
    )
    for idx in range(remaining):
        base[remainders[idx % len(remainders)][0]] += 1

    state_series = outcome_state_series(df, outcome_columns)
    parts = []
    for state in values:
        n_needed = base[state]
        if n_needed <= 0:
            continue

        subset = df.loc[state_series[state_series == state].index]
        if subset.empty:
            continue

        replace = len(subset) < n_needed
        sampled_idx = rng.choice(subset.index.to_numpy(), size=n_needed, replace=replace)
        parts.append(df.loc[sampled_idx])

    if not parts:
        return df.sample(
            n=n_samples,
            replace=(len(df) < n_samples),
            random_state=42,
        ).reset_index(drop=True)

    out = pd.concat(parts, axis=0)
    if len(out) < n_samples:
        extra = df.sample(n=n_samples - len(out), replace=True, random_state=42)
        out = pd.concat([out, extra], axis=0)

    out = out.sample(frac=1, random_state=42).reset_index(drop=True)
    return out
