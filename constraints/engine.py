from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.outcome_utils import enforce_joint_outcome_distribution

from .spec import (
    AllowedValuesConstraint,
    ConstraintSpec,
    FormulaConstraint,
    LogicalRule,
    RangeConstraint,
    RuleCondition,
)

@dataclass
class ConstraintReport:
    clipped: Dict[str, int]
    repaired: Dict[str, int]
    rejected_rows: int
    warnings: List[str]

def _safe_series(df: pd.DataFrame, col: str) -> Optional[pd.Series]:
    if col not in df.columns:
        return None
    return df[col]

def _eval_condition(df: pd.DataFrame, cond: RuleCondition) -> pd.Series:
    s = _safe_series(df, cond.col)
    if s is None:
        return pd.Series([False] * len(df), index=df.index)

    op = cond.op
    v = cond.value

    if op == 'is_null':
        return s.isnull()
    if op == 'not_null':
        return ~s.isnull()

    if op in ('in', 'not_in'):
        if isinstance(v, list):
            vals = v
        else:
            vals = [v]
        mask = s.isin(vals)
        return mask if op == 'in' else ~mask

    if op in ('==', '!='):
        mask = s == v
        return mask if op == '==' else ~mask

    s_num = pd.to_numeric(s, errors='coerce')
    v_num = pd.to_numeric(pd.Series([v]), errors='coerce').iloc[0]
    if pd.isna(v_num):
        return pd.Series([False] * len(df), index=df.index)

    if op == '>':
        return s_num > v_num
    if op == '>=':
        return s_num >= v_num
    if op == '<':
        return s_num < v_num
    if op == '<=':
        return s_num <= v_num

    return pd.Series([False] * len(df), index=df.index)

def _apply_range_clip(df: pd.DataFrame, c: RangeConstraint, report: ConstraintReport) -> None:
    s = _safe_series(df, c.column)
    if s is None:
        report.warnings.append(f"范围约束列不存在，已忽略: {c.column}")
        return

    s_num = pd.to_numeric(s, errors='coerce')
    before = s_num.copy()

    if c.min is not None:
        s_num = s_num.clip(lower=float(c.min))
    if c.max is not None:
        s_num = s_num.clip(upper=float(c.max))

    changed = (before != s_num) & ~(before.isnull() & s_num.isnull())
    changed_count = int(changed.sum())
    if changed_count > 0:
        report.clipped[c.column] = report.clipped.get(c.column, 0) + changed_count
        df[c.column] = s_num

def _apply_allowed_repair_random(
    df: pd.DataFrame,
    c: AllowedValuesConstraint,
    rng: np.random.Generator,
    report: ConstraintReport,
) -> None:
    s = _safe_series(df, c.column)
    if s is None:
        report.warnings.append(f"取值约束列不存在，已忽略: {c.column}")
        return

    allowed = list(c.values)
    mask_bad = ~s.isin(allowed) & ~s.isnull()
    n_bad = int(mask_bad.sum())
    if n_bad <= 0:
        return

    repaired_vals = rng.choice(allowed, size=n_bad, replace=True)
    df.loc[mask_bad, c.column] = repaired_vals
    report.repaired[c.column] = report.repaired.get(c.column, 0) + n_bad

def _apply_logical_rules(df: pd.DataFrame, rules: List[LogicalRule], report: ConstraintReport) -> None:
    for r in rules:
        if not r.if_all or not r.then:
            continue
        mask = pd.Series([True] * len(df), index=df.index)
        for cond in r.if_all:
            mask &= _eval_condition(df, cond)

        n_hit = int(mask.sum())
        if n_hit <= 0:
            continue

        for action in r.then:
            if action.col not in df.columns:
                report.warnings.append(f"规则'{r.name}'的目标列不存在，已忽略: {action.col}")
                continue
            df.loc[mask, action.col] = action.value
            report.repaired[action.col] = report.repaired.get(action.col, 0) + n_hit

def _apply_formula_constraints(
    df: pd.DataFrame,
    formulas: List[FormulaConstraint],
    report: ConstraintReport,
) -> None:
    for formula in formulas:
        if formula.target_col not in df.columns:
            report.warnings.append(f"公式约束'{formula.name}'的目标列不存在，已忽略: {formula.target_col}")
            continue

        missing_cols = [col for col in formula.source_cols if col not in df.columns]
        if missing_cols:
            report.warnings.append(
                f"公式约束'{formula.name}'的源列不存在，已忽略: {', '.join(missing_cols)}"
            )
            continue

        mask = pd.Series([True] * len(df), index=df.index)
        for cond in formula.if_all:
            mask &= _eval_condition(df, cond)

        if not mask.any():
            continue

        numeric_sources: List[pd.Series] = []
        valid_mask = mask.copy()
        for col in formula.source_cols:
            source_num = pd.to_numeric(df[col], errors='coerce')
            numeric_sources.append(source_num)
            valid_mask &= source_num.notnull()

        n_valid = int(valid_mask.sum())
        if n_valid <= 0:
            report.warnings.append(
                f"公式约束'{formula.name}'命中行中没有可用于计算的数值型操作数，已跳过"
            )
            continue

        if formula.op == 'multiply':
            computed = pd.Series(1.0, index=df.index)
            for source_num in numeric_sources:
                computed = computed * source_num
            computed = computed * float(formula.constant)
        else:
            report.warnings.append(f"公式约束'{formula.name}'包含未支持的运算，已忽略: {formula.op}")
            continue

        df.loc[valid_mask, formula.target_col] = computed.loc[valid_mask]
        report.repaired[formula.target_col] = report.repaired.get(formula.target_col, 0) + n_valid

        n_invalid = int((mask & ~valid_mask).sum())
        if n_invalid > 0:
            report.warnings.append(
                f"公式约束'{formula.name}'有{n_invalid}行因源列非数值或缺失未更新"
            )

def _apply_continuous_precision_alignment(
    df: pd.DataFrame,
    spec: ConstraintSpec,
    report: ConstraintReport,
) -> None:
    if not spec.match_continuous_precision:
        return

    for column, decimal_places in spec.continuous_decimal_places.items():
        s = _safe_series(df, column)
        if s is None:
            continue
        s_num = pd.to_numeric(s, errors='coerce')
        if s_num.notnull().sum() <= 0:
            continue
        rounded = s_num.round(int(decimal_places))
        changed = (s_num != rounded) & ~(s_num.isnull() & rounded.isnull())
        changed_count = int(changed.sum())
        if changed_count > 0:
            report.repaired[column] = report.repaired.get(column, 0) + changed_count
            df[column] = rounded

def _hard_valid_mask(df: pd.DataFrame, spec: ConstraintSpec, report: ConstraintReport) -> pd.Series:
    mask = pd.Series([True] * len(df), index=df.index)

    for c in spec.ranges:
        if c.mode != 'reject':
            continue
        s = _safe_series(df, c.column)
        if s is None:
            report.warnings.append(f"范围约束列不存在，已忽略: {c.column}")
            continue
        s_num = pd.to_numeric(s, errors='coerce')
        m = ~s_num.isnull()
        if c.min is not None:
            m &= s_num >= float(c.min)
        if c.max is not None:
            m &= s_num <= float(c.max)
        mask &= m

    for c in spec.allowed_values:
        if c.mode != 'reject':
            continue
        s = _safe_series(df, c.column)
        if s is None:
            report.warnings.append(f"取值约束列不存在，已忽略: {c.column}")
            continue
        allowed = list(c.values)
        mask &= (s.isin(allowed) | s.isnull())

    if spec.not_null is not None:
        for col in spec.not_null.columns:
            if col not in df.columns:
                report.warnings.append(f"not_null列不存在，已忽略: {col}")
                continue
            mask &= ~df[col].isnull()

    return mask

def apply_constraints(
    df: pd.DataFrame,
    spec: ConstraintSpec,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.Series, ConstraintReport]:
    result = df.copy()
    report = ConstraintReport(clipped={}, repaired={}, rejected_rows=0, warnings=[])

    for c in spec.ranges:
        if c.mode == 'clip':
            _apply_range_clip(result, c, report)

    for c in spec.allowed_values:
        if c.mode == 'repair_random':
            _apply_allowed_repair_random(result, c, rng, report)

    if spec.rules:
        _apply_logical_rules(result, spec.rules, report)

    if spec.formulas:
        _apply_formula_constraints(result, spec.formulas, report)

    _apply_continuous_precision_alignment(result, spec, report)

    for c in spec.ranges:
        if c.mode == 'clip':
            _apply_range_clip(result, c, report)

    mask = _hard_valid_mask(result, spec, report)
    report.rejected_rows = int((~mask).sum())
    return result, mask, report

def enforce_outcome_distribution(
    df: pd.DataFrame,
    outcome_columns: List[str],
    target_dist: Dict[Any, float],
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    return enforce_joint_outcome_distribution(
        df=df,
        outcome_columns=outcome_columns,
        target_dist=target_dist,
        n_samples=n_samples,
        rng=rng,
    )
