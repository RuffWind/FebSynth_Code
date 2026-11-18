from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd

RangeMode = Literal['clip', 'reject']
AllowedValuesMode = Literal['keep', 'reject', 'repair_random']
ConditionOp = Literal[
    '==', '!=', '>', '>=', '<', '<=',
    'in', 'not_in',
    'is_null', 'not_null',
]

@dataclass(frozen=True)
class RangeConstraint:
    column: str
    min: Optional[float] = None
    max: Optional[float] = None
    real_trim_count: int = 0
    mode: RangeMode = 'clip'

@dataclass(frozen=True)
class AllowedValuesConstraint:
    column: str
    values: List[Any]
    mode: AllowedValuesMode = 'repair_random'

@dataclass(frozen=True)
class NotNullConstraint:
    columns: List[str]

@dataclass(frozen=True)
class RuleCondition:
    col: str
    op: ConditionOp
    value: Optional[Any] = None

@dataclass(frozen=True)
class RuleAction:
    action: Literal['set']
    col: str
    value: Any

@dataclass(frozen=True)
class LogicalRule:
    name: str
    if_all: List[RuleCondition] = field(default_factory=list)
    then: List[RuleAction] = field(default_factory=list)

@dataclass(frozen=True)
class FormulaConstraint:
    name: str
    target_col: str
    op: Literal['multiply']
    source_cols: List[str]
    constant: float = 1.0
    if_all: List[RuleCondition] = field(default_factory=list)

@dataclass(frozen=True)
class OutcomeConstraint:
    columns: List[str]
    target: Union[Literal['real'], Dict[str, float], Dict[float, float]]
    tolerance: float = 0.02
    oversample_factor: float = 2.5
    max_rounds: int = 30

@dataclass(frozen=True)
class ConstraintSpec:
    version: int = 1
    ranges: List[RangeConstraint] = field(default_factory=list)
    allowed_values: List[AllowedValuesConstraint] = field(default_factory=list)
    not_null: Optional[NotNullConstraint] = None
    rules: List[LogicalRule] = field(default_factory=list)
    formulas: List[FormulaConstraint] = field(default_factory=list)
    outcome: Optional[OutcomeConstraint] = None
    match_continuous_precision: bool = True
    continuous_decimal_places: Dict[str, int] = field(default_factory=dict)

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"不支持的约束版本: {self.version}")

        for r in self.ranges:
            if r.min is None and r.max is None and int(r.real_trim_count) <= 0:
                raise ValueError(f"范围约束至少需要min/max或real_trim_count: {r.column}")
            if int(r.real_trim_count) < 0:
                raise ValueError(f"范围约束real_trim_count不能为负数: {r.column}")
            if r.min is not None and r.max is not None and r.min > r.max:
                raise ValueError(f"范围约束min>max: {r.column} ({r.min}>{r.max})")

        for a in self.allowed_values:
            if not a.values:
                raise ValueError(f"allowed_values.values不能为空: {a.column}")

        for f in self.formulas:
            if not f.target_col:
                raise ValueError(f"formula.target_col不能为空: {f.name}")
            if f.op != 'multiply':
                raise ValueError(f"暂不支持的公式运算: {f.name} ({f.op})")
            if len(f.source_cols) < 1:
                raise ValueError(f"multiply 公式至少需要1个源列: {f.name}")
            if any(not col for col in f.source_cols):
                raise ValueError(f"formula.source_cols 中存在空列名: {f.name}")

        if self.outcome is not None:
            if not self.outcome.columns:
                raise ValueError("outcome.columns不能为空")
            if any(not column for column in self.outcome.columns):
                raise ValueError("outcome.columns 中存在空列名")
            if isinstance(self.outcome.target, dict):
                total = sum(float(v) for v in self.outcome.target.values())
                if total <= 0:
                    raise ValueError("outcome.target的比例之和必须>0")
            if self.outcome.oversample_factor <= 1.0:
                raise ValueError("outcome.oversample_factor建议>1.0")
            if self.outcome.max_rounds <= 0:
                raise ValueError("outcome.max_rounds必须>0")

def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def parse_constraint_spec(raw: Dict[str, Any]) -> ConstraintSpec:
    version = int(raw.get('version', 1))

    ranges: List[RangeConstraint] = []
    allowed_values: List[AllowedValuesConstraint] = []
    rules: List[LogicalRule] = []
    formulas: List[FormulaConstraint] = []
    not_null: Optional[NotNullConstraint] = None
    outcome: Optional[OutcomeConstraint] = None
    match_continuous_precision = raw.get('match_continuous_precision', True)
    if not isinstance(match_continuous_precision, bool):
        raise ValueError("match_continuous_precision 必须是布尔值")

    columns = raw.get('columns', {}) if isinstance(raw.get('columns', {}), dict) else {}

    ranges_raw = columns.get('range', {})
    if isinstance(ranges_raw, dict):
        for col, cfg in ranges_raw.items():
            if not isinstance(cfg, dict):
                continue
            ranges.append(
                RangeConstraint(
                    column=str(col),
                    min=cfg.get('min', None),
                    max=cfg.get('max', None),
                    real_trim_count=int(cfg.get('real_trim_count', 0)),
                    mode=str(cfg.get('mode', 'clip')).lower(),
                )
            )

    allowed_raw = columns.get('allowed_values', {})
    if isinstance(allowed_raw, dict):
        for col, cfg in allowed_raw.items():
            if not isinstance(cfg, dict):
                continue
            allowed_values.append(
                AllowedValuesConstraint(
                    column=str(col),
                    values=_as_list(cfg.get('values')),
                    mode=str(cfg.get('mode', 'repair_random')).lower(),
                )
            )

    not_null_raw = raw.get('not_null', None)
    if isinstance(not_null_raw, dict):
        cols = _as_list(not_null_raw.get('columns'))
        if cols:
            not_null = NotNullConstraint(columns=[str(c) for c in cols])

    rules_raw = raw.get('rules', [])
    if isinstance(rules_raw, list):
        for idx, r in enumerate(rules_raw):
            if not isinstance(r, dict):
                continue
            name = str(r.get('name', f'rule_{idx+1}'))

            if_all: List[RuleCondition] = []
            conds = r.get('if', r.get('if_all', []))
            if isinstance(conds, list):
                for c in conds:
                    if not isinstance(c, dict):
                        continue
                    if_all.append(
                        RuleCondition(
                            col=str(c.get('col')),
                            op=str(c.get('op')).lower(),
                            value=c.get('value', None),
                        )
                    )

            then_actions: List[RuleAction] = []
            thens = r.get('then', [])
            if isinstance(thens, list):
                for a in thens:
                    if not isinstance(a, dict):
                        continue
                    action = str(a.get('action', 'set')).lower()
                    if action != 'set':
                        continue
                    then_actions.append(
                        RuleAction(
                            action='set',
                            col=str(a.get('col')),
                            value=a.get('value', None),
                        )
                    )

            if if_all and then_actions:
                rules.append(LogicalRule(name=name, if_all=if_all, then=then_actions))

    formulas_raw = raw.get('formulas', [])
    if isinstance(formulas_raw, list):
        for idx, item in enumerate(formulas_raw):
            if not isinstance(item, dict):
                continue

            name = str(item.get('name', f'formula_{idx+1}'))
            target_col = str(item.get('target_col', '')).strip()
            op = str(item.get('op', 'multiply')).lower()
            source_cols = [str(col) for col in _as_list(item.get('source_cols')) if str(col).strip()]
            constant_raw = item.get('constant', item.get('factor', 1.0))

            if_all: List[RuleCondition] = []
            conds = item.get('if', item.get('if_all', []))
            if isinstance(conds, list):
                for c in conds:
                    if not isinstance(c, dict):
                        continue
                    if_all.append(
                        RuleCondition(
                            col=str(c.get('col')),
                            op=str(c.get('op')).lower(),
                            value=c.get('value', None),
                        )
                    )

            formulas.append(
                FormulaConstraint(
                    name=name,
                    target_col=target_col,
                    op=op,
                    source_cols=source_cols,
                    constant=float(constant_raw),
                    if_all=if_all,
                )
            )

    outcome_raw = raw.get('outcome', None)
    if isinstance(outcome_raw, dict):
        if 'column' in outcome_raw and 'columns' not in outcome_raw:
            raise ValueError("outcome 配置已升级为多结局接口，请使用 outcome.columns 列表，不再使用 outcome.column")
        cols = _as_list(outcome_raw.get('columns'))
        target = outcome_raw.get('target', 'real')
        if cols:
            outcome = OutcomeConstraint(
                columns=[str(col) for col in cols],
                target=target,
                tolerance=float(outcome_raw.get('tolerance', 0.02)),
                oversample_factor=float(outcome_raw.get('oversample_factor', 2.5)),
                max_rounds=int(outcome_raw.get('max_rounds', 30)),
            )

    spec = ConstraintSpec(
        version=version,
        ranges=ranges,
        allowed_values=allowed_values,
        not_null=not_null,
        rules=rules,
        formulas=formulas,
        outcome=outcome,
        match_continuous_precision=match_continuous_precision,
    )
    spec.validate()
    return spec

def _infer_decimal_places_from_series(series: pd.Series) -> int:
    max_decimal_places = 0
    for value in series.dropna():
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_value):
            continue
        if np.isclose(numeric_value, round(numeric_value)):
            decimal_places = 0
            max_decimal_places = max(max_decimal_places, decimal_places)
            continue
        try:
            decimal_value = Decimal(str(value)).normalize()
        except (InvalidOperation, ValueError):
            try:
                decimal_value = Decimal(str(numeric_value)).normalize()
            except (InvalidOperation, ValueError):
                continue
        exponent = decimal_value.as_tuple().exponent
        decimal_places = max(0, -int(exponent))
        max_decimal_places = max(max_decimal_places, decimal_places)
    return int(max_decimal_places)

def resolve_constraints_with_reference_data(
    spec: ConstraintSpec,
    reference_df: pd.DataFrame,
    continuous_columns: Optional[List[str]] = None,
) -> ConstraintSpec:
    resolved_ranges: List[RangeConstraint] = []
    for constraint in spec.ranges:
        if int(constraint.real_trim_count) <= 0:
            resolved_ranges.append(constraint)
            continue

        if constraint.column not in reference_df.columns:
            raise ValueError(
                f"范围约束 `{constraint.column}` 配置了 real_trim_count，但参考真实数据中不存在该列"
            )

        numeric_values = pd.to_numeric(reference_df[constraint.column], errors='coerce').dropna().to_numpy(dtype=float)
        if numeric_values.size == 0:
            raise ValueError(
                f"范围约束 `{constraint.column}` 配置了 real_trim_count，但参考真实数据中没有可用数值"
            )

        trim_count = int(constraint.real_trim_count)
        if numeric_values.size <= 2 * trim_count:
            raise ValueError(
                f"范围约束 `{constraint.column}` 的 real_trim_count={trim_count} 过大，"
                f"参考真实数据有效数值仅 {numeric_values.size} 条，无法同时删去两端极值"
            )

        sorted_values = np.sort(numeric_values)
        trimmed_values = sorted_values[trim_count: sorted_values.size - trim_count]
        derived_min = float(trimmed_values[0])
        derived_max = float(trimmed_values[-1])
        resolved_ranges.append(
            replace(
                constraint,
                min=float(constraint.min) if constraint.min is not None else derived_min,
                max=float(constraint.max) if constraint.max is not None else derived_max,
            )
        )

    resolved_continuous_decimal_places: Dict[str, int] = {}
    if spec.match_continuous_precision:
        for column in list(continuous_columns or []):
            if column not in reference_df.columns:
                continue
            resolved_continuous_decimal_places[str(column)] = _infer_decimal_places_from_series(reference_df[column])

    resolved_spec = replace(
        spec,
        ranges=resolved_ranges,
        continuous_decimal_places=resolved_continuous_decimal_places,
    )
    resolved_spec.validate()
    return resolved_spec
