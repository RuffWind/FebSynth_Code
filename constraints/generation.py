from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .engine import apply_constraints, enforce_outcome_distribution
from .spec import ConstraintSpec
from utils.outcome_utils import coerce_target_distribution, format_outcome_distribution, outcome_state_series

def _normalize_dist(dist: Dict[Any, float]) -> Dict[Any, float]:
    total = sum(float(v) for v in dist.values())
    if total <= 0:
        raise ValueError("分布之和必须>0")
    return {k: float(v) / total for k, v in dist.items()}

def generate_with_constraints(
    generator: Any,
    n_samples: int,
    spec: ConstraintSpec,
    outcome_distribution_real: Optional[Dict[Any, float]] = None,
    rng_seed: int = 42,
    verbose: bool = True,
    generate_kwargs: Optional[Dict[str, Any]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rng = np.random.default_rng(rng_seed)
    generate_kwargs = dict(generate_kwargs or {})

    outcome_columns = list(spec.outcome.columns) if spec.outcome is not None else []
    target_dist: Optional[Dict[Any, float]] = None

    if spec.outcome is not None:
        if spec.outcome.target == 'real':
            if outcome_distribution_real is None:
                raise ValueError("outcome.target=real 但未提供 outcome_distribution_real")
            target_dist = _normalize_dist(outcome_distribution_real)
        else:
            target_dist = coerce_target_distribution(spec.outcome.target, outcome_columns)

    oversample_factor = spec.outcome.oversample_factor if spec.outcome is not None else 2.0
    max_rounds = spec.outcome.max_rounds if spec.outcome is not None else 20

    pool = []
    reports = []

    if verbose:
        print("\n[约束生成] 开始生成并应用约束")
        if outcome_columns and target_dist:
            print(f"[约束生成] 结局列: {outcome_columns}")
            for line in format_outcome_distribution(outcome_columns, target_dist):
                print(f"[约束生成] 目标联合分布: {line}")

    for round_idx in range(max_rounds):
        current = pd.concat(pool, ignore_index=True) if pool else pd.DataFrame()
        if outcome_columns and target_dist:
            cur_counts = (
                outcome_state_series(current, outcome_columns).value_counts()
                if not current.empty else pd.Series(dtype=int)
            )
            needed = {}
            for state, ratio in target_dist.items():
                needed[state] = int(np.ceil(n_samples * ratio)) - int(cur_counts.get(state, 0))
            remaining = max(0, sum(max(0, v) for v in needed.values()))
        else:
            remaining = max(0, n_samples - (len(current) if not current.empty else 0))

        if remaining <= 0:
            break

        n_generate = int(np.ceil(remaining * oversample_factor))
        if verbose:
            print(f"[约束生成] Round {round_idx+1}/{max_rounds}: 生成候选 {n_generate} 条")

        candidates = generator.generate(n_generate, **generate_kwargs)
        fixed, mask, report = apply_constraints(candidates, spec, rng)
        reports.append(report)
        accepted = fixed[mask].copy()

        if verbose:
            print(f"[约束生成] 通过硬约束: {len(accepted)}/{len(fixed)}")
            if report.clipped:
                top = sorted(report.clipped.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"[约束生成] clip列(前5): {top}")
            if report.repaired:
                top = sorted(report.repaired.items(), key=lambda x: x[1], reverse=True)[:5]
                print(f"[约束生成] repair列(前5): {top}")

        if not accepted.empty:
            pool.append(accepted)

    pooled = pd.concat(pool, ignore_index=True) if pool else pd.DataFrame()

    if pooled.empty:
        raise ValueError("约束生成失败：在最大轮次内未获得任何满足硬约束的样本")

    if outcome_columns and target_dist:
        final = enforce_outcome_distribution(pooled, outcome_columns, target_dist, n_samples, rng)
    else:
        replace = len(pooled) < n_samples
        idx = rng.choice(pooled.index.to_numpy(), size=n_samples, replace=replace)
        final = pooled.loc[idx].reset_index(drop=True)

    diagnostics = {
        'pool_size': int(len(pooled)),
        'rounds': int(len(reports)),
        'rejected_rows_total': int(sum(r.rejected_rows for r in reports)),
        'match_continuous_precision': bool(spec.match_continuous_precision),
        'continuous_precision_columns': sorted(list(spec.continuous_decimal_places.keys())),
        'warnings': [w for r in reports for w in r.warnings],
    }

    if verbose:
        print(f"[约束生成] 完成: pool={diagnostics['pool_size']}, rounds={diagnostics['rounds']}")
        if outcome_columns and all(column in final.columns for column in outcome_columns):
            dist = _normalize_dist(outcome_state_series(final, outcome_columns).value_counts(normalize=True).to_dict())
            for line in format_outcome_distribution(outcome_columns, dist):
                print(f"[约束生成] 最终联合分布: {line}")

    return final, diagnostics
