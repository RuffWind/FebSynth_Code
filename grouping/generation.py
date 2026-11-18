from __future__ import annotations

import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from constraints.engine import apply_constraints, enforce_outcome_distribution
from constraints.spec import ConstraintSpec
from generators import get_generator
from utils.outcome_utils import (
    coerce_target_distribution,
    format_outcome_distribution,
    get_outcome_joint_distribution,
)

from .spec import GroupFeatureSpec, GroupingSpec

NULL_GROUP_VALUE = "__NULL__"
OTHER_GROUP_VALUE = "__OTHER__"

@dataclass(frozen=True)
class GroupFeatureValue:
    column: str
    kind: str
    value: Any = None
    group_index: Optional[int] = None
    lower: Optional[float] = None
    upper: Optional[float] = None
    include_null: bool = False

@dataclass(frozen=True)
class GroupDefinition:
    key: Tuple[Any, ...]
    display_name: str
    feature_values: Tuple[GroupFeatureValue, ...]
    merged_member_feature_values: Tuple[Tuple[GroupFeatureValue, ...], ...] = ()

def _normalize_dist(dist: Dict[Any, float]) -> Dict[Any, float]:
    normalized = {k: float(v) for k, v in dist.items()}
    total = sum(normalized.values())
    if total <= 0:
        raise ValueError("分布之和必须>0")
    return {k: v / total for k, v in normalized.items()}

def _sanitize_key(text: str) -> str:
    cleaned = re.sub(r'[^0-9a-zA-Z_\-.]+', '_', text.strip())
    return cleaned[:120] or "group"

def _largest_remainder_allocate(counts: pd.Series, total_samples: int) -> pd.Series:
    if counts.empty:
        raise ValueError("无法分配分组样本数：有效分组为空")

    proportions = counts / counts.sum()
    expected = proportions * total_samples
    base = np.floor(expected).astype(int)
    remainder = int(total_samples - base.sum())

    if remainder > 0:

        fractional = (expected - base).to_numpy()
        order = np.argsort(-fractional, kind='stable')
        for pos in order[:remainder]:
            base.iloc[int(pos)] += 1

    return base.astype(int)

def _ordered_group_counts(
    group_keys: pd.Series,
    group_defs: Dict[Tuple[Any, ...], GroupDefinition],
) -> pd.Series:
    counts = group_keys.value_counts()
    ordered_counts = {
        key: int(counts[key])
        for key in group_defs.keys()
        if key in counts.index
    }
    return pd.Series(ordered_counts, dtype=int)

def _format_group_value(group_value: GroupFeatureValue) -> str:
    if group_value.kind == 'range':
        if group_value.lower is None:
            return f"{group_value.column}<= {group_value.upper}"
        if group_value.upper is None:
            return f"{group_value.column}> {group_value.lower}"
        return f"{group_value.lower}< {group_value.column}<= {group_value.upper}"
    if group_value.value == NULL_GROUP_VALUE:
        return f"{group_value.column}=NULL"
    return f"{group_value.column}={group_value.value}"

def _build_feature_assignment(
    series: pd.Series,
    feature: GroupFeatureSpec,
) -> Tuple[pd.Series, Dict[Any, GroupFeatureValue], int]:
    assignment = pd.Series(index=series.index, dtype='object')
    value_map: Dict[Any, GroupFeatureValue] = {}
    excluded_missing = 0

    if feature.thresholds:
        numeric = pd.to_numeric(series, errors='coerce')
        missing_mask = numeric.isnull()
        thresholds = list(feature.thresholds)

        codes = pd.Series(index=series.index, dtype='object')
        if thresholds:
            codes.loc[numeric.notnull()] = np.searchsorted(thresholds, numeric[numeric.notnull()], side='left')

        if (feature.include_null or feature.missing_to_other) and missing_mask.any():
            codes.loc[missing_mask] = NULL_GROUP_VALUE
            value_map[NULL_GROUP_VALUE] = GroupFeatureValue(
                column=feature.column,
                kind='exact',
                value=NULL_GROUP_VALUE,
                include_null=True,
            )
        else:
            excluded_missing = int(missing_mask.sum())

        for group_idx in range(len(thresholds) + 1):
            if group_idx == 0:
                lower = None
                upper = float(thresholds[0])
            elif group_idx == len(thresholds):
                lower = float(thresholds[-1])
                upper = None
            else:
                lower = float(thresholds[group_idx - 1])
                upper = float(thresholds[group_idx])

            value_map[group_idx] = GroupFeatureValue(
                column=feature.column,
                kind='range',
                group_index=group_idx,
                lower=lower,
                upper=upper,
            )

        assignment = codes
        return assignment, value_map, excluded_missing

    assignment = series.astype('object').copy()
    missing_mask = assignment.isnull()
    if (feature.include_null or feature.missing_to_other) and missing_mask.any():
        assignment.loc[missing_mask] = NULL_GROUP_VALUE
        value_map[NULL_GROUP_VALUE] = GroupFeatureValue(
            column=feature.column,
            kind='exact',
            value=NULL_GROUP_VALUE,
            include_null=True,
        )
    else:
        excluded_missing = int(missing_mask.sum())
        assignment.loc[missing_mask] = pd.NA

    for value in pd.unique(assignment.dropna()):
        value_map[value] = GroupFeatureValue(
            column=feature.column,
            kind='exact',
            value=value,
            include_null=(value == NULL_GROUP_VALUE),
        )

    return assignment, value_map, excluded_missing

def _build_group_index(
    data_raw: pd.DataFrame,
    spec: GroupingSpec,
) -> Tuple[pd.Series, Dict[Tuple[Any, ...], GroupDefinition], List[str]]:
    assignments: List[pd.Series] = []
    feature_maps: List[Dict[Any, GroupFeatureValue]] = []
    warnings: List[str] = []

    for feature in spec.features:
        if feature.column not in data_raw.columns:
            raise ValueError(f"分组特征不存在: {feature.column}")
        assignment, value_map, excluded_missing = _build_feature_assignment(data_raw[feature.column], feature)
        assignments.append(assignment)
        feature_maps.append(value_map)
        if excluded_missing > 0:
            warnings.append(
                f"分组特征'{feature.column}'有{excluded_missing}行缺失值未纳入任何分组；"
                f"如需单独成组，请在分组JSON中设置 include_null=true"
            )

    assignment_df = pd.concat(assignments, axis=1)
    valid_mask = assignment_df.notnull().all(axis=1)
    assignment_df = assignment_df.loc[valid_mask]

    group_keys = assignment_df.apply(lambda row: tuple(row.tolist()), axis=1)
    group_defs: Dict[Tuple[Any, ...], GroupDefinition] = {}
    for key in group_keys.drop_duplicates().tolist():
        feature_values = tuple(
            feature_maps[idx][key[idx]]
            for idx in range(len(spec.features))
        )
        display_name = " | ".join(_format_group_value(item) for item in feature_values)
        group_defs[key] = GroupDefinition(
            key=key,
            display_name=display_name,
            feature_values=feature_values,
        )

    missing_to_other_positions = [
        idx for idx, feature in enumerate(spec.features) if feature.missing_to_other
    ]
    forced_other_keys = {
        key for key in group_defs
        if any(key[idx] == NULL_GROUP_VALUE for idx in missing_to_other_positions)
    }
    if spec.rare_group_min_ratio > 0 or forced_other_keys:
        group_keys, group_defs, merge_warnings = _merge_rare_joint_groups(
            group_keys=group_keys,
            group_defs=group_defs,
            min_ratio=spec.rare_group_min_ratio,
            n_features=len(spec.features),
            forced_other_keys=forced_other_keys,
        )
        warnings.extend(merge_warnings)

    return group_keys, group_defs, warnings

def _merge_rare_joint_groups(
    group_keys: pd.Series,
    group_defs: Dict[Tuple[Any, ...], GroupDefinition],
    min_ratio: float,
    n_features: int,
    forced_other_keys: Optional[set[Tuple[Any, ...]]] = None,
) -> Tuple[pd.Series, Dict[Tuple[Any, ...], GroupDefinition], List[str]]:
    warnings: List[str] = []
    if group_keys.empty:
        return group_keys, group_defs, warnings

    counts = group_keys.value_counts()
    total = int(counts.sum())
    rare_keys = [
        key for key, count in counts.items()
        if (count / total) < min_ratio or key in (forced_other_keys or ())
    ]
    if not rare_keys:
        return group_keys, group_defs, warnings

    other_key = tuple([OTHER_GROUP_VALUE] * n_features)
    merged_member_feature_values = tuple(
        group_defs[key].feature_values
        for key in rare_keys
    )
    other_display_name = f"其他(合并{len(rare_keys)}个稀有/缺失组)"

    merged_group_keys = group_keys.copy()
    merged_group_keys.loc[merged_group_keys.isin(rare_keys)] = [other_key] * int(merged_group_keys.isin(rare_keys).sum())

    merged_group_defs = {
        key: group_def
        for key, group_def in group_defs.items()
        if key not in rare_keys
    }
    merged_group_defs[other_key] = GroupDefinition(
        key=other_key,
        display_name=other_display_name,
        feature_values=tuple(),
        merged_member_feature_values=merged_member_feature_values,
    )

    rare_count_total = int(sum(counts[key] for key in rare_keys))
    warnings.append(
        "联合分组中有"
        f"{len(rare_keys)}个稀有或指定缺失组并入'其他'（稀有阈值 rare_group_min_ratio={min_ratio:.4f}），"
        f"合计 {rare_count_total}/{total} 行，{rare_count_total / total:.2%}"
    )

    return merged_group_keys, merged_group_defs, warnings

def _match_group_feature(df: pd.DataFrame, group_value: GroupFeatureValue) -> pd.Series:
    if group_value.column not in df.columns:
        return pd.Series([False] * len(df), index=df.index)

    series = df[group_value.column]

    if group_value.kind == 'exact':
        if group_value.value == NULL_GROUP_VALUE:
            return series.isnull()
        return series.eq(group_value.value)

    numeric = pd.to_numeric(series, errors='coerce')
    if group_value.lower is None and group_value.upper is None:
        return numeric.notnull()
    if group_value.lower is None:
        return numeric.notnull() & (numeric <= group_value.upper)
    if group_value.upper is None:
        return numeric.notnull() & (numeric > group_value.lower)
    return numeric.notnull() & (numeric > group_value.lower) & (numeric <= group_value.upper)

def _match_feature_values(
    df: pd.DataFrame,
    feature_values: Tuple[GroupFeatureValue, ...],
    skip_missing_columns: bool = False,
) -> pd.Series:
    mask = pd.Series([True] * len(df), index=df.index)
    matched_columns = 0
    for group_value in feature_values:
        if group_value.column not in df.columns:
            if skip_missing_columns:
                continue
            return pd.Series([False] * len(df), index=df.index)
        mask &= _match_group_feature(df, group_value)
        matched_columns += 1

    if skip_missing_columns and matched_columns == 0:
        return pd.Series([True] * len(df), index=df.index)

    return mask

def _match_group(
    df: pd.DataFrame,
    group_def: GroupDefinition,
    skip_missing_columns: bool = False,
) -> pd.Series:
    if group_def.merged_member_feature_values:
        merged_mask = pd.Series([False] * len(df), index=df.index)
        for member_feature_values in group_def.merged_member_feature_values:
            merged_mask |= _match_feature_values(
                df,
                member_feature_values,
                skip_missing_columns=skip_missing_columns,
            )
        return merged_mask

    return _match_feature_values(
        df,
        group_def.feature_values,
        skip_missing_columns=skip_missing_columns,
    )

def _group_definition_columns(group_def: GroupDefinition) -> List[str]:
    columns = {group_value.column for group_value in group_def.feature_values}
    for member_feature_values in group_def.merged_member_feature_values:
        for group_value in member_feature_values:
            columns.add(group_value.column)
    return list(columns)

def _fixed_exact_group_values(group_def: GroupDefinition) -> Dict[str, Any]:
    if group_def.merged_member_feature_values:
        return {}

    fixed_values: Dict[str, Any] = {}
    for group_value in group_def.feature_values:
        if group_value.kind != 'exact':
            continue
        fixed_values[group_value.column] = np.nan if group_value.value == NULL_GROUP_VALUE else group_value.value
    return fixed_values

def _generated_columns_for_group(
    ordered_generation_columns: List[str],
    group_def: GroupDefinition,
) -> List[str]:
    fixed_exact_columns = set(_fixed_exact_group_values(group_def).keys())
    return [column for column in ordered_generation_columns if column not in fixed_exact_columns]

def _filter_group_model_kwargs(
    kwargs: Dict[str, Any],
    model_columns: List[str],
) -> Dict[str, Any]:
    result = dict(kwargs)
    model_column_set = set(model_columns)
    for key in ('continuous_columns', 'categorical_columns', 'outcome_columns'):
        if key in result:
            result[key] = [
                column for column in result.get(key, [])
                if column in model_column_set
            ]
    return result

def _apply_fixed_exact_group_columns(df: pd.DataFrame, group_def: GroupDefinition) -> pd.DataFrame:
    result = df.copy()
    for column, value in _fixed_exact_group_values(group_def).items():
        result[column] = value
    return result

def _sample_from_real_group(
    group_data: pd.DataFrame,
    n_samples: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    replace = len(group_data) < n_samples
    random_state = int(rng.integers(0, 2**31 - 1))
    return group_data.sample(n=n_samples, replace=replace, random_state=random_state).reset_index(drop=True)

def _make_group_save_dir(base_save_dir: Optional[str], model_name: str, group_def: GroupDefinition) -> str:
    base_dir = base_save_dir or os.path.join("models", model_name.lower())
    return os.path.join(base_dir, "grouped", _sanitize_key(group_def.display_name))

def _exception_debug_info(exc: Exception) -> Dict[str, Any]:
    return {
        'exception_type': type(exc).__name__,
        'exception_str': str(exc),
        'exception_repr': repr(exc),
        'traceback': traceback.format_exc(),
    }

def _generate_one_group(
    generator: Any,
    provider: str,
    group_def: GroupDefinition,
    group_data_processed: pd.DataFrame,
    n_samples: int,
    grouping_spec: GroupingSpec,
    constraints_spec: Optional[ConstraintSpec],
    generate_kwargs: Dict[str, Any],
    rng: np.random.Generator,
    verbose: bool,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    accepted_pool: List[pd.DataFrame] = []
    warnings: List[str] = []
    rounds = 0
    rejected_total = 0

    for round_idx in range(grouping_spec.max_rounds):
        rounds = round_idx + 1
        current_count = sum(len(x) for x in accepted_pool)
        remaining = n_samples - current_count
        if remaining <= 0:
            break

        n_generate = int(np.ceil(remaining * grouping_spec.oversample_factor))
        if provider == 'bootstrap':
            candidates = _sample_from_real_group(group_data_processed, n_generate, rng)
        elif provider == 'fixed':
            candidates = pd.DataFrame(index=pd.RangeIndex(n_generate))
        else:
            candidates = generator.generate(n_generate, **generate_kwargs)

        fixed = _apply_fixed_exact_group_columns(candidates, group_def)
        constraint_mask = pd.Series([True] * len(candidates), index=candidates.index)
        if constraints_spec is not None:
            fixed, constraint_mask, report = apply_constraints(fixed, constraints_spec, rng)
            warnings.extend(report.warnings)
            rejected_total += int(report.rejected_rows)

        group_mask = _match_group(fixed, group_def)
        group_columns = _group_definition_columns(group_def)
        if not group_mask.any() and any(
            column not in fixed.columns
            for column in group_columns
        ):
            group_mask = _match_group(fixed, group_def, skip_missing_columns=True)
        accepted = fixed[constraint_mask & group_mask].copy()
        if not accepted.empty:
            accepted_pool.append(accepted)

        if verbose:
            print(
                f"    Round {rounds}/{grouping_spec.max_rounds}: "
                f"候选={len(candidates)}, 通过={len(accepted)}, 已累计={current_count + len(accepted)}"
            )

    pooled = pd.concat(accepted_pool, ignore_index=True) if accepted_pool else pd.DataFrame(columns=group_data_processed.columns)
    if pooled.empty:
        raise ValueError(f"分组 '{group_def.display_name}' 在最大轮次内未获得任何有效样本")

    replace = len(pooled) < n_samples
    idx = rng.choice(pooled.index.to_numpy(), size=n_samples, replace=replace)
    final_group = pooled.loc[idx].reset_index(drop=True)
    final_group = _apply_fixed_exact_group_columns(final_group, group_def)

    diagnostics = {
        'group_name': group_def.display_name,
        'provider': provider,
        'rounds': rounds,
        'pool_size': int(len(pooled)),
        'warnings': warnings,
        'rejected_rows_total': rejected_total,
    }
    return final_group, diagnostics

def generate_by_groups(
    model_name: str,
    model_config: Dict[str, Any],
    data_raw: pd.DataFrame,
    data_processed: pd.DataFrame,
    grouping_spec: GroupingSpec,
    n_samples: int,
    fit_kwargs: Dict[str, Any],
    generate_kwargs: Dict[str, Any],
    constraints_spec: Optional[ConstraintSpec] = None,
    outcome_distribution_real: Optional[Dict[Any, float]] = None,
    generation_columns: Optional[List[str]] = None,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if generation_columns is None:
        ordered_generation_columns = list(data_processed.columns)
    else:
        generation_set = set(generation_columns)
        ordered_generation_columns = [
            col for col in data_processed.columns
            if col in generation_set
        ]
        if not ordered_generation_columns:
            raise ValueError("分组条件生成未收到任何有效的目标生成列")

    rng = np.random.default_rng(grouping_spec.random_seed)
    group_keys, group_defs, build_warnings = _build_group_index(data_raw, grouping_spec)
    real_group_counts = _ordered_group_counts(group_keys, group_defs)
    final_group_counts = _largest_remainder_allocate(real_group_counts, n_samples)

    pooled_target_total = n_samples
    pooled_group_counts = final_group_counts
    if constraints_spec is not None and constraints_spec.outcome is not None:
        pooled_target_total = int(np.ceil(n_samples * constraints_spec.outcome.oversample_factor))
        pooled_group_counts = _largest_remainder_allocate(real_group_counts, pooled_target_total)

    diagnostics_groups: List[Dict[str, Any]] = []
    warnings: List[str] = list(build_warnings)
    generated_groups: List[pd.DataFrame] = []
    total_training_time = 0.0
    total_generation_time = 0.0

    if verbose:
        print("\n[分组条件生成] 开始按真实分组比例生成")
        print(f"[分组条件生成] 有效分组数: {len(real_group_counts)}")
        if constraints_spec is not None and constraints_spec.outcome is not None:
            print(
                f"[分组条件生成] 已启用 outcome 约束过采样池: "
                f"最终目标样本数={n_samples}, 分组合并池目标数={pooled_target_total}"
            )

    for group_idx, (group_key, pooled_target_count) in enumerate(pooled_group_counts.items(), start=1):
        if pooled_target_count <= 0:
            continue

        group_def = group_defs[group_key]
        row_index = group_keys[group_keys == group_key].index
        group_data_processed = data_processed.loc[row_index, ordered_generation_columns].copy()
        group_model_columns = _generated_columns_for_group(ordered_generation_columns, group_def)
        group_model_data = data_processed.loc[row_index, group_model_columns].copy()
        group_fit_kwargs = _filter_group_model_kwargs(fit_kwargs, group_model_columns)
        group_generate_kwargs = _filter_group_model_kwargs(generate_kwargs, group_model_columns)

        group_generate_kwargs['correct_outcome_distribution'] = False

        subgroup_outcome_cols = [
            column for column in group_fit_kwargs.get('outcome_columns', [])
            if column in group_model_data.columns
        ]
        if subgroup_outcome_cols:
            subgroup_outcome_dist = get_outcome_joint_distribution(group_model_data, subgroup_outcome_cols)
            group_fit_kwargs['outcome_distribution'] = subgroup_outcome_dist
        else:
            group_fit_kwargs.pop('outcome_distribution', None)

        provider = 'model'
        generator = None
        fit_failure_debug: Optional[Dict[str, Any]] = None

        if verbose:
            print(
                f"\n[分组条件生成] 组 {group_idx}/{len(pooled_group_counts)}: "
                f"{group_def.display_name}"
            )
            print(
                f"  真实样本数: {len(group_data_processed)}, "
                f"最终目标数: {int(final_group_counts[group_key])}, "
                f"组内池目标数: {int(pooled_target_count)}"
            )
            fixed_exact_columns = sorted(set(ordered_generation_columns) - set(group_model_columns))
            if fixed_exact_columns:
                print(f"  直接固定的 exact 分组列: {fixed_exact_columns}")

        if not group_model_columns:
            provider = 'fixed'
            warnings.append(
                f"分组 '{group_def.display_name}' 的目标生成列均为 exact 分组常量，"
                "已直接固定赋值生成"
            )
        elif len(group_data_processed) < grouping_spec.min_group_size:
            provider = 'bootstrap'
            warnings.append(
                f"分组 '{group_def.display_name}' 真实样本数仅 {len(group_data_processed)}，"
                f"低于 min_group_size={grouping_spec.min_group_size}，已回退为 bootstrap 生成"
            )
        else:
            group_model_config = dict(model_config)
            group_model_config['save_dir'] = _make_group_save_dir(
                base_save_dir=model_config.get('save_dir'),
                model_name=model_name,
                group_def=group_def,
            )

            try:
                generator = get_generator(model_name, **group_model_config)
                start_fit = time.time()
                if group_fit_kwargs.get('load_model', False):
                    try:
                        generator.load(checkpoint='best')
                    except FileNotFoundError:
                        generator.fit(group_model_data, **group_fit_kwargs)
                    if not getattr(generator, 'is_fitted', False):
                        raise RuntimeError(
                            f"分组 '{group_def.display_name}' 的已保存模型加载后仍处于未训练状态"
                        )
                else:
                    generator.fit(group_model_data, **group_fit_kwargs)
                total_training_time += time.time() - start_fit
            except Exception as exc:
                if group_fit_kwargs.get('load_model', False):
                    raise RuntimeError(
                        f"分组 '{group_def.display_name}' 的已保存模型加载失败，"
                        "已阻止静默回退为 bootstrap"
                    ) from exc
                if not grouping_spec.fallback_to_bootstrap:
                    raise
                provider = 'bootstrap'
                debug_info = _exception_debug_info(exc)
                warnings.append(
                    f"分组 '{group_def.display_name}' 的模型训练/加载失败，"
                    f"已回退为 bootstrap 生成: {debug_info['exception_repr']}"
                )
                generator = None
                fit_failure_debug = {
                    'stage': 'fit_or_load',
                    'model_name': model_name,
                    'group_name': group_def.display_name,
                    'provider_after_fallback': provider,
                    'group_model_shape': [int(group_model_data.shape[0]), int(group_model_data.shape[1])],
                    'group_model_columns': list(group_model_columns),
                    'group_model_dtypes': {col: str(dtype) for col, dtype in group_model_data.dtypes.items()},
                    'group_fit_kwargs_summary': {
                        'outcome_columns': list(group_fit_kwargs.get('outcome_columns', []) or []),
                        'continuous_columns': list(group_fit_kwargs.get('continuous_columns', []) or []),
                        'categorical_columns': list(group_fit_kwargs.get('categorical_columns', []) or []),
                        'load_model': bool(group_fit_kwargs.get('load_model', False)),
                        'save_best': bool(group_fit_kwargs.get('save_best', False)),
                    },
                    'model_config': dict(group_model_config),
                    'exception': debug_info,
                }

        fallback_reason: Optional[str] = None
        training_debug: Optional[Dict[str, Any]] = fit_failure_debug

        start_generate = time.time()
        try:
            final_group, group_diag = _generate_one_group(
                generator=generator,
                provider=provider,
                group_def=group_def,
                group_data_processed=group_data_processed,
                n_samples=int(pooled_target_count),
                grouping_spec=grouping_spec,
                constraints_spec=constraints_spec if constraints_spec is not None else None,
                generate_kwargs=group_generate_kwargs,
                rng=rng,
                verbose=verbose,
            )
            total_generation_time += time.time() - start_generate
        except Exception as exc:
            total_generation_time += time.time() - start_generate
            if group_fit_kwargs.get('load_model', False):
                raise RuntimeError(
                    f"分组 '{group_def.display_name}' 的已加载模型生成失败，"
                    "已阻止静默回退为 bootstrap"
                ) from exc
            if provider != 'model' or not grouping_spec.fallback_to_bootstrap:
                raise

            provider = 'bootstrap'
            debug_info = _exception_debug_info(exc)
            fallback_reason = debug_info['exception_repr']
            warnings.append(
                f"分组 '{group_def.display_name}' 的模型生成失败或未产生有效样本，"
                f"已回退为 bootstrap 生成: {debug_info['exception_repr']}"
            )

            fallback_start = time.time()
            final_group, group_diag = _generate_one_group(
                generator=None,
                provider=provider,
                group_def=group_def,
                group_data_processed=group_data_processed,
                n_samples=int(pooled_target_count),
                grouping_spec=grouping_spec,
                constraints_spec=constraints_spec if constraints_spec is not None else None,
                generate_kwargs=group_generate_kwargs,
                rng=rng,
                verbose=verbose,
            )
            total_generation_time += time.time() - fallback_start
            training_debug = {
                'stage': 'generate',
                'model_name': model_name,
                'group_name': group_def.display_name,
                'provider_after_fallback': provider,
                'group_model_shape': [int(group_model_data.shape[0]), int(group_model_data.shape[1])],
                'group_model_columns': list(group_model_columns),
                'group_model_dtypes': {col: str(dtype) for col, dtype in group_model_data.dtypes.items()},
                'group_generate_kwargs_summary': {
                    'outcome_columns': list(group_generate_kwargs.get('outcome_columns', []) or []),
                    'continuous_columns': list(group_generate_kwargs.get('continuous_columns', []) or []),
                    'categorical_columns': list(group_generate_kwargs.get('categorical_columns', []) or []),
                    'correct_outcome_distribution': bool(group_generate_kwargs.get('correct_outcome_distribution', False)),
                },
                'exception': debug_info,
            }

        group_diag.update({
            'target_count': int(final_group_counts[group_key]),
            'pooled_target_count': int(pooled_target_count),
            'real_count': int(len(group_data_processed)),
            'fixed_exact_columns': sorted(set(ordered_generation_columns) - set(group_model_columns)),
            'generated_columns_count': int(len(group_model_columns)),
        })
        if fallback_reason is not None:
            group_diag['fallback_reason'] = fallback_reason
        if training_debug is not None:
            group_diag['debug'] = training_debug
        diagnostics_groups.append(group_diag)
        generated_groups.append(final_group)

    if not generated_groups:
        raise ValueError("分组条件生成失败：没有成功生成任何分组样本")

    final = pd.concat(generated_groups, ignore_index=True)
    final = final.reindex(columns=ordered_generation_columns)
    final = final.sample(frac=1, random_state=grouping_spec.random_seed).reset_index(drop=True)

    if constraints_spec is not None and constraints_spec.outcome is not None:
        outcome_columns = list(constraints_spec.outcome.columns)
        if constraints_spec.outcome.target == 'real':
            if outcome_distribution_real is None:
                raise ValueError("分组条件生成需要全局 outcome=real 校正，但未提供真实结局分布")
            target_dist = _normalize_dist(outcome_distribution_real)
        else:
            target_dist = coerce_target_distribution(constraints_spec.outcome.target, outcome_columns)

        if verbose:
            print("\n[分组条件生成] 对最终合并结果执行全局结局分布校正")
            print(f"[分组条件生成] 结局列: {outcome_columns}")
            for line in format_outcome_distribution(outcome_columns, target_dist):
                print(f"[分组条件生成] 目标联合分布: {line}")

        final = enforce_outcome_distribution(
            df=final,
            outcome_columns=outcome_columns,
            target_dist=target_dist,
            n_samples=n_samples,
            rng=rng,
        )
        warnings.append("已在分组生成完成后执行全局 outcome 分布校正；这可能使最终分组比例与真实比例产生轻微偏差")

    diagnostics = {
        'group_count': int(len(diagnostics_groups)),
        'excluded_rows': int(len(data_raw) - len(group_keys)),
        'pooled_target_total': int(pooled_target_total),
        'final_target_total': int(n_samples),
        'training_time': float(total_training_time),
        'generation_time': float(total_generation_time),
        'warnings': warnings + [w for g in diagnostics_groups for w in g.get('warnings', [])],
        'groups': diagnostics_groups,
    }

    if verbose:
        print(f"\n[分组条件生成] 完成: 总分组={diagnostics['group_count']}, 样本数={len(final)}")

    return final, diagnostics
