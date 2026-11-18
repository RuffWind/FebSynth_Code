from __future__ import annotations

import itertools
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from output_localization import OutputLocalizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import PrivacyEvaluator
from .context import EvaluationContext
from .eval_config import (
    ATTRIBUTE_INFERENCE_K_LIST,
    ATTRIBUTE_INFERENCE_N_ATT_LIST,
    ATTRIBUTE_INFERENCE_N_TOKEN_RANGE,
    ATTRIBUTE_INFERENCE_P_GIVEN_LIST,
)
from .results import MetricResult
from .utils import (
    configure_matplotlib_for_chinese,
    normalize_categorical_dataframe,
    REAL_DATA_COLOR,
    REFERENCE_LINE_COLOR,
    SECONDARY_ACCENT_COLOR,
    SYNTHETIC_DATA_COLOR,
)

PRIMARY_COLOR = REAL_DATA_COLOR
SECONDARY_COLOR = SYNTHETIC_DATA_COLOR
ACCENT_COLOR = SECONDARY_ACCENT_COLOR
REFERENCE_COLOR = REFERENCE_LINE_COLOR

def _filter_distance_outliers(values: np.ndarray, whisker_scale: float = 3.0) -> np.ndarray:
    clean_values = np.asarray(values, dtype=float)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size < 4:
        return clean_values

    q1, q3 = np.percentile(clean_values, [25, 75])
    iqr = float(q3 - q1)
    if not np.isfinite(iqr) or iqr <= 0:
        return clean_values

    lower_bound = float(q1 - whisker_scale * iqr)
    upper_bound = float(q3 + whisker_scale * iqr)
    filtered = clean_values[(clean_values >= lower_bound) & (clean_values <= upper_bound)]
    return filtered if filtered.size > 0 else clean_values

def _resolve_hist_bin_count(*value_groups: np.ndarray) -> int:
    sizes = [int(group.size) for group in value_groups if group.size > 0]
    if not sizes:
        return 12
    return int(min(50, max(12, np.sqrt(max(sizes)))))

def _compute_nearest_neighbor_distances(
    query_matrix: np.ndarray,
    reference_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    neighbor_model = NearestNeighbors(n_neighbors=1, metric="euclidean", algorithm="auto")
    neighbor_model.fit(reference_matrix)
    distances, indices = neighbor_model.kneighbors(query_matrix)
    return (
        distances.reshape(-1).astype(float),
        indices.reshape(-1).astype(int),
    )

def save_dcr_distribution_plot(
    details_df: pd.DataFrame,
    output_dir: str,
    localizer: OutputLocalizer | None = None,
) -> List[str]:
    if details_df.empty or "dcr_distance" not in details_df.columns:
        return []
    localizer = localizer or OutputLocalizer()
    localizer.configure_matplotlib()

    output_path = os.path.join(output_dir, "dcr_distance_distribution.png")
    os.makedirs(output_dir, exist_ok=True)

    if "comparison_type" in details_df.columns:
        synthetic_distances = pd.to_numeric(
            details_df.loc[details_df["comparison_type"] == "synthetic_to_real", "dcr_distance"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
        baseline_distances = pd.to_numeric(
            details_df.loc[details_df["comparison_type"] == "real_holdout_to_real_baseline", "dcr_distance"],
            errors="coerce",
        ).dropna().to_numpy(dtype=float)
    else:
        synthetic_distances = pd.to_numeric(details_df["dcr_distance"], errors="coerce").dropna().to_numpy(dtype=float)
        baseline_distances = np.array([], dtype=float)

    if synthetic_distances.size == 0 and baseline_distances.size == 0:
        return []

    synthetic_display = _filter_distance_outliers(synthetic_distances)
    baseline_display = _filter_distance_outliers(baseline_distances)
    bin_count = _resolve_hist_bin_count(synthetic_display, baseline_display)

    combined_display = np.concatenate([
        group for group in (synthetic_display, baseline_display) if group.size > 0
    ]) if (synthetic_display.size > 0 or baseline_display.size > 0) else np.array([], dtype=float)
    x_limits = None
    if combined_display.size > 0:
        x_min = float(np.min(combined_display))
        x_max = float(np.max(combined_display))
        if np.isclose(x_min, x_max):
            padding = 1.0 if np.isclose(x_min, 0.0) else abs(x_min) * 0.05
            x_limits = (x_min - padding, x_max + padding)
        else:
            padding = (x_max - x_min) * 0.03
            x_limits = (x_min - padding, x_max + padding)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2) if localizer.is_english else (14, 6))

    panel_specs = [
        {
            "values": synthetic_distances,
            "display_values": synthetic_display,
            "panel_label": "a",
            "title": "Synthetic-to-real DCR" if localizer.is_english else "Synthetic-real DCR 分布",
            "bar_color": SECONDARY_COLOR,
            "y_label": "Synthetic sample count" if localizer.is_english else "合成样本数",
        },
        {
            "values": baseline_distances,
            "display_values": baseline_display,
            "panel_label": "b",
            "title": "Real-to-real baseline DCR" if localizer.is_english else "Real-real baseline DCR 分布",
            "bar_color": PRIMARY_COLOR,
            "y_label": "Holdout real sample count" if localizer.is_english else "holdout 真实样本数",
        },
    ]

    for ax, panel in zip(axes, panel_specs):
        values = panel["values"]
        display_values = panel["display_values"]
        ax.text(-0.12, 1.05, panel["panel_label"], transform=ax.transAxes, fontsize=11, fontweight="bold")
        if values.size == 0:
            ax.text(
                0.5,
                0.5,
                "Not available" if localizer.is_english else "暂无可用结果",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=REFERENCE_COLOR,
            )
            ax.set_title(panel["title"])
            ax.set_xlabel("Euclidean distance to closest training record" if localizer.is_english else "到最近训练集记录的欧氏距离")
            ax.set_ylabel(panel["y_label"])
            ax.grid(alpha=0.2, linestyle="--")
            continue

        median_distance = float(np.median(values))
        percentile_5 = float(np.percentile(values, 5))
        ax.hist(
            display_values,
            bins=bin_count,
            color=panel["bar_color"],
            edgecolor="white",
            alpha=0.82,
        )
        ax.axvline(
            median_distance,
            color=REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.15,
            label=(f"Median = {median_distance:.4f}" if localizer.is_english else f"median={median_distance:.4f}"),
        )
        ax.axvline(
            percentile_5,
            color=ACCENT_COLOR,
            linestyle=":",
            linewidth=1.15,
            label=(f"5th percentile = {percentile_5:.4f}" if localizer.is_english else f"p5={percentile_5:.4f}"),
        )
        ax.set_title(panel["title"])
        ax.set_xlabel("Euclidean distance to closest training record" if localizer.is_english else "到最近训练集记录的欧氏距离")
        ax.set_ylabel(panel["y_label"])
        if x_limits is not None:
            ax.set_xlim(*x_limits)
        ax.legend(frameon=False)
        ax.grid(alpha=0.2, linestyle="--")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300 if localizer.is_english else 200, bbox_inches="tight")
    plt.close(fig)
    return [output_path]

def _build_preprocessor(
    continuous_columns: List[str],
    categorical_columns: List[str],
) -> ColumnTransformer:
    transformers = []
    if continuous_columns:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            continuous_columns,
        ))
    if categorical_columns:
        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_columns,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")

def _resolve_privacy_feature_columns(
    context: EvaluationContext,
) -> Tuple[List[str], List[str], List[str]]:
    feature_columns = [
        col for col in context.schema.common_columns
        if col in context.comparison_real_data.columns and col in context.synthetic_data.columns
    ]
    continuous_columns = [col for col in feature_columns if col in context.schema.continuous_columns]
    categorical_like_columns = set(context.schema.categorical_columns) | set(context.schema.outcome_columns)
    categorical_columns = [col for col in feature_columns if col in categorical_like_columns and col not in continuous_columns]
    remaining_columns = [col for col in feature_columns if col not in continuous_columns and col not in categorical_columns]
    categorical_columns.extend(remaining_columns)
    return feature_columns, continuous_columns, categorical_columns

def _build_discretization_bin_labels(thresholds: List[float]) -> List[str]:
    edges = [-np.inf, *thresholds, np.inf]
    labels = []
    for start, end in zip(edges[:-1], edges[1:]):
        if np.isneginf(start):
            labels.append(f"<={end:.6g}")
        elif np.isposinf(end):
            labels.append(f">{start:.6g}")
        else:
            labels.append(f"({start:.6g},{end:.6g}]")
    return labels

def _build_quantile_thresholds(
    reference_series: pd.Series,
    n_token_range: int,
) -> List[float]:
    if int(n_token_range) < 2:
        raise ValueError("ATTRIBUTE_INFERENCE_N_TOKEN_RANGE 必须是大于等于 2 的整数。")

    reference_numeric = pd.to_numeric(reference_series, errors="coerce").dropna()
    if reference_numeric.empty:
        return []

    quantiles = [i / int(n_token_range) for i in range(1, int(n_token_range))]
    return sorted({float(reference_numeric.quantile(q)) for q in quantiles})

def _discretize_continuous_series(
    series: pd.Series,
    reference_series: pd.Series,
    n_token_range: int,
) -> pd.Series:
    numeric_series = pd.to_numeric(series, errors="coerce")
    reference_numeric = pd.to_numeric(reference_series, errors="coerce").dropna()
    if reference_numeric.empty:
        return pd.Series(["__MISSING__"] * len(series), index=series.index, dtype=str)

    thresholds = _build_quantile_thresholds(reference_numeric, n_token_range)
    if not thresholds:
        filled = pd.Series(["bin_all"] * len(series), index=series.index, dtype=str)
        filled.loc[numeric_series.isna()] = "__MISSING__"
        return filled

    labels = _build_discretization_bin_labels(thresholds)
    binned = pd.cut(
        numeric_series,
        bins=[-np.inf, *thresholds, np.inf],
        labels=labels,
        include_lowest=True,
        duplicates="drop",
    )
    tokens = binned.astype(object).where(~numeric_series.isna(), "__MISSING__")
    return tokens.fillna("__MISSING__").astype(str)

def _prepare_attribute_inference_binary_features(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    continuous_columns: List[str],
    categorical_columns: List[str],
    n_token_range: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], pd.DataFrame]:
    transformed_real = pd.DataFrame(index=real_df.index)
    transformed_synthetic = pd.DataFrame(index=synthetic_df.index)

    if categorical_columns:
        normalized_real = normalize_categorical_dataframe(real_df[categorical_columns], categorical_columns)
        normalized_synthetic = normalize_categorical_dataframe(synthetic_df[categorical_columns], categorical_columns)
        for column in categorical_columns:
            transformed_real[column] = normalized_real[column].astype(str)
            transformed_synthetic[column] = normalized_synthetic[column].astype(str)

    for column in continuous_columns:
        transformed_real[column] = _discretize_continuous_series(real_df[column], real_df[column], n_token_range)
        transformed_synthetic[column] = _discretize_continuous_series(synthetic_df[column], real_df[column], n_token_range)

    ordered_columns = categorical_columns + continuous_columns
    transformed_real = transformed_real[ordered_columns]
    transformed_synthetic = transformed_synthetic[ordered_columns]

    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.fit(pd.concat([transformed_real, transformed_synthetic], axis=0, ignore_index=True))
    real_matrix = encoder.transform(transformed_real)
    synthetic_matrix = encoder.transform(transformed_synthetic)

    if hasattr(real_matrix, "toarray"):
        real_matrix = real_matrix.toarray()
    if hasattr(synthetic_matrix, "toarray"):
        synthetic_matrix = synthetic_matrix.toarray()

    real_matrix = np.asarray(real_matrix, dtype=int)
    synthetic_matrix = np.asarray(synthetic_matrix, dtype=int)
    encoded_feature_names = encoder.get_feature_names_out(ordered_columns).tolist()
    encoded_source_columns: List[str] = []
    for source_column, categories in zip(ordered_columns, encoder.categories_):
        encoded_source_columns.extend([source_column] * len(categories))
    encoded_feature_df = pd.DataFrame({
        "encoded_feature": encoded_feature_names,
        "source_column": encoded_source_columns,
    })
    return real_matrix, synthetic_matrix, encoded_feature_names, encoded_feature_df

class DistanceToClosestRecordEvaluator(PrivacyEvaluator):
    name = "dcr"

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        feature_columns, continuous_columns, categorical_columns = _resolve_privacy_feature_columns(context)
        if not feature_columns:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"status": "skipped", "reason": "无可用共同特征列"},
                tables={},
                metadata={"method": "dcr"},
                warnings=["无可用共同特征列，无法计算 DCR"],
            )

        real_df = context.comparison_real_data[feature_columns].copy().reset_index(drop=True)
        evaluation_real_df = context.evaluation_real_data[feature_columns].copy().reset_index(drop=True)
        synthetic_df = context.synthetic_data[feature_columns].copy().reset_index(drop=True)
        if real_df.empty or synthetic_df.empty:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"status": "skipped", "reason": "真实或合成数据为空"},
                tables={},
                metadata={"method": "dcr"},
                warnings=["真实或合成数据为空，无法计算 DCR"],
            )

        if categorical_columns:
            real_df = normalize_categorical_dataframe(real_df, categorical_columns)
            synthetic_df = normalize_categorical_dataframe(synthetic_df, categorical_columns)
            evaluation_real_df = normalize_categorical_dataframe(evaluation_real_df, categorical_columns)

        preprocessor = _build_preprocessor(continuous_columns, categorical_columns)
        real_matrix = preprocessor.fit_transform(real_df)
        synthetic_matrix = preprocessor.transform(synthetic_df)
        evaluation_real_matrix = preprocessor.transform(evaluation_real_df)
        if hasattr(real_matrix, "toarray"):
            real_matrix = real_matrix.toarray()
        if hasattr(synthetic_matrix, "toarray"):
            synthetic_matrix = synthetic_matrix.toarray()
        if hasattr(evaluation_real_matrix, "toarray"):
            evaluation_real_matrix = evaluation_real_matrix.toarray()

        real_matrix = np.asarray(real_matrix, dtype=float)
        synthetic_matrix = np.asarray(synthetic_matrix, dtype=float)
        evaluation_real_matrix = np.asarray(evaluation_real_matrix, dtype=float)

        distances, indices = _compute_nearest_neighbor_distances(synthetic_matrix, real_matrix)
        synthetic_details_df = pd.DataFrame({
            "comparison_type": "synthetic_to_real",
            "query_index": np.arange(len(synthetic_df), dtype=int),
            "synthetic_index": np.arange(len(synthetic_df), dtype=int),
            "holdout_real_index": pd.Series([pd.NA] * len(synthetic_df), dtype="Int64"),
            "nearest_real_index": indices,
            "dcr_distance": distances,
        })

        baseline_available = False
        baseline_distances = np.array([], dtype=float)
        baseline_indices = np.array([], dtype=int)
        baseline_details_df = pd.DataFrame()
        warnings: List[str] = []
        if not evaluation_real_df.empty and not evaluation_real_df.equals(real_df):
            baseline_distances, baseline_indices = _compute_nearest_neighbor_distances(evaluation_real_matrix, real_matrix)
            baseline_available = baseline_distances.size > 0
            if baseline_available:
                baseline_details_df = pd.DataFrame({
                    "comparison_type": "real_holdout_to_real_baseline",
                    "query_index": np.arange(len(evaluation_real_df), dtype=int),
                    "synthetic_index": pd.Series([pd.NA] * len(evaluation_real_df), dtype="Int64"),
                    "holdout_real_index": np.arange(len(evaluation_real_df), dtype=int),
                    "nearest_real_index": baseline_indices,
                    "dcr_distance": baseline_distances,
                })
        else:
            warnings.append("DCR 的 real-real baseline 未计算：evaluation_real_data 与 comparison_real_data 相同或不可区分。")

        details_df = pd.concat([synthetic_details_df, baseline_details_df], ignore_index=True)
        summary_row = {
            "status": "ok",
            "feature_count": int(len(feature_columns)),
            "continuous_feature_count": int(len(continuous_columns)),
            "categorical_feature_count": int(len(categorical_columns)),
            "real_reference_rows": int(len(real_df)),
            "synthetic_rows": int(len(synthetic_df)),
            "real_holdout_rows": int(len(evaluation_real_df)),
            "mean_dcr_distance": float(np.mean(distances)),
            "median_dcr_distance": float(np.median(distances)),
            "dcr_distance_percentile_5": float(np.percentile(distances, 5)),
            "baseline_available": bool(baseline_available),
            "mean_dcr_distance_real_real_baseline": float(np.mean(baseline_distances)) if baseline_available else np.nan,
            "median_dcr_distance_real_real_baseline": float(np.median(baseline_distances)) if baseline_available else np.nan,
            "dcr_distance_percentile_5_real_real_baseline": float(np.percentile(baseline_distances, 5)) if baseline_available else np.nan,
        }
        summary_df = pd.DataFrame([summary_row])

        return MetricResult(
            name=self.name,
            domain=self.domain,
            level=self.level,
            summary=summary_row,
            tables={
                "summary": summary_df,
                "details": details_df,
            },
            metadata={
                "method": "dcr",
                "feature_columns": feature_columns,
                "continuous_columns": continuous_columns,
                "categorical_columns": categorical_columns,
                "reference_real_data": "comparison_real_data",
                "baseline_real_data": "evaluation_real_data" if baseline_available else None,
            },
            warnings=warnings,
        )

class AttributeInferenceEvaluator(PrivacyEvaluator):
    name = "attribute_inference"

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        feature_columns, continuous_columns, categorical_columns = _resolve_privacy_feature_columns(context)
        if not feature_columns:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"status": "skipped", "reason": "无可用共同特征列"},
                tables={},
                metadata={"method": "attribute_inference"},
                warnings=["无可用共同特征列，无法计算 Attribute Inference"],
            )

        real_df = context.comparison_real_data[feature_columns].copy().reset_index(drop=True)
        synthetic_df = context.synthetic_data[feature_columns].copy().reset_index(drop=True)
        if real_df.empty or synthetic_df.empty:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"status": "skipped", "reason": "真实或合成数据为空"},
                tables={},
                metadata={"method": "attribute_inference"},
                warnings=["真实或合成数据为空，无法计算 Attribute Inference"],
            )

        experiment_configs = list(itertools.product(
            ATTRIBUTE_INFERENCE_N_ATT_LIST,
            ATTRIBUTE_INFERENCE_P_GIVEN_LIST,
            ATTRIBUTE_INFERENCE_K_LIST,
        ))
        if not experiment_configs:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"status": "skipped", "reason": "Attribute Inference 配置为空"},
                tables={},
                metadata={"method": "attribute_inference"},
                warnings=["Attribute Inference 配置为空，无法运行实验"],
            )

        real_matrix, synthetic_matrix, encoded_feature_names, encoded_feature_df = _prepare_attribute_inference_binary_features(
            real_df=real_df,
            synthetic_df=synthetic_df,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            n_token_range=int(ATTRIBUTE_INFERENCE_N_TOKEN_RANGE),
        )
        encoded_feature_count = int(real_matrix.shape[1])
        if encoded_feature_count < 2:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"status": "skipped", "reason": "编码后的二值特征数不足 2"},
                tables={},
                metadata={"method": "attribute_inference"},
                warnings=["编码后的二值特征数不足 2，无法同时划分已知和未知特征"],
            )

        summary_rows: List[Dict[str, Any]] = []
        detail_rows: List[Dict[str, Any]] = []
        feature_score_rows: List[Dict[str, Any]] = []
        synthetic_prior = synthetic_matrix.mean(axis=0)
        base_seed = int(context.random_state or 0)

        for experiment_id, (n_att, p_given, k_neighbors) in enumerate(experiment_configs, start=1):
            experiment_seed = base_seed + experiment_id
            rng = np.random.default_rng(experiment_seed)

            effective_n_att = int(min(max(int(n_att), 1), len(real_matrix)))
            effective_k = int(min(max(int(k_neighbors), 1), len(synthetic_matrix)))
            requested_known_count = int(round(float(p_given) * encoded_feature_count))
            known_feature_count = int(min(max(requested_known_count, 1), encoded_feature_count - 1))
            unknown_feature_count = int(encoded_feature_count - known_feature_count)

            attacked_indices = rng.choice(len(real_matrix), size=effective_n_att, replace=False)
            feature_true_values: List[List[int]] = [[] for _ in range(encoded_feature_count)]
            feature_pred_values: List[List[int]] = [[] for _ in range(encoded_feature_count)]

            for attacked_real_index in attacked_indices:
                known_indices = np.sort(rng.choice(encoded_feature_count, size=known_feature_count, replace=False))
                known_mask = np.zeros(encoded_feature_count, dtype=bool)
                known_mask[known_indices] = True
                unknown_mask = ~known_mask
                unknown_indices = np.flatnonzero(unknown_mask)

                attacked_vector = real_matrix[attacked_real_index]
                distance_vector = np.sqrt(np.sum(
                    np.square(synthetic_matrix[:, known_indices] - attacked_vector[known_indices]),
                    axis=1,
                ))
                neighbor_order = np.argsort(distance_vector, kind="stable")[:effective_k]
                neighbor_matrix = synthetic_matrix[neighbor_order][:, unknown_indices]
                vote_fraction = neighbor_matrix.mean(axis=0)
                tie_mask = np.isclose(vote_fraction, 0.5)
                predicted_unknown = (vote_fraction > 0.5).astype(int)
                if tie_mask.any():
                    predicted_unknown[tie_mask] = (synthetic_prior[unknown_indices][tie_mask] >= 0.5).astype(int)

                true_unknown = attacked_vector[unknown_indices].astype(int)
                for unknown_position, feature_index in enumerate(unknown_indices):
                    feature_true_values[int(feature_index)].append(int(true_unknown[unknown_position]))
                    feature_pred_values[int(feature_index)].append(int(predicted_unknown[unknown_position]))

                detail_rows.append({
                    "experiment_id": experiment_id,
                    "n_att": int(n_att),
                    "p_given": float(p_given),
                    "k_neighbors": int(k_neighbors),
                    "real_attack_index": int(attacked_real_index),
                    "known_feature_count": known_feature_count,
                    "unknown_feature_count": unknown_feature_count,
                    "known_feature_ratio_actual": float(known_feature_count / encoded_feature_count),
                    "effective_k_neighbors": effective_k,
                    "nearest_synthetic_indices": "|".join(str(int(idx)) for idx in neighbor_order.tolist()),
                    "nearest_synthetic_distances": "|".join(f"{float(distance_vector[idx]):.6f}" for idx in neighbor_order.tolist()),
                    "true_unknown_positive_count": int(true_unknown.sum()),
                    "predicted_unknown_positive_count": int(predicted_unknown.sum()),
                })

            per_feature_f1_scores: List[float] = []
            total_unknown_binary_targets = 0
            total_true_unknown_positive_count = 0
            total_predicted_unknown_positive_count = 0
            for feature_index, (true_values, pred_values) in enumerate(zip(feature_true_values, feature_pred_values)):
                if not true_values:
                    continue

                true_array = np.asarray(true_values, dtype=int)
                pred_array = np.asarray(pred_values, dtype=int)
                feature_f1 = float(f1_score(true_array, pred_array, zero_division=0))
                per_feature_f1_scores.append(feature_f1)
                total_unknown_binary_targets += int(true_array.size)
                total_true_unknown_positive_count += int(true_array.sum())
                total_predicted_unknown_positive_count += int(pred_array.sum())

                feature_score_rows.append({
                    "experiment_id": experiment_id,
                    "n_att": int(n_att),
                    "p_given": float(p_given),
                    "k_neighbors": int(k_neighbors),
                    "n_token_range": int(ATTRIBUTE_INFERENCE_N_TOKEN_RANGE),
                    "encoded_feature": encoded_feature_names[feature_index],
                    "source_column": encoded_feature_df.iloc[feature_index]["source_column"],
                    "evaluated_attack_rows": int(true_array.size),
                    "true_positive_count": int(true_array.sum()),
                    "predicted_positive_count": int(pred_array.sum()),
                    "f1_score": feature_f1,
                })

            mean_feature_f1_score = float(np.mean(per_feature_f1_scores)) if per_feature_f1_scores else np.nan
            feature_f1_std = float(np.std(per_feature_f1_scores)) if per_feature_f1_scores else np.nan

            summary_rows.append({
                "status": "ok",
                "experiment_id": experiment_id,
                "n_att": int(n_att),
                "effective_n_att": effective_n_att,
                "p_given": float(p_given),
                "k_neighbors": int(k_neighbors),
                "n_token_range": int(ATTRIBUTE_INFERENCE_N_TOKEN_RANGE),
                "effective_k_neighbors": effective_k,
                "encoded_feature_count": encoded_feature_count,
                "evaluated_encoded_feature_count": int(len(per_feature_f1_scores)),
                "known_feature_count": known_feature_count,
                "unknown_feature_count": unknown_feature_count,
                "known_feature_ratio_actual": float(known_feature_count / encoded_feature_count),
                "original_feature_count": int(len(feature_columns)),
                "continuous_feature_count": int(len(continuous_columns)),
                "categorical_feature_count": int(len(categorical_columns)),
                "real_reference_rows": int(len(real_df)),
                "synthetic_rows": int(len(synthetic_df)),
                "f1_score": mean_feature_f1_score,
                "feature_f1_std": feature_f1_std,
                "total_unknown_binary_targets": total_unknown_binary_targets,
                "true_unknown_positive_count": total_true_unknown_positive_count,
                "predicted_unknown_positive_count": total_predicted_unknown_positive_count,
            })

        summary_df = pd.DataFrame(summary_rows)
        details_df = pd.DataFrame(detail_rows)
        feature_scores_df = pd.DataFrame(feature_score_rows)
        summary_row = summary_rows[0] if len(summary_rows) == 1 else {
            "status": "ok",
            "experiment_count": int(len(summary_rows)),
            "n_token_range": int(ATTRIBUTE_INFERENCE_N_TOKEN_RANGE),
            "encoded_feature_count": encoded_feature_count,
            "original_feature_count": int(len(feature_columns)),
            "continuous_feature_count": int(len(continuous_columns)),
            "categorical_feature_count": int(len(categorical_columns)),
            "lowest_f1_score": float(summary_df["f1_score"].min()) if not summary_df.empty else np.nan,
            "mean_f1_score": float(summary_df["f1_score"].mean()) if not summary_df.empty else np.nan,
            "reference_real_data": "comparison_real_data",
        }

        return MetricResult(
            name=self.name,
            domain=self.domain,
            level=self.level,
            summary=summary_row,
            tables={
                "summary": summary_df,
                "details": details_df,
                "feature_scores": feature_scores_df,
                "encoded_features": encoded_feature_df,
            },
            metadata={
                "method": "attribute_inference",
                "feature_columns": feature_columns,
                "continuous_columns": continuous_columns,
                "categorical_columns": categorical_columns,
                "encoded_feature_names": encoded_feature_names,
                "n_token_range": int(ATTRIBUTE_INFERENCE_N_TOKEN_RANGE),
                "reference_real_data": "comparison_real_data",
                "experiment_configs": [
                    {
                        "n_att": int(n_att),
                        "p_given": float(p_given),
                        "k_neighbors": int(k_neighbors),
                        "n_token_range": int(ATTRIBUTE_INFERENCE_N_TOKEN_RANGE),
                    }
                    for n_att, p_given, k_neighbors in experiment_configs
                ],
            },
            warnings=[],
        )
