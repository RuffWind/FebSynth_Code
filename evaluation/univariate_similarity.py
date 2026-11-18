from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from scipy import stats
from output_localization import OutputLocalizer

from .base import UnivariateSimilarityEvaluator
from .context import EvaluationContext
from .results import MetricResult
from .utils import (
    calculate_tvd_and_js,
    categorical_frequency_summary,
    configure_matplotlib_for_chinese,
    REAL_DATA_COLOR,
    safe_relative_error,
    series_to_categorical_tokens,
    split_real_reference_data,
    SYNTHETIC_DATA_COLOR,
)

PRIMARY_COLOR = REAL_DATA_COLOR
SECONDARY_COLOR = SYNTHETIC_DATA_COLOR

def _single_row_summary_df(values: Dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame([values])

def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", str(value))
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "unnamed"

def _build_density_curve(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    numeric_values = np.asarray(values, dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    if numeric_values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    min_value = float(np.min(numeric_values))
    max_value = float(np.max(numeric_values))
    if np.isclose(min_value, max_value):
        center = min_value
        scale = max(abs(center) * 0.05, 1e-3)
        x_grid = np.linspace(center - 4 * scale, center + 4 * scale, 200)
        density = np.exp(-0.5 * ((x_grid - center) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
        return x_grid, density

    padding = max((max_value - min_value) * 0.08, 1e-3)
    x_grid = np.linspace(min_value - padding, max_value + padding, 300)
    if numeric_values.size >= 2 and np.unique(numeric_values).size >= 2:
        density = stats.gaussian_kde(numeric_values)(x_grid)
    else:
        center = float(numeric_values[0])
        scale = max(abs(center) * 0.05, 1e-3)
        density = np.exp(-0.5 * ((x_grid - center) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
    return x_grid, np.asarray(density, dtype=float)

def _trim_values_by_median_iqr(values: np.ndarray, multiplier: float = 3.0) -> np.ndarray:
    numeric_values = np.asarray(values, dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    if numeric_values.size == 0:
        return numeric_values

    median_value = float(np.median(numeric_values))
    q1 = float(np.percentile(numeric_values, 25))
    q3 = float(np.percentile(numeric_values, 75))
    iqr = float(q3 - q1)
    if iqr <= 0:
        return numeric_values

    lower_bound = median_value - multiplier * iqr
    upper_bound = median_value + multiplier * iqr
    trimmed = numeric_values[(numeric_values >= lower_bound) & (numeric_values <= upper_bound)]
    return trimmed if trimmed.size > 0 else numeric_values

def save_continuous_univariate_distribution_plots(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    columns: List[str],
    output_dir: str,
    localizer: OutputLocalizer | None = None,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    localizer = localizer or OutputLocalizer()
    localizer.configure_matplotlib()
    os.makedirs(output_dir, exist_ok=True)
    plot_paths: List[str] = []
    for column in columns:
        if column not in real_df.columns or column not in synthetic_df.columns:
            continue
        real_values = pd.to_numeric(real_df[column], errors="coerce").dropna().to_numpy(dtype=float)
        synthetic_values = pd.to_numeric(synthetic_df[column], errors="coerce").dropna().to_numpy(dtype=float)
        if real_values.size < 2 or synthetic_values.size < 2:
            continue

        real_plot_values = _trim_values_by_median_iqr(real_values)
        synthetic_plot_values = _trim_values_by_median_iqr(synthetic_values)
        if real_plot_values.size < 2 or synthetic_plot_values.size < 2:
            continue

        real_x, real_density = _build_density_curve(real_plot_values)
        synthetic_x, synthetic_density = _build_density_curve(synthetic_plot_values)
        if real_x.size == 0 or synthetic_x.size == 0:
            continue

        line_width = 1.3
        real_label = (
            f"Real (n={real_plot_values.size})"
            if localizer.is_english else f"真实数据 (n={real_plot_values.size})"
        )
        synthetic_label = (
            f"Synthetic (n={synthetic_plot_values.size})"
            if localizer.is_english else f"合成数据 (n={synthetic_plot_values.size})"
        )
        x_label = localizer.translate_text(column)
        y_label = "Density" if localizer.is_english else "平滑密度"
        title = (
            f"Distribution of {x_label}"
            if localizer.is_english else f"{column}：真实数据 vs 合成数据分布"
        )

        fig, ax = plt.subplots(figsize=(8.2, 5.2) if localizer.is_english else (8.5, 5.5))
        ax.plot(real_x, real_density, color=PRIMARY_COLOR, linewidth=line_width, label=real_label)
        ax.fill_between(real_x, real_density, color=PRIMARY_COLOR, alpha=0.12)
        ax.plot(
            synthetic_x,
            synthetic_density,
            color=SECONDARY_COLOR,
            linewidth=line_width,
            label=synthetic_label,
        )
        ax.fill_between(synthetic_x, synthetic_density, color=SECONDARY_COLOR, alpha=0.12)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()

        file_stem = localizer.translate_text(column)
        output_path = Path(output_dir) / f"{localizer.sanitize_filename(file_stem)}_distribution.png"
        fig.savefig(output_path, dpi=300 if localizer.is_english else 200, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(output_path))
    return plot_paths

def save_categorical_univariate_distribution_plots(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    columns: List[str],
    output_dir: str,
    localizer: OutputLocalizer | None = None,
) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    localizer = localizer or OutputLocalizer()
    localizer.configure_matplotlib()
    os.makedirs(output_dir, exist_ok=True)
    plot_paths: List[str] = []
    for column in columns:
        if column not in real_df.columns or column not in synthetic_df.columns:
            continue

        real_tokens = series_to_categorical_tokens(real_df[column])
        synthetic_tokens = series_to_categorical_tokens(synthetic_df[column])
        categories = sorted(set(real_tokens.unique()) | set(synthetic_tokens.unique()))
        if not categories:
            continue

        real_freq = real_tokens.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        synthetic_freq = synthetic_tokens.value_counts(normalize=True).reindex(categories, fill_value=0.0)
        x_positions = np.arange(len(categories), dtype=float)
        bar_width = 0.42
        figure_width = min(max(8.5, len(categories) * 0.45), 18.0)

        x_label = localizer.translate_text(column)
        tick_labels = [localizer.translate_text(str(category)) for category in categories]
        real_label = f"Real (n={len(real_tokens)})" if localizer.is_english else f"真实数据 (n={len(real_tokens)})"
        synthetic_label = (
            f"Synthetic (n={len(synthetic_tokens)})"
            if localizer.is_english else f"合成数据 (n={len(synthetic_tokens)})"
        )
        y_label = "Frequency" if localizer.is_english else "类别频率"
        title = f"Distribution of {x_label}" if localizer.is_english else f"{column}：真实数据 vs 合成数据分布"

        fig, ax = plt.subplots(figsize=((min(max(7.0, len(categories) * 0.6), 14.0), 5.2) if localizer.is_english else (figure_width, 5.8)))
        ax.bar(
            x_positions - bar_width / 2,
            real_freq.to_numpy(dtype=float),
            width=bar_width,
            color=PRIMARY_COLOR,
            alpha=0.8,
            label=real_label,
        )
        ax.bar(
            x_positions + bar_width / 2,
            synthetic_freq.to_numpy(dtype=float),
            width=bar_width,
            color=SECONDARY_COLOR,
            alpha=0.8,
            label=synthetic_label,
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()

        file_stem = localizer.translate_text(column)
        output_path = Path(output_dir) / f"{localizer.sanitize_filename(file_stem)}_distribution.png"
        fig.savefig(output_path, dpi=300 if localizer.is_english else 200, bbox_inches="tight")
        plt.close(fig)
        plot_paths.append(str(output_path))
    return plot_paths

class ContinuousUnivariateSimilarityEvaluator(UnivariateSimilarityEvaluator):
    name = "continuous_univariate_similarity"

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        rows: List[Dict[str, Any]] = []
        details: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []

        baseline_real_a, baseline_real_b = split_real_reference_data(
            context.comparison_real_data,
            context.random_state,
        )

        for column in context.valid_continuous_columns:
            real_values = pd.to_numeric(context.comparison_real_data[column], errors="coerce").dropna()
            synthetic_values = pd.to_numeric(context.synthetic_data[column], errors="coerce").dropna()
            baseline_values_a = pd.to_numeric(baseline_real_a[column], errors="coerce").dropna()
            baseline_values_b = pd.to_numeric(baseline_real_b[column], errors="coerce").dropna()

            row: Dict[str, Any] = {
                "column": column,
                "status": "ok",
                "real_count": int(len(real_values)),
                "synthetic_count": int(len(synthetic_values)),
                "baseline_half_a_count": int(len(baseline_values_a)),
                "baseline_half_b_count": int(len(baseline_values_b)),
            }

            if len(real_values) < 2 or len(synthetic_values) < 2:
                row["status"] = "skipped"
                row["reason"] = "真实或合成样本的非空数值数量不足"
                warnings.append(f"连续变量 '{column}' 因有效样本不足被跳过")
                rows.append(row)
                details[column] = {
                    "real_summary": {},
                    "synthetic_summary": {},
                    "baseline_summary": {},
                }
                continue

            real_mean = float(real_values.mean())
            synthetic_mean = float(synthetic_values.mean())
            real_std = float(real_values.std())
            synthetic_std = float(synthetic_values.std())
            real_median = float(real_values.median())
            synthetic_median = float(synthetic_values.median())

            ks_statistic = float(stats.ks_2samp(real_values, synthetic_values).statistic)
            wasserstein_distance = float(stats.wasserstein_distance(real_values, synthetic_values))

            baseline_ks_statistic = np.nan
            baseline_wasserstein_distance = np.nan
            if len(baseline_values_a) >= 2 and len(baseline_values_b) >= 2:
                baseline_ks_statistic = float(stats.ks_2samp(baseline_values_a, baseline_values_b).statistic)
                baseline_wasserstein_distance = float(
                    stats.wasserstein_distance(baseline_values_a, baseline_values_b)
                )
            else:
                warnings.append(f"连续变量 '{column}' 的固有距离基准样本不足，已记为 NaN")

            normalized_wasserstein_distance = np.nan
            normalized_intrinsic_wasserstein_distance = np.nan
            if np.isfinite(real_std) and real_std > 0:
                normalized_wasserstein_distance = float(wasserstein_distance / real_std)
                if np.isfinite(baseline_wasserstein_distance):
                    normalized_intrinsic_wasserstein_distance = float(
                        baseline_wasserstein_distance / real_std
                    )
            else:
                warnings.append(
                    f"连续变量 '{column}' 的真实标准差无法用于 Wasserstein 标准化，相关汇总将跳过该列"
                )

            row.update({
                "real_mean": real_mean,
                "synthetic_mean": synthetic_mean,
                "mean_relative_error": safe_relative_error(real_mean, synthetic_mean),
                "real_std": real_std,
                "synthetic_std": synthetic_std,
                "std_relative_error": safe_relative_error(real_std, synthetic_std),
                "real_median": real_median,
                "synthetic_median": synthetic_median,
                "median_relative_error": safe_relative_error(real_median, synthetic_median),
                "ks_statistic": ks_statistic,
                "intrinsic_ks_statistic": baseline_ks_statistic,
                "wasserstein_distance": wasserstein_distance,
                "intrinsic_wasserstein_distance": baseline_wasserstein_distance,
                "normalized_wasserstein_distance": normalized_wasserstein_distance,
                "normalized_intrinsic_wasserstein_distance": normalized_intrinsic_wasserstein_distance,
            })
            rows.append(row)

            details[column] = {
                "real_summary": {
                    "count": int(len(real_values)),
                    "mean": real_mean,
                    "std": real_std,
                    "median": real_median,
                },
                "synthetic_summary": {
                    "count": int(len(synthetic_values)),
                    "mean": synthetic_mean,
                    "std": synthetic_std,
                    "median": synthetic_median,
                },
                "baseline_summary": {
                    "half_a_count": int(len(baseline_values_a)),
                    "half_b_count": int(len(baseline_values_b)),
                    "ks_statistic": baseline_ks_statistic,
                    "wasserstein_distance": baseline_wasserstein_distance,
                    "normalized_wasserstein_distance": normalized_intrinsic_wasserstein_distance,
                },
            }

        metrics_df = pd.DataFrame(rows)
        valid_metrics_df = (
            metrics_df[metrics_df["status"] == "ok"].copy()
            if "status" in metrics_df.columns
            else metrics_df.copy()
        )
        summary = {
            "evaluated_columns": int((metrics_df.get("status", pd.Series(dtype=object)) == "ok").sum()),
            "skipped_columns": int((metrics_df.get("status", pd.Series(dtype=object)) == "skipped").sum()),
        }
        normalized_wasserstein_mean = (
            float(valid_metrics_df["normalized_wasserstein_distance"].dropna().mean())
            if not valid_metrics_df.empty and "normalized_wasserstein_distance" in valid_metrics_df.columns
            and not valid_metrics_df["normalized_wasserstein_distance"].dropna().empty
            else np.nan
        )
        normalized_intrinsic_wasserstein_mean = (
            float(valid_metrics_df["normalized_intrinsic_wasserstein_distance"].dropna().mean())
            if not valid_metrics_df.empty and "normalized_intrinsic_wasserstein_distance" in valid_metrics_df.columns
            and not valid_metrics_df["normalized_intrinsic_wasserstein_distance"].dropna().empty
            else np.nan
        )

        summary_df = _single_row_summary_df({
            "evaluated_columns": int(len(valid_metrics_df)),
            "real_mean_mean": float(valid_metrics_df["real_mean"].mean()) if not valid_metrics_df.empty else np.nan,
            "synthetic_mean_mean": float(valid_metrics_df["synthetic_mean"].mean()) if not valid_metrics_df.empty else np.nan,
            "mean_mean_relative_error": (
                safe_relative_error(
                    float(valid_metrics_df["real_mean"].mean()),
                    float(valid_metrics_df["synthetic_mean"].mean()),
                )
                if not valid_metrics_df.empty else np.nan
            ),
            "real_std_mean": float(valid_metrics_df["real_std"].mean()) if not valid_metrics_df.empty else np.nan,
            "synthetic_std_mean": float(valid_metrics_df["synthetic_std"].mean()) if not valid_metrics_df.empty else np.nan,
            "std_mean_relative_error": (
                safe_relative_error(
                    float(valid_metrics_df["real_std"].mean()),
                    float(valid_metrics_df["synthetic_std"].mean()),
                )
                if not valid_metrics_df.empty else np.nan
            ),
            "real_median_mean": float(valid_metrics_df["real_median"].mean()) if not valid_metrics_df.empty else np.nan,
            "synthetic_median_mean": float(valid_metrics_df["synthetic_median"].mean()) if not valid_metrics_df.empty else np.nan,
            "median_mean_relative_error": (
                safe_relative_error(
                    float(valid_metrics_df["real_median"].mean()),
                    float(valid_metrics_df["synthetic_median"].mean()),
                )
                if not valid_metrics_df.empty else np.nan
            ),
            "ks_statistic_mean": float(valid_metrics_df["ks_statistic"].mean()) if not valid_metrics_df.empty else np.nan,
            "intrinsic_ks_statistic_mean": float(valid_metrics_df["intrinsic_ks_statistic"].mean()) if not valid_metrics_df.empty else np.nan,
            "ks_mean_relative_error": (
                safe_relative_error(
                    float(valid_metrics_df["intrinsic_ks_statistic"].mean()),
                    float(valid_metrics_df["ks_statistic"].mean()),
                )
                if not valid_metrics_df.empty else np.nan
            ),
            "wasserstein_distance_mean": normalized_wasserstein_mean,
            "intrinsic_wasserstein_distance_mean": normalized_intrinsic_wasserstein_mean,
            "wasserstein_mean_relative_error": (
                safe_relative_error(
                    normalized_intrinsic_wasserstein_mean,
                    normalized_wasserstein_mean,
                )
                if np.isfinite(normalized_wasserstein_mean) and np.isfinite(normalized_intrinsic_wasserstein_mean)
                else np.nan
            ),
        })
        return MetricResult(
            name=self.name,
            domain=self.domain,
            level=self.level,
            summary=summary,
            tables={"metrics": metrics_df, "summary": summary_df},
            metadata={"variable_type": "continuous", "details": details},
            warnings=warnings,
        )

class CategoricalUnivariateSimilarityEvaluator(UnivariateSimilarityEvaluator):
    name = "categorical_univariate_similarity"

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        rows: List[Dict[str, Any]] = []
        frequency_detail_rows: List[Dict[str, Any]] = []
        details: Dict[str, Dict[str, Any]] = {}
        warnings: List[str] = []

        baseline_real_a, baseline_real_b = split_real_reference_data(
            context.comparison_real_data,
            context.random_state,
        )

        valid_categorical = list(context.valid_categorical_columns)
        ordered_columns = valid_categorical + [
            column for column in context.schema.outcome_columns
            if column not in valid_categorical
            and column in context.comparison_real_data.columns
            and column in context.synthetic_data.columns
        ]

        for column in ordered_columns:
            real_tokens = series_to_categorical_tokens(context.comparison_real_data[column])
            synthetic_tokens = series_to_categorical_tokens(context.synthetic_data[column])
            baseline_tokens_a = series_to_categorical_tokens(baseline_real_a[column])
            baseline_tokens_b = series_to_categorical_tokens(baseline_real_b[column])

            categories = sorted(set(real_tokens.unique()) | set(synthetic_tokens.unique()))
            baseline_categories = sorted(set(baseline_tokens_a.unique()) | set(baseline_tokens_b.unique()))

            real_counts = real_tokens.value_counts().reindex(categories, fill_value=0)
            synthetic_counts = synthetic_tokens.value_counts().reindex(categories, fill_value=0)
            real_freq = real_counts / max(real_counts.sum(), 1)
            synthetic_freq = synthetic_counts / max(synthetic_counts.sum(), 1)
            tvd, js_distance = calculate_tvd_and_js(real_freq, synthetic_freq)

            baseline_counts_a = baseline_tokens_a.value_counts().reindex(baseline_categories, fill_value=0)
            baseline_counts_b = baseline_tokens_b.value_counts().reindex(baseline_categories, fill_value=0)
            baseline_freq_a = baseline_counts_a / max(baseline_counts_a.sum(), 1)
            baseline_freq_b = baseline_counts_b / max(baseline_counts_b.sum(), 1)
            intrinsic_tvd, intrinsic_js_distance = calculate_tvd_and_js(baseline_freq_a, baseline_freq_b)

            real_category_set = set(real_tokens.unique())
            synthetic_category_set = set(synthetic_tokens.unique())
            missing_categories = sorted(real_category_set - synthetic_category_set)
            extra_categories = sorted(synthetic_category_set - real_category_set)
            coverage = float(len(real_category_set & synthetic_category_set) / max(len(real_category_set), 1))

            row = {
                "column": column,
                "status": "ok",
                "real_count": int(len(real_tokens)),
                "synthetic_count": int(len(synthetic_tokens)),
                "real_category_count": int(len(real_category_set)),
                "synthetic_category_count": int(len(synthetic_category_set)),
                "coverage": coverage,
                "missing_category_count": int(len(missing_categories)),
                "extra_category_count": int(len(extra_categories)),
                "tvd": tvd,
                "intrinsic_tvd": intrinsic_tvd,
                "js_distance": js_distance,
                "intrinsic_js_distance": intrinsic_js_distance,
                "max_category_probability_gap": float(np.max(np.abs(real_freq - synthetic_freq))) if len(categories) > 0 else np.nan,
                "is_outcome_column": bool(column in context.schema.outcome_columns),
            }
            rows.append(row)

            signed_gaps = synthetic_freq - real_freq
            for category in categories:
                frequency_detail_rows.append({
                    "column": column,
                    "category": str(category),
                    "real_frequency": float(real_freq.get(category, 0.0)),
                    "synthetic_frequency": float(synthetic_freq.get(category, 0.0)),
                    "signed_frequency_gap": float(signed_gaps.get(category, 0.0)),
                    "absolute_frequency_gap": float(abs(signed_gaps.get(category, 0.0))),
                    "is_outcome_column": bool(column in context.schema.outcome_columns),
                })

            details[column] = {
                "real_distribution": categorical_frequency_summary(real_tokens),
                "synthetic_distribution": categorical_frequency_summary(synthetic_tokens),
                "baseline_distribution_half_a": categorical_frequency_summary(baseline_tokens_a),
                "baseline_distribution_half_b": categorical_frequency_summary(baseline_tokens_b),
                "missing_categories": missing_categories,
                "extra_categories": extra_categories,
            }

        metrics_df = pd.DataFrame(rows)
        frequency_details_df = pd.DataFrame(frequency_detail_rows)
        valid_metrics_df = (
            metrics_df[metrics_df["status"] == "ok"].copy()
            if "status" in metrics_df.columns
            else metrics_df.copy()
        )
        summary = {
            "evaluated_columns": int((metrics_df.get("status", pd.Series(dtype=object)) == "ok").sum()),
            "outcome_columns_evaluated": int(metrics_df.get("is_outcome_column", pd.Series(dtype=bool)).sum()) if not metrics_df.empty else 0,
        }
        summary_df = _single_row_summary_df({
            "evaluated_columns": int(len(valid_metrics_df)),
            "outcome_columns_evaluated": int(valid_metrics_df["is_outcome_column"].sum()) if not valid_metrics_df.empty else 0,
            "real_category_count_mean": float(valid_metrics_df["real_category_count"].mean()) if not valid_metrics_df.empty else np.nan,
            "synthetic_category_count_mean": float(valid_metrics_df["synthetic_category_count"].mean()) if not valid_metrics_df.empty else np.nan,
            "category_count_mean_relative_error": (
                safe_relative_error(
                    float(valid_metrics_df["real_category_count"].mean()),
                    float(valid_metrics_df["synthetic_category_count"].mean()),
                )
                if not valid_metrics_df.empty else np.nan
            ),
            "coverage_mean": float(valid_metrics_df["coverage"].mean()) if not valid_metrics_df.empty else np.nan,
            "missing_category_count_mean": float(valid_metrics_df["missing_category_count"].mean()) if not valid_metrics_df.empty else np.nan,
            "extra_category_count_mean": float(valid_metrics_df["extra_category_count"].mean()) if not valid_metrics_df.empty else np.nan,
            "tvd_mean": float(valid_metrics_df["tvd"].mean()) if not valid_metrics_df.empty else np.nan,
            "intrinsic_tvd_mean": float(valid_metrics_df["intrinsic_tvd"].mean()) if not valid_metrics_df.empty else np.nan,
            "tvd_mean_relative_error": (
                safe_relative_error(
                    float(valid_metrics_df["intrinsic_tvd"].mean()),
                    float(valid_metrics_df["tvd"].mean()),
                )
                if not valid_metrics_df.empty else np.nan
            ),
            "js_distance_mean": float(valid_metrics_df["js_distance"].mean()) if not valid_metrics_df.empty else np.nan,
            "intrinsic_js_distance_mean": float(valid_metrics_df["intrinsic_js_distance"].mean()) if not valid_metrics_df.empty else np.nan,
            "js_mean_relative_error": (
                safe_relative_error(
                    float(valid_metrics_df["intrinsic_js_distance"].mean()),
                    float(valid_metrics_df["js_distance"].mean()),
                )
                if not valid_metrics_df.empty else np.nan
            ),
            "max_category_probability_gap_mean": float(valid_metrics_df["max_category_probability_gap"].mean()) if not valid_metrics_df.empty else np.nan,
        })
        return MetricResult(
            name=self.name,
            domain=self.domain,
            level=self.level,
            summary=summary,
            tables={
                "metrics": metrics_df,
                "summary": summary_df,
                "frequency_details": frequency_details_df,
            },
            metadata={"variable_type": "categorical", "details": details},
            warnings=warnings,
        )
