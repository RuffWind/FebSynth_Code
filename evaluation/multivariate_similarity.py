from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from output_localization import OutputLocalizer

from .base import MultivariateSimilarityEvaluator
from .context import EvaluationContext
from .eval_config import (
    CCD_GROUPING_CATEGORICAL_COLUMNS,
    CCD_GROUPING_CONTINUOUS_COLUMNS,
    CVT_DIFFERENCE_COLUMN_PAIRS,
    CVT_RANGE_HISTOGRAM_COLUMNS,
)
from .results import MetricResult
from .utils import (
    configure_matplotlib_for_chinese,
    cramers_v,
    correlation_ratio,
    normalize_categorical_value,
    REAL_DATA_COLOR,
    REFERENCE_LINE_COLOR,
    series_to_categorical_tokens,
    split_real_reference_data,
    SYNTHETIC_DATA_COLOR,
)

PRIMARY_COLOR = REAL_DATA_COLOR
SECONDARY_COLOR = SYNTHETIC_DATA_COLOR
ACCENT_COLOR = REFERENCE_LINE_COLOR

def _upper_triangle_values(matrix: pd.DataFrame) -> np.ndarray:
    if matrix.empty or len(matrix) < 2:
        return np.array([], dtype=float)
    values = matrix.to_numpy(dtype=float)
    upper_indices = np.triu_indices_from(values, k=1)
    return values[upper_indices]

def _pairwise_absolute_diff_summary(diff_matrix: pd.DataFrame) -> Dict[str, float]:
    upper_values = _upper_triangle_values(diff_matrix)
    valid_values = upper_values[~np.isnan(upper_values)]
    if len(valid_values) == 0:
        return {
            "pair_count": 0,
            "frobenius_norm": np.nan,
            "mae": np.nan,
            "mse": np.nan,
        }

    return {
        "pair_count": int(len(valid_values)),
        "frobenius_norm": float(np.linalg.norm(valid_values, ord=2)),
        "mae": float(np.mean(np.abs(valid_values))),
        "mse": float(np.mean(valid_values ** 2)),
    }

def _diagonal_distance(x_value: float, y_value: float) -> float:
    return float(abs(y_value - x_value) / np.sqrt(2.0))

def _is_valid_standardization_scale(scale: float) -> bool:
    return pd.notna(scale) and np.isfinite(scale) and float(scale) > 0.0

def _sanitize_filename(value: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]+', "_", value)
    sanitized = re.sub(r"\s+", "_", sanitized)
    sanitized = sanitized.strip("._")
    return sanitized or "unnamed"

def _filter_distance_outliers(values: np.ndarray) -> np.ndarray:
    numeric_values = np.asarray(values, dtype=float)
    numeric_values = numeric_values[np.isfinite(numeric_values)]
    if numeric_values.size == 0:
        return numeric_values

    median_value = float(np.median(numeric_values))
    q1 = float(np.percentile(numeric_values, 25))
    q3 = float(np.percentile(numeric_values, 75))
    iqr = float(q3 - q1)
    deviation = np.abs(numeric_values - median_value)
    if iqr <= 0:
        return numeric_values[np.isclose(deviation, 0.0)]
    return numeric_values[deviation <= (3.0 * iqr)]

def _build_smooth_density_curve(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    filtered_values = _filter_distance_outliers(values)
    if filtered_values.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    min_value = float(np.min(filtered_values))
    max_value = float(np.max(filtered_values))
    if np.isclose(min_value, max_value):
        center = min_value
        scale = max(abs(center) * 0.05, 1e-3)
        x_grid = np.linspace(center - 4 * scale, center + 4 * scale, 200)
        density = np.exp(-0.5 * ((x_grid - center) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
        return x_grid, density

    padding = max((max_value - min_value) * 0.08, 1e-3)
    x_grid = np.linspace(min_value - padding, max_value + padding, 300)
    if filtered_values.size >= 2 and np.unique(filtered_values).size >= 2:
        density = gaussian_kde(filtered_values)(x_grid)
    else:
        center = float(filtered_values[0])
        scale = max(abs(center) * 0.05, 1e-3)
        density = np.exp(-0.5 * ((x_grid - center) / scale) ** 2) / (scale * np.sqrt(2 * np.pi))
    return x_grid, np.asarray(density, dtype=float)

def _plot_distance_distribution_comparison(
    real_vs_synthetic_distances: np.ndarray,
    real_vs_real_reference_distances: np.ndarray,
    title: str,
    file_path: Path,
    localizer: OutputLocalizer,
    x_axis_label: str | None = None,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    localizer.configure_matplotlib()
    rs_filtered = _filter_distance_outliers(real_vs_synthetic_distances)
    rr_filtered = _filter_distance_outliers(real_vs_real_reference_distances)
    if rs_filtered.size == 0 and rr_filtered.size == 0:
        return False

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    has_curve = False
    curve_specs = [
        (
            "Real vs Synthetic" if localizer.is_english else "真实-合成",
            np.asarray(real_vs_synthetic_distances, dtype=float),
            rs_filtered,
            PRIMARY_COLOR,
        ),
        (
            "Real vs Real reference" if localizer.is_english else "真实-真实参考",
            np.asarray(real_vs_real_reference_distances, dtype=float),
            rr_filtered,
            SECONDARY_COLOR,
        ),
    ]
    for label, raw_values, filtered_values, color in curve_specs:
        x_grid, density = _build_smooth_density_curve(filtered_values)
        if x_grid.size == 0 or density.size == 0:
            continue
        has_curve = True
        ax.plot(
            x_grid,
            density,
            linewidth=1.3,
            color=color,
            label=label,
        )
        ax.fill_between(x_grid, density, color=color, alpha=0.15)

    if not has_curve:
        plt.close(fig)
        return False

    ax.set_xlabel(
        x_axis_label
        if x_axis_label is not None
        else ("Distance to diagonal" if localizer.is_english else "点到对角线距离")
    )
    ax.set_ylabel("Density" if localizer.is_english else "平滑密度")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(file_path, dpi=300 if localizer.is_english else 200, bbox_inches="tight")
    plt.close(fig)
    return True

def save_pwcc_heatmap_triptych(
    real_matrix: pd.DataFrame,
    synthetic_matrix: pd.DataFrame,
    diff_matrix: pd.DataFrame,
    output_dir: str,
    localizer: OutputLocalizer | None = None,
) -> List[str]:
    if real_matrix.empty or synthetic_matrix.empty or diff_matrix.empty:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    localizer = localizer or OutputLocalizer()
    localizer.configure_matplotlib()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    translated_real = real_matrix.copy()
    translated_synthetic = synthetic_matrix.copy()
    translated_diff = diff_matrix.copy()
    translated_columns = [localizer.translate_text(str(column)) for column in real_matrix.columns]
    translated_index = [localizer.translate_text(str(index)) for index in real_matrix.index]
    for matrix_df in (translated_real, translated_synthetic, translated_diff):
        matrix_df.columns = translated_columns
        matrix_df.index = translated_index

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 6.2))
    matrix_specs = [
        (
            translated_real,
            "a",
            "Real data correlation matrix" if localizer.is_english else "a 真实数据相关性矩阵",
            "RdBu_r",
            -1.0,
            1.0,
        ),
        (
            translated_synthetic,
            "b",
            "Synthetic data correlation matrix" if localizer.is_english else "b 合成数据相关性矩阵",
            "RdBu_r",
            -1.0,
            1.0,
        ),
        (
            translated_diff,
            "c",
            "Difference matrix" if localizer.is_english else "c 差异矩阵",
            "cividis",
            0.0,
            1.0,
        ),
    ]

    colorbar_handles: list[tuple[Any, Any]] = []
    for ax, (matrix_df, panel_label, title, cmap_name, vmin, vmax) in zip(axes, matrix_specs):
        image = ax.imshow(matrix_df.to_numpy(dtype=float), cmap=cmap_name, vmin=vmin, vmax=vmax, aspect="auto")
        colorbar_handles.append((ax, image))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=11, pad=10)
        ax.text(
            0.0,
            1.04,
            panel_label,
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    for ax, image in colorbar_handles:
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    file_path = output_path / "pwcc_heatmaps_triptych.png"
    fig.savefig(file_path, dpi=300 if localizer.is_english else 200, bbox_inches="tight")
    plt.close(fig)
    return [str(file_path)]

def save_ccd_dot_to_diagonal_plots(
    details_df: pd.DataFrame,
    output_dir: str,
    localizer: OutputLocalizer | None = None,
) -> List[str]:
    if details_df.empty:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    localizer = localizer or OutputLocalizer()
    localizer.configure_matplotlib()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    valid_details = details_df[details_df["point_status"] == "ok"].copy()
    if valid_details.empty:
        return saved_paths

    use_standardized_axis = {
        "x_value_standardized",
        "y_value_standardized",
    }.issubset(valid_details.columns)
    x_value_column = "x_value_standardized" if use_standardized_axis else "x_value"
    y_value_column = "y_value_standardized" if use_standardized_axis else "y_value"
    axis_suffix = (
        " (standardized by real SD)" if localizer.is_english else "（按真实数据标准差标准化）"
    ) if use_standardized_axis else ""

    comparison_label_map = {
        "real_vs_synthetic": ("Real", "Synthetic", "real_vs_synthetic"),
        "real_vs_real_reference": ("Real Half A", "Real Half B", "real_vs_real_reference"),
    }

    def _plot_subset(plot_df: pd.DataFrame, title: str, file_name: str, x_label: str, y_label: str) -> None:
        if plot_df.empty:
            return

        x_values = plot_df[x_value_column].to_numpy(dtype=float)
        y_values = plot_df[y_value_column].to_numpy(dtype=float)
        min_axis = float(np.nanmin(np.concatenate([x_values, y_values])))
        max_axis = float(np.nanmax(np.concatenate([x_values, y_values])))
        if np.isfinite(min_axis) and np.isfinite(max_axis) and min_axis == max_axis:
            padding = 1.0 if min_axis == 0 else abs(min_axis) * 0.05
            min_axis -= padding
            max_axis += padding

        fig, ax = plt.subplots(figsize=(6.6, 6.4) if localizer.is_english else (7, 7))
        ax.scatter(x_values, y_values, alpha=0.7, s=24, color=PRIMARY_COLOR)
        ax.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--", linewidth=1.0, color=ACCENT_COLOR)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        file_path = output_path / file_name
        fig.savefig(file_path, dpi=300 if localizer.is_english else 200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(file_path))

    for comparison_type, (x_prefix, y_prefix, comparison_slug) in comparison_label_map.items():
        comparison_df = valid_details[valid_details["comparison_type"] == comparison_type].copy()
        if comparison_df.empty:
            continue

        for statistic in ("mean", "std"):
            statistic_df = comparison_df[comparison_df["statistic"] == statistic].copy()
            if statistic_df.empty:
                continue

            overall_file_name = f"ccd_all_variables_{statistic}_{comparison_slug}_dot_to_diagonal.png"
            overall_title = (
                f"CCD: all variables ({statistic})"
                if localizer.is_english else f"CCD Dot-to-diagonal Plot: All Variables ({statistic}, {comparison_slug})"
            )
            _plot_subset(
                statistic_df,
                title=overall_title,
                file_name=overall_file_name,
                x_label=f"{x_prefix} {statistic}{axis_suffix}",
                y_label=f"{y_prefix} {statistic}{axis_suffix}",
            )

            for continuous_column in sorted(statistic_df["continuous_column"].dropna().unique()):
                feature_df = statistic_df[statistic_df["continuous_column"] == continuous_column].copy()
                if feature_df.empty:
                    continue

                file_stem = localizer.translate_text(continuous_column)
                file_name = (
                    f"ccd_{localizer.sanitize_filename(file_stem)}_{statistic}_{comparison_slug}_dot_to_diagonal.png"
                )
                title = (
                    f"CCD: {localizer.translate_text(continuous_column)} ({statistic})"
                    if localizer.is_english else f"CCD Dot-to-diagonal Plot: {continuous_column} ({statistic}, {comparison_slug})"
                )
                _plot_subset(
                    feature_df,
                    title=title,
                    file_name=file_name,
                    x_label=f"{x_prefix} {statistic}{axis_suffix}",
                    y_label=f"{y_prefix} {statistic}{axis_suffix}",
                )

    return saved_paths

def save_ccd_distance_distribution_plots(
    details_df: pd.DataFrame,
    output_dir: str,
    localizer: OutputLocalizer | None = None,
) -> List[str]:
    if details_df.empty:
        return []

    localizer = localizer or OutputLocalizer()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    valid_details = details_df[details_df["point_status"] == "ok"].copy()
    if valid_details.empty:
        return saved_paths

    uses_standardized_distance = "diagonal_distance_raw" in valid_details.columns
    distance_label_suffix = (
        " (standardized by real SD)" if localizer.is_english else "（按真实数据标准差标准化）"
    ) if uses_standardized_distance else ""
    x_axis_label = (
        "Distance to diagonal (standardized by real SD)"
        if localizer.is_english else "点到对角线距离（按真实数据标准差标准化）"
    ) if uses_standardized_distance else None

    for statistic in ("mean", "std"):
        statistic_df = valid_details[valid_details["statistic"] == statistic].copy()
        if statistic_df.empty:
            continue

        real_vs_synthetic_df = statistic_df[statistic_df["comparison_type"] == "real_vs_synthetic"].copy()
        real_vs_real_reference_df = statistic_df[statistic_df["comparison_type"] == "real_vs_real_reference"].copy()

        overall_file_path = output_path / f"ccd_all_variables_{statistic}_distance_distribution_comparison.png"
        if _plot_distance_distribution_comparison(
            real_vs_synthetic_distances=real_vs_synthetic_df["diagonal_distance"].to_numpy(dtype=float),
            real_vs_real_reference_distances=real_vs_real_reference_df["diagonal_distance"].to_numpy(dtype=float),
            title=(
                f"CCD distance distribution{distance_label_suffix} (all variables, {statistic})"
                if localizer.is_english else f"CCD 距离分布对比{distance_label_suffix}：全部变量（{statistic}）"
            ),
            file_path=overall_file_path,
            localizer=localizer,
            x_axis_label=x_axis_label,
        ):
            saved_paths.append(str(overall_file_path))

        for continuous_column in sorted(statistic_df["continuous_column"].dropna().unique()):
            feature_rs_df = real_vs_synthetic_df[real_vs_synthetic_df["continuous_column"] == continuous_column].copy()
            feature_rr_df = real_vs_real_reference_df[
                real_vs_real_reference_df["continuous_column"] == continuous_column
            ].copy()
            feature_name = localizer.translate_text(continuous_column)
            feature_file_path = output_path / (
                f"ccd_{localizer.sanitize_filename(feature_name)}_{statistic}_distance_distribution_comparison.png"
            )
            if _plot_distance_distribution_comparison(
                real_vs_synthetic_distances=feature_rs_df["diagonal_distance"].to_numpy(dtype=float),
                real_vs_real_reference_distances=feature_rr_df["diagonal_distance"].to_numpy(dtype=float),
                title=(
                    f"CCD distance distribution{distance_label_suffix}: {feature_name} ({statistic})"
                    if localizer.is_english else f"CCD 距离分布对比{distance_label_suffix}：{continuous_column}（{statistic}）"
                ),
                file_path=feature_file_path,
                localizer=localizer,
                x_axis_label=x_axis_label,
            ):
                saved_paths.append(str(feature_file_path))

    return saved_paths

def save_cvt_histogram_plots(
    synthetic_df: pd.DataFrame,
    range_columns: List[str],
    difference_column_pairs: List[Tuple[str, str]],
    output_dir: str,
    localizer: OutputLocalizer | None = None,
) -> List[str]:
    if synthetic_df.empty:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    localizer = localizer or OutputLocalizer()
    localizer.configure_matplotlib()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []

    def _bin_count(values: np.ndarray) -> int:
        return int(min(60, max(12, np.sqrt(max(len(values), 1)))))

    for column in range_columns:
        if column not in synthetic_df.columns:
            continue
        numeric_values = pd.to_numeric(synthetic_df[column], errors="coerce").dropna().to_numpy(dtype=float)
        if numeric_values.size == 0:
            continue

        translated_column = localizer.translate_text(column)
        fig, ax = plt.subplots(figsize=(8.0, 5.0) if localizer.is_english else (8.5, 5.5))
        ax.hist(
            numeric_values,
            bins=_bin_count(numeric_values),
            color=PRIMARY_COLOR,
            edgecolor="white",
            alpha=0.85,
        )
        ax.set_xlabel(translated_column)
        ax.set_ylabel("Count" if localizer.is_english else "频数")
        ax.set_title(
            f"CVT histogram: {translated_column}"
            if localizer.is_english else f"CVT 频数直方图：{column}"
        )
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        file_path = output_path / f"cvt_range_{localizer.sanitize_filename(translated_column)}_frequency_histogram.png"
        fig.savefig(file_path, dpi=300 if localizer.is_english else 200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(file_path))

    for left_column, right_column in difference_column_pairs:
        if left_column not in synthetic_df.columns or right_column not in synthetic_df.columns:
            continue

        left_values = pd.to_numeric(synthetic_df[left_column], errors="coerce")
        right_values = pd.to_numeric(synthetic_df[right_column], errors="coerce")
        diff_values = (left_values - right_values).dropna().to_numpy(dtype=float)
        if diff_values.size == 0:
            continue

        translated_left = localizer.translate_text(left_column)
        translated_right = localizer.translate_text(right_column)
        fig, ax = plt.subplots(figsize=(8.0, 5.0) if localizer.is_english else (8.5, 5.5))
        ax.hist(
            diff_values,
            bins=_bin_count(diff_values),
            color=SECONDARY_COLOR,
            edgecolor="white",
            alpha=0.85,
        )
        ax.axvline(
            0.0,
            color=ACCENT_COLOR,
            linestyle="--",
            linewidth=1.0,
            label="Reference = 0" if localizer.is_english else "0 参考线",
        )
        ax.set_xlabel(f"{translated_left} - {translated_right}")
        ax.set_ylabel("Count" if localizer.is_english else "频数")
        ax.set_title(
            f"CVT histogram: {translated_left} - {translated_right}"
            if localizer.is_english else f"CVT 差值频数直方图：{left_column} - {right_column}"
        )
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=False)
        fig.tight_layout()

        file_path = output_path / (
            f"cvt_difference_{localizer.sanitize_filename(translated_left)}_minus_{localizer.sanitize_filename(translated_right)}_frequency_histogram.png"
        )
        fig.savefig(file_path, dpi=300 if localizer.is_english else 200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(file_path))

    return saved_paths

class PairwiseCorrelationCoefficientEvaluator(MultivariateSimilarityEvaluator):
    name = "pairwise_correlation_coefficient"

    def _resolve_variable_types(self, context: EvaluationContext) -> Dict[str, str]:
        continuous_set = set(context.schema.continuous_columns)
        categorical_set = set(context.schema.categorical_columns) | set(context.schema.outcome_columns)
        variable_types: Dict[str, str] = {}
        for column in context.schema.common_columns:
            if column in continuous_set:
                variable_types[column] = "continuous"
            elif column in categorical_set:
                variable_types[column] = "categorical"
        return variable_types

    def _association_method(self, left_type: str, right_type: str) -> str:
        pair_types = {left_type, right_type}
        if pair_types == {"continuous"}:
            return "pearson"
        if pair_types == {"categorical"}:
            return "cramers_v"
        return "correlation_ratio_eta"

    def _pairwise_association(
        self,
        df: pd.DataFrame,
        left_column: str,
        left_type: str,
        right_column: str,
        right_type: str,
    ) -> float:
        if left_column == right_column:
            return 1.0

        if left_type == "continuous" and right_type == "continuous":
            pair_df = df[[left_column, right_column]].copy()
            pair_df[left_column] = pd.to_numeric(pair_df[left_column], errors="coerce")
            pair_df[right_column] = pd.to_numeric(pair_df[right_column], errors="coerce")
            pair_df = pair_df.dropna()
            if len(pair_df) < 2:
                return np.nan
            if pair_df[left_column].nunique(dropna=True) < 2 or pair_df[right_column].nunique(dropna=True) < 2:
                return np.nan
            return float(pair_df[left_column].corr(pair_df[right_column], method="pearson"))

        if left_type == "categorical" and right_type == "categorical":
            pair_df = df[[left_column, right_column]].copy().dropna()
            if len(pair_df) == 0:
                return np.nan
            left_tokens = series_to_categorical_tokens(pair_df[left_column])
            right_tokens = series_to_categorical_tokens(pair_df[right_column])
            return float(cramers_v(pd.crosstab(left_tokens, right_tokens)))

        if left_type == "continuous":
            continuous_column, categorical_column = left_column, right_column
        else:
            continuous_column, categorical_column = right_column, left_column

        pair_df = df[[continuous_column, categorical_column]].copy()
        pair_df[continuous_column] = pd.to_numeric(pair_df[continuous_column], errors="coerce")
        pair_df[categorical_column] = series_to_categorical_tokens(pair_df[categorical_column])
        pair_df = pair_df.dropna()
        if len(pair_df) < 2:
            return np.nan
        if pair_df[categorical_column].nunique(dropna=True) < 2:
            return np.nan
        return float(correlation_ratio(pair_df[categorical_column], pair_df[continuous_column]))

    def _build_association_matrix(
        self,
        df: pd.DataFrame,
        variable_types: Dict[str, str],
    ) -> pd.DataFrame:
        columns = list(variable_types.keys())
        matrix = pd.DataFrame(np.nan, index=columns, columns=columns, dtype=float)
        for idx, left_column in enumerate(columns):
            matrix.loc[left_column, left_column] = 1.0
            for right_column in columns[idx + 1:]:
                association_value = self._pairwise_association(
                    df=df,
                    left_column=left_column,
                    left_type=variable_types[left_column],
                    right_column=right_column,
                    right_type=variable_types[right_column],
                )
                matrix.loc[left_column, right_column] = association_value
                matrix.loc[right_column, left_column] = association_value
        return matrix

    def _build_pair_details(
        self,
        real_matrix: pd.DataFrame,
        synthetic_matrix: pd.DataFrame,
        diff_matrix: pd.DataFrame,
        variable_types: Dict[str, str],
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        columns = list(variable_types.keys())
        for idx, left_column in enumerate(columns):
            for right_column in columns[idx + 1:]:
                left_type = variable_types[left_column]
                right_type = variable_types[right_column]
                rows.append({
                    "column_1": left_column,
                    "column_2": right_column,
                    "type_1": left_type,
                    "type_2": right_type,
                    "association_method": self._association_method(left_type, right_type),
                    "real_association": float(real_matrix.loc[left_column, right_column]),
                    "synthetic_association": float(synthetic_matrix.loc[left_column, right_column]),
                    "absolute_difference": float(diff_matrix.loc[left_column, right_column]),
                })
        return pd.DataFrame(rows)

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        variable_types = self._resolve_variable_types(context)
        if len(variable_types) < 2:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"variable_count": int(len(variable_types)), "pair_count": 0},
                tables={},
                metadata={"method": "pwcc", "variable_types": variable_types},
                warnings=["有效变量数不足，无法计算 PWCC 关联矩阵"],
            )

        real_matrix = self._build_association_matrix(context.comparison_real_data, variable_types)
        synthetic_matrix = self._build_association_matrix(context.synthetic_data, variable_types)
        absolute_diff_matrix = (real_matrix - synthetic_matrix).abs()
        pair_details_df = self._build_pair_details(
            real_matrix=real_matrix,
            synthetic_matrix=synthetic_matrix,
            diff_matrix=absolute_diff_matrix,
            variable_types=variable_types,
        )

        summary_values = _pairwise_absolute_diff_summary(absolute_diff_matrix)
        summary_values["variable_count"] = int(len(variable_types))
        summary_df = pd.DataFrame([summary_values])

        return MetricResult(
            name=self.name,
            domain=self.domain,
            level=self.level,
            summary=summary_values,
            tables={
                "summary": summary_df,
                "real_matrix": real_matrix,
                "synthetic_matrix": synthetic_matrix,
                "absolute_diff_matrix": absolute_diff_matrix,
                "pair_details": pair_details_df,
            },
            metadata={
                "method": "pwcc",
                "variable_types": variable_types,
            },
            warnings=[],
        )

class CrossTypeConditionalDistributionEvaluator(MultivariateSimilarityEvaluator):
    name = "cross_type_conditional_distribution"

    def _resolve_grouping_spec(self, context: EvaluationContext) -> Tuple[List[str], List[Dict[str, Any]], List[str]]:
        warnings: List[str] = []
        categorical_like_columns = set(context.schema.categorical_columns) | set(context.schema.outcome_columns)

        valid_categorical_columns: List[str] = []
        for column in list(CCD_GROUPING_CATEGORICAL_COLUMNS):
            if column not in context.schema.common_columns:
                warnings.append(f"CCD 分组列 `{column}` 不在真实/合成公共列中，已忽略")
                continue
            if column not in categorical_like_columns:
                warnings.append(f"CCD 分组列 `{column}` 不是离散类别列，已忽略")
                continue
            valid_categorical_columns.append(column)

        valid_continuous_specs: List[Dict[str, Any]] = []
        for raw_spec in list(CCD_GROUPING_CONTINUOUS_COLUMNS):
            if not isinstance(raw_spec, dict):
                warnings.append("CCD 连续分组配置存在非 dict 元素，已忽略")
                continue

            column = raw_spec.get("column")
            thresholds = raw_spec.get("thresholds", [])
            if not isinstance(column, str):
                warnings.append("CCD 连续分组配置缺少合法的 `column` 字段，已忽略")
                continue
            if column not in context.schema.common_columns:
                warnings.append(f"CCD 连续分组列 `{column}` 不在真实/合成公共列中，已忽略")
                continue
            if column not in context.schema.continuous_columns:
                warnings.append(f"CCD 连续分组列 `{column}` 不是连续变量，已忽略")
                continue
            if not isinstance(thresholds, list):
                warnings.append(f"CCD 连续分组列 `{column}` 的 thresholds 不是列表，已忽略")
                continue

            numeric_thresholds: List[float] = []
            invalid_threshold = False
            for threshold in thresholds:
                numeric_threshold = pd.to_numeric(pd.Series([threshold]), errors="coerce").iloc[0]
                if pd.isna(numeric_threshold):
                    invalid_threshold = True
                    break
                numeric_thresholds.append(float(numeric_threshold))
            if invalid_threshold:
                warnings.append(f"CCD 连续分组列 `{column}` 的 thresholds 含非数值项，已忽略")
                continue

            valid_continuous_specs.append({
                "column": column,
                "thresholds": sorted(set(numeric_thresholds)),
            })

        if not valid_categorical_columns and not valid_continuous_specs:
            warnings.append("CCD 没有可用的分组列，已跳过评估")

        return valid_categorical_columns, valid_continuous_specs, warnings

    def _continuous_group_token(
        self,
        series: pd.Series,
        thresholds: List[float],
        column_name: str,
    ) -> pd.Series:
        numeric_series = pd.to_numeric(series, errors="coerce")
        tokens = pd.Series(index=series.index, dtype=object)
        missing_mask = numeric_series.isna()
        tokens.loc[missing_mask] = "__MISSING__"

        valid_values = numeric_series.loc[~missing_mask]
        if not thresholds:
            tokens.loc[~missing_mask] = valid_values.apply(normalize_categorical_value)
            return tokens.astype(str)

        def _value_to_bin_label(value: float) -> str:
            previous = None
            for threshold in thresholds:
                if value <= threshold:
                    if previous is None:
                        return f"{column_name}<={normalize_categorical_value(threshold)}"
                    return (
                        f"{normalize_categorical_value(previous)}<{column_name}<={normalize_categorical_value(threshold)}"
                    )
                previous = threshold
            return f"{column_name}>{normalize_categorical_value(thresholds[-1])}"

        tokens.loc[~missing_mask] = valid_values.apply(lambda value: _value_to_bin_label(float(value)))
        return tokens.astype(str)

    def _build_group_labels(
        self,
        df: pd.DataFrame,
        categorical_grouping_columns: List[str],
        continuous_grouping_specs: List[Dict[str, Any]],
    ) -> pd.Series:
        token_df = pd.DataFrame(index=df.index)
        for column in categorical_grouping_columns:
            token_df[column] = series_to_categorical_tokens(df[column])
        for spec in continuous_grouping_specs:
            column = spec["column"]
            thresholds = list(spec["thresholds"])
            token_df[column] = self._continuous_group_token(df[column], thresholds, column)

        grouping_columns = list(token_df.columns)
        if len(grouping_columns) == 1:
            return token_df[grouping_columns[0]].astype(str)

        return token_df.apply(
            lambda row: " | ".join(f"{column}={row[column]}" for column in grouping_columns),
            axis=1,
        )

    def _build_pair_point_rows(
        self,
        x_df: pd.DataFrame,
        y_df: pd.DataFrame,
        continuous_columns: List[str],
        real_std_scales: Dict[str, float],
        grouping_label: str,
        comparison_type: str,
        x_label: str,
        y_label: str,
    ) -> pd.DataFrame:
        if not continuous_columns:
            return pd.DataFrame()

        all_group_keys = sorted(set(x_df["_ccd_group_key"].dropna()) | set(y_df["_ccd_group_key"].dropna()))
        rows: List[Dict[str, Any]] = []

        for continuous_column in continuous_columns:
            x_numeric = pd.to_numeric(x_df[continuous_column], errors="coerce")
            y_numeric = pd.to_numeric(y_df[continuous_column], errors="coerce")
            for group_key in all_group_keys:
                x_group_values = x_numeric[x_df["_ccd_group_key"] == group_key].dropna()
                y_group_values = y_numeric[y_df["_ccd_group_key"] == group_key].dropna()

                statistic_specs = [
                    ("mean", float(x_group_values.mean()) if len(x_group_values) > 0 else np.nan, float(y_group_values.mean()) if len(y_group_values) > 0 else np.nan),
                    ("std", float(x_group_values.std()) if len(x_group_values) > 1 else np.nan, float(y_group_values.std()) if len(y_group_values) > 1 else np.nan),
                ]
                for statistic, x_value, y_value in statistic_specs:
                    real_std_scale = float(real_std_scales.get(continuous_column, np.nan))
                    if pd.isna(x_value) or pd.isna(y_value):
                        point_status = "missing_group_value"
                        x_value_standardized = np.nan
                        y_value_standardized = np.nan
                        signed_gap_standardized = np.nan
                        diagonal_distance_raw = np.nan
                        diagonal_distance = np.nan
                    elif not _is_valid_standardization_scale(real_std_scale):
                        point_status = "invalid_real_std"
                        x_value_standardized = np.nan
                        y_value_standardized = np.nan
                        signed_gap_standardized = np.nan
                        diagonal_distance_raw = _diagonal_distance(x_value, y_value)
                        diagonal_distance = np.nan
                    else:
                        point_status = "ok"
                        x_value_standardized = float(x_value / real_std_scale)
                        y_value_standardized = float(y_value / real_std_scale)
                        signed_gap_standardized = float(y_value_standardized - x_value_standardized)
                        diagonal_distance_raw = _diagonal_distance(x_value, y_value)
                        diagonal_distance = _diagonal_distance(x_value_standardized, y_value_standardized)
                    rows.append({
                        "comparison_type": comparison_type,
                        "continuous_column": continuous_column,
                        "statistic": statistic,
                        "group_key": str(group_key),
                        "grouping_columns": grouping_label,
                        "x_label": x_label,
                        "y_label": y_label,
                        "x_count": int(len(x_group_values)),
                        "y_count": int(len(y_group_values)),
                        "x_value": x_value,
                        "y_value": y_value,
                        "signed_gap": float(y_value - x_value) if pd.notna(x_value) and pd.notna(y_value) else np.nan,
                        "real_std_scale": real_std_scale,
                        "x_value_standardized": x_value_standardized,
                        "y_value_standardized": y_value_standardized,
                        "signed_gap_standardized": signed_gap_standardized,
                        "diagonal_distance_raw": diagonal_distance_raw,
                        "diagonal_distance": diagonal_distance,
                        "point_status": point_status,
                    })

        return pd.DataFrame(rows)

    def _build_point_rows(
        self,
        context: EvaluationContext,
        categorical_grouping_columns: List[str],
        continuous_grouping_specs: List[Dict[str, Any]],
    ) -> pd.DataFrame:
        continuous_columns = list(context.valid_continuous_columns)
        if not continuous_columns:
            return pd.DataFrame()

        grouping_source_columns = list(dict.fromkeys(
            list(categorical_grouping_columns)
            + [str(spec["column"]) for spec in continuous_grouping_specs]
        ))
        selected_columns = list(dict.fromkeys(grouping_source_columns + continuous_columns))
        real_df = context.comparison_real_data[selected_columns].copy()
        synthetic_df = context.synthetic_data[selected_columns].copy()
        real_half_a, real_half_b = split_real_reference_data(context.comparison_real_data, context.random_state)
        real_ref_a_df = real_half_a[selected_columns].copy()
        real_ref_b_df = real_half_b[selected_columns].copy()
        real_std_scales: Dict[str, float] = {}
        for continuous_column in continuous_columns:
            real_numeric_values = pd.to_numeric(context.comparison_real_data[continuous_column], errors="coerce").dropna()
            real_std_scales[continuous_column] = (
                float(real_numeric_values.std()) if len(real_numeric_values) > 1 else np.nan
            )

        real_df["_ccd_group_key"] = self._build_group_labels(real_df, categorical_grouping_columns, continuous_grouping_specs)
        synthetic_df["_ccd_group_key"] = self._build_group_labels(
            synthetic_df,
            categorical_grouping_columns,
            continuous_grouping_specs,
        )
        real_ref_a_df["_ccd_group_key"] = self._build_group_labels(
            real_ref_a_df,
            categorical_grouping_columns,
            continuous_grouping_specs,
        )
        real_ref_b_df["_ccd_group_key"] = self._build_group_labels(
            real_ref_b_df,
            categorical_grouping_columns,
            continuous_grouping_specs,
        )

        grouping_label_parts = list(categorical_grouping_columns) + [str(spec["column"]) for spec in continuous_grouping_specs]
        grouping_label = " | ".join(grouping_label_parts)

        rows = [
            self._build_pair_point_rows(
                x_df=real_df,
                y_df=synthetic_df,
                continuous_columns=continuous_columns,
                real_std_scales=real_std_scales,
                grouping_label=grouping_label,
                comparison_type="real_vs_synthetic",
                x_label="real",
                y_label="synthetic",
            ),
            self._build_pair_point_rows(
                x_df=real_ref_a_df,
                y_df=real_ref_b_df,
                continuous_columns=continuous_columns,
                real_std_scales=real_std_scales,
                grouping_label=grouping_label,
                comparison_type="real_vs_real_reference",
                x_label="real_half_a",
                y_label="real_half_b",
            ),
        ]
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def _build_summary_rows(
        self,
        details_df: pd.DataFrame,
        grouping_label: str,
    ) -> pd.DataFrame:
        if details_df.empty:
            return pd.DataFrame()

        rows: List[Dict[str, Any]] = []
        for comparison_type, comparison_df in details_df.groupby("comparison_type"):
            for statistic, statistic_df in comparison_df.groupby("statistic"):
                for continuous_column, feature_df in statistic_df.groupby("continuous_column"):
                    valid_distances = feature_df.loc[feature_df["point_status"] == "ok", "diagonal_distance"].dropna()
                    rows.append({
                        "comparison_type": comparison_type,
                        "continuous_column": continuous_column,
                        "grouping_columns": grouping_label,
                        "statistic": statistic,
                        "group_point_count": int(len(feature_df)),
                        "valid_point_count": int(len(valid_distances)),
                        "mean_diagonal_distance": float(valid_distances.mean()) if not valid_distances.empty else np.nan,
                        "median_diagonal_distance": float(valid_distances.median()) if not valid_distances.empty else np.nan,
                        "std_diagonal_distance": float(valid_distances.std()) if len(valid_distances) > 1 else np.nan,
                    })

                all_valid_distances = statistic_df.loc[statistic_df["point_status"] == "ok", "diagonal_distance"].dropna()
                rows.append({
                    "comparison_type": comparison_type,
                    "continuous_column": "__overall__",
                    "grouping_columns": grouping_label,
                    "statistic": statistic,
                    "group_point_count": int(len(statistic_df)),
                    "valid_point_count": int(len(all_valid_distances)),
                    "mean_diagonal_distance": float(all_valid_distances.mean()) if not all_valid_distances.empty else np.nan,
                    "median_diagonal_distance": float(all_valid_distances.median()) if not all_valid_distances.empty else np.nan,
                    "std_diagonal_distance": float(all_valid_distances.std()) if len(all_valid_distances) > 1 else np.nan,
                })

        return pd.DataFrame(rows)

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        categorical_grouping_columns, continuous_grouping_specs, warnings = self._resolve_grouping_spec(context)
        if not categorical_grouping_columns and not continuous_grouping_specs:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={
                    "grouping_column_count": 0,
                    "categorical_grouping_column_count": 0,
                    "continuous_grouping_column_count": 0,
                    "continuous_column_count": int(len(context.valid_continuous_columns)),
                },
                tables={},
                metadata={
                    "method": "ccd",
                    "categorical_grouping_columns": [],
                    "continuous_grouping_columns": [],
                    "plot_file_patterns": [],
                },
                warnings=warnings,
            )

        details_df = self._build_point_rows(context, categorical_grouping_columns, continuous_grouping_specs)
        grouping_label_parts = list(categorical_grouping_columns) + [str(spec["column"]) for spec in continuous_grouping_specs]
        grouping_label = " | ".join(grouping_label_parts)
        if not details_df.empty:
            invalid_scale_columns = sorted(
                details_df.loc[details_df["point_status"] == "invalid_real_std", "continuous_column"].dropna().unique()
            )
            if invalid_scale_columns:
                warnings.append(
                    "CCD 中以下连续变量在真实参考数据中的标准差不可用或为 0，已跳过标准化距离计算："
                    + ", ".join(str(column) for column in invalid_scale_columns)
                )
        summary_df = self._build_summary_rows(details_df, grouping_label)

        def _summary_value(comparison_type: str, statistic: str, metric_column: str) -> float:
            row = summary_df[
                (summary_df["comparison_type"] == comparison_type)
                & (summary_df["continuous_column"] == "__overall__")
                & (summary_df["statistic"] == statistic)
            ]
            return float(row[metric_column].iloc[0]) if not row.empty else np.nan

        summary = {
            "grouping_column_count": int(len(categorical_grouping_columns) + len(continuous_grouping_specs)),
            "categorical_grouping_column_count": int(len(categorical_grouping_columns)),
            "continuous_grouping_column_count": int(len(continuous_grouping_specs)),
            "continuous_column_count": int(len(context.valid_continuous_columns)),
            "group_count": int(details_df["group_key"].nunique()) if not details_df.empty else 0,
            "real_vs_synthetic_mean_distance_mean": _summary_value("real_vs_synthetic", "mean", "mean_diagonal_distance"),
            "real_vs_synthetic_mean_distance_median": _summary_value("real_vs_synthetic", "mean", "median_diagonal_distance"),
            "real_vs_synthetic_mean_distance_std": _summary_value("real_vs_synthetic", "mean", "std_diagonal_distance"),
            "real_vs_synthetic_std_distance_mean": _summary_value("real_vs_synthetic", "std", "mean_diagonal_distance"),
            "real_vs_synthetic_std_distance_median": _summary_value("real_vs_synthetic", "std", "median_diagonal_distance"),
            "real_vs_synthetic_std_distance_std": _summary_value("real_vs_synthetic", "std", "std_diagonal_distance"),
            "real_vs_real_reference_mean_distance_mean": _summary_value("real_vs_real_reference", "mean", "mean_diagonal_distance"),
            "real_vs_real_reference_mean_distance_median": _summary_value("real_vs_real_reference", "mean", "median_diagonal_distance"),
            "real_vs_real_reference_mean_distance_std": _summary_value("real_vs_real_reference", "mean", "std_diagonal_distance"),
            "real_vs_real_reference_std_distance_mean": _summary_value("real_vs_real_reference", "std", "mean_diagonal_distance"),
            "real_vs_real_reference_std_distance_median": _summary_value("real_vs_real_reference", "std", "median_diagonal_distance"),
            "real_vs_real_reference_std_distance_std": _summary_value("real_vs_real_reference", "std", "std_diagonal_distance"),
        }

        return MetricResult(
            name=self.name,
            domain=self.domain,
            level=self.level,
            summary=summary,
            tables={
                "summary": summary_df,
                "details": details_df,
            },
            metadata={
                "method": "ccd",
                "categorical_grouping_columns": categorical_grouping_columns,
                "continuous_grouping_columns": continuous_grouping_specs,
                "distance_definition": "standardized_diagonal_distance_by_real_std",
                "point_value_definition": {
                    "x_value": "raw_group_statistic",
                    "y_value": "raw_group_statistic",
                    "x_value_standardized": "raw_group_statistic_divided_by_real_reference_std",
                    "y_value_standardized": "raw_group_statistic_divided_by_real_reference_std",
                    "diagonal_distance": "distance_to_diagonal_after_standardizing_by_real_reference_std",
                    "diagonal_distance_raw": "distance_to_diagonal_on_original_scale",
                },
                "plot_file_patterns": [
                    "ccd_<continuous_column>_<statistic>_real_vs_synthetic_dot_to_diagonal.png",
                    "ccd_<continuous_column>_<statistic>_real_vs_real_reference_dot_to_diagonal.png",
                    "ccd_all_variables_<statistic>_real_vs_synthetic_dot_to_diagonal.png",
                    "ccd_all_variables_<statistic>_real_vs_real_reference_dot_to_diagonal.png",
                    "ccd_<continuous_column>_<statistic>_distance_distribution_comparison.png",
                    "ccd_all_variables_<statistic>_distance_distribution_comparison.png",
                ],
            },
            warnings=warnings,
        )

class ConstraintViolationTestEvaluator(MultivariateSimilarityEvaluator):
    name = "constraint_violation_test"

    def _resolve_configs(
        self,
        context: EvaluationContext,
    ) -> Tuple[List[str], List[Tuple[str, str]], List[str]]:
        warnings: List[str] = []

        valid_range_columns: List[str] = []
        for column in list(CVT_RANGE_HISTOGRAM_COLUMNS):
            if not isinstance(column, str):
                warnings.append("CVT_RANGE_HISTOGRAM_COLUMNS 中存在非字符串元素，已忽略")
                continue
            if column not in context.synthetic_data.columns:
                warnings.append(f"CVT 范围展示列 `{column}` 不在合成数据中，已忽略")
                continue
            valid_range_columns.append(column)

        valid_difference_pairs: List[Tuple[str, str]] = []
        for pair in list(CVT_DIFFERENCE_COLUMN_PAIRS):
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                warnings.append("CVT_DIFFERENCE_COLUMN_PAIRS 中存在非二元组元素，已忽略")
                continue
            left_column = str(pair[0])
            right_column = str(pair[1])
            missing_columns = [
                column for column in [left_column, right_column]
                if column not in context.synthetic_data.columns
            ]
            if missing_columns:
                warnings.append(
                    f"CVT 差值展示列 `{left_column} - {right_column}` 缺少列 {missing_columns}，已忽略"
                )
                continue
            valid_difference_pairs.append((left_column, right_column))

        if not valid_range_columns and not valid_difference_pairs:
            warnings.append("CVT 没有可用展示配置，已跳过评估")

        return valid_range_columns, valid_difference_pairs, warnings

    def _build_summary_rows(
        self,
        synthetic_df: pd.DataFrame,
        range_columns: List[str],
        difference_pairs: List[Tuple[str, str]],
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []

        for column in range_columns:
            raw_series = synthetic_df[column]
            numeric_series = pd.to_numeric(raw_series, errors="coerce")
            valid_values = numeric_series.dropna()
            rows.append({
                "item_type": "range_column",
                "item_name": str(column),
                "left_column": str(column),
                "right_column": "",
                "display_label": str(column),
                "row_count": int(len(raw_series)),
                "valid_count": int(valid_values.size),
                "missing_count": int(raw_series.isna().sum() + (numeric_series.isna() & raw_series.notna()).sum()),
                "min_value": float(valid_values.min()) if not valid_values.empty else np.nan,
                "max_value": float(valid_values.max()) if not valid_values.empty else np.nan,
                "mean_value": float(valid_values.mean()) if not valid_values.empty else np.nan,
                "median_value": float(valid_values.median()) if not valid_values.empty else np.nan,
                "std_value": float(valid_values.std()) if len(valid_values) > 1 else np.nan,
                "negative_value_ratio": float((valid_values < 0).mean()) if not valid_values.empty else np.nan,
            })

        for left_column, right_column in difference_pairs:
            left_series = pd.to_numeric(synthetic_df[left_column], errors="coerce")
            right_series = pd.to_numeric(synthetic_df[right_column], errors="coerce")
            diff_series = (left_series - right_series).dropna()
            rows.append({
                "item_type": "difference_pair",
                "item_name": f"{left_column} - {right_column}",
                "left_column": str(left_column),
                "right_column": str(right_column),
                "display_label": f"{left_column} - {right_column}",
                "row_count": int(len(synthetic_df)),
                "valid_count": int(diff_series.size),
                "missing_count": int(len(synthetic_df) - diff_series.size),
                "min_value": float(diff_series.min()) if not diff_series.empty else np.nan,
                "max_value": float(diff_series.max()) if not diff_series.empty else np.nan,
                "mean_value": float(diff_series.mean()) if not diff_series.empty else np.nan,
                "median_value": float(diff_series.median()) if not diff_series.empty else np.nan,
                "std_value": float(diff_series.std()) if len(diff_series) > 1 else np.nan,
                "negative_value_ratio": float((diff_series < 0).mean()) if not diff_series.empty else np.nan,
            })

        return pd.DataFrame(rows)

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        range_columns, difference_pairs, warnings = self._resolve_configs(context)
        if not range_columns and not difference_pairs:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={
                    "range_column_count": 0,
                    "difference_pair_count": 0,
                },
                tables={},
                metadata={
                    "method": "cvt",
                    "range_columns": [],
                    "difference_column_pairs": [],
                    "plot_file_patterns": [],
                },
                warnings=warnings,
            )

        summary_df = self._build_summary_rows(context.synthetic_data, range_columns, difference_pairs)
        summary = {
            "range_column_count": int(len(range_columns)),
            "difference_pair_count": int(len(difference_pairs)),
            "summary_item_count": int(len(summary_df)),
            "synthetic_row_count": int(len(context.synthetic_data)),
        }

        return MetricResult(
            name=self.name,
            domain=self.domain,
            level=self.level,
            summary=summary,
            tables={
                "summary": summary_df,
            },
            metadata={
                "method": "cvt",
                "range_columns": range_columns,
                "difference_column_pairs": difference_pairs,
                "plot_file_patterns": [
                    "cvt_range_<column>_frequency_histogram.png",
                    "cvt_difference_<left_column>_minus_<right_column>_frequency_histogram.png",
                ],
            },
            warnings=warnings,
        )
