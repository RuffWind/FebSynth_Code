from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon

REAL_DATA_COLOR = "#2F4B7C"
SYNTHETIC_DATA_COLOR = "#B07D4F"
REFERENCE_LINE_COLOR = "#6C757D"
SECONDARY_ACCENT_COLOR = "#8B3A3A"
PLOT_COLOR_CYCLE = [
    REAL_DATA_COLOR,
    SYNTHETIC_DATA_COLOR,
    "#4F6D7A",
    SECONDARY_ACCENT_COLOR,
    REFERENCE_LINE_COLOR,
]

def configure_matplotlib_for_chinese() -> None:
    try:
        import matplotlib
        from matplotlib import font_manager
        from cycler import cycler
    except ImportError:
        return

    preferred_fonts = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "PingFang SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    available_font_names = {font.name for font in font_manager.fontManager.ttflist}
    resolved_fonts = [font_name for font_name in preferred_fonts if font_name in available_font_names]
    if not resolved_fonts:
        resolved_fonts = preferred_fonts
    matplotlib.rcParams["font.family"] = ["sans-serif"]
    matplotlib.rcParams["font.sans-serif"] = resolved_fonts
    matplotlib.rcParams["axes.unicode_minus"] = False
    matplotlib.rcParams["axes.edgecolor"] = "#4a4a4a"
    matplotlib.rcParams["axes.linewidth"] = 0.7
    matplotlib.rcParams["axes.labelsize"] = 10.5
    matplotlib.rcParams["axes.titlesize"] = 11.5
    matplotlib.rcParams["axes.titleweight"] = "regular"
    matplotlib.rcParams["xtick.labelsize"] = 9
    matplotlib.rcParams["ytick.labelsize"] = 9
    matplotlib.rcParams["xtick.color"] = "#3d3d3d"
    matplotlib.rcParams["ytick.color"] = "#3d3d3d"
    matplotlib.rcParams["xtick.major.width"] = 0.6
    matplotlib.rcParams["ytick.major.width"] = 0.6
    matplotlib.rcParams["xtick.major.size"] = 3.0
    matplotlib.rcParams["ytick.major.size"] = 3.0
    matplotlib.rcParams["legend.fontsize"] = 8.5
    matplotlib.rcParams["legend.frameon"] = False
    matplotlib.rcParams["grid.color"] = "#d7d7d7"
    matplotlib.rcParams["grid.linewidth"] = 0.5
    matplotlib.rcParams["grid.alpha"] = 0.7
    matplotlib.rcParams["lines.linewidth"] = 1.25
    matplotlib.rcParams["patch.linewidth"] = 0.4
    matplotlib.rcParams["savefig.facecolor"] = "white"
    matplotlib.rcParams["figure.facecolor"] = "white"
    matplotlib.rcParams["axes.facecolor"] = "white"
    matplotlib.rcParams["axes.prop_cycle"] = cycler(color=PLOT_COLOR_CYCLE)

def normalize_categorical_value(value: Any) -> str:
    if pd.isna(value):
        return "__MISSING__"

    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"

    if isinstance(value, (int, np.integer)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric) and np.isclose(numeric, round(numeric)):
            return str(int(round(numeric)))
        return np.format_float_positional(numeric, trim="-")

    text = str(value).strip()
    numeric = pd.to_numeric(pd.Series([text]), errors="coerce").iloc[0]
    if not pd.isna(numeric):
        numeric = float(numeric)
        if np.isfinite(numeric) and np.isclose(numeric, round(numeric)):
            return str(int(round(numeric)))
        return np.format_float_positional(numeric, trim="-")

    return text

def series_to_categorical_tokens(series: pd.Series) -> pd.Series:
    return series.apply(normalize_categorical_value).astype(str)

def normalize_categorical_dataframe(
    df: pd.DataFrame,
    categorical_columns: Iterable[str],
) -> pd.DataFrame:
    normalized_df = df.copy()
    for column in categorical_columns:
        if column in normalized_df.columns:
            normalized_df[column] = series_to_categorical_tokens(normalized_df[column])
    return normalized_df

def safe_relative_error(real_value: float, synthetic_value: float, eps: float = 1e-10) -> float:
    return float(abs(real_value - synthetic_value) / (abs(real_value) + eps))

def split_real_reference_data(
    real_data: pd.DataFrame,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if len(real_data) < 2:
        return real_data.iloc[0:0].copy(), real_data.copy()

    rng = np.random.default_rng(random_state)
    shuffled_indices = rng.permutation(real_data.index.to_numpy())
    split_idx = len(shuffled_indices) // 2
    first_idx = shuffled_indices[:split_idx]
    second_idx = shuffled_indices[split_idx:]
    return (
        real_data.loc[first_idx].copy(),
        real_data.loc[second_idx].copy(),
    )

def categorical_frequency_summary(tokens: pd.Series) -> Dict[str, float]:
    freq = tokens.value_counts(normalize=True)
    return {str(category): float(value) for category, value in freq.items()}

def calculate_tvd_and_js(
    freq_a: pd.Series,
    freq_b: pd.Series,
) -> Tuple[float, float]:
    freq_a_smooth = (freq_a + 1e-10) / (freq_a + 1e-10).sum()
    freq_b_smooth = (freq_b + 1e-10) / (freq_b + 1e-10).sum()
    tvd = float(0.5 * np.abs(freq_a - freq_b).sum())
    js_distance = float(jensenshannon(freq_a_smooth, freq_b_smooth))
    return tvd, js_distance

def cramers_v(contingency_table: pd.DataFrame) -> float:
    if contingency_table.empty:
        return np.nan
    total = contingency_table.to_numpy().sum()
    if total <= 0:
        return np.nan
    chi2, _, _, _ = stats.chi2_contingency(contingency_table)
    min_dim = min(contingency_table.shape) - 1
    if min_dim <= 0:
        return 0.0
    return float(np.sqrt(chi2 / (total * min_dim)))

def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    valid_mask = categories.notna() & measurements.notna()
    if not valid_mask.any():
        return np.nan

    categories = categories.loc[valid_mask]
    measurements = measurements.loc[valid_mask]
    if len(measurements) < 2:
        return np.nan

    overall_mean = float(measurements.mean())
    total_ss = float(((measurements - overall_mean) ** 2).sum())
    if total_ss <= 0:
        return 0.0

    between_ss = 0.0
    for category_value, group in measurements.groupby(categories):
        if len(group) <= 0:
            continue
        group_mean = float(group.mean())
        between_ss += float(len(group) * ((group_mean - overall_mean) ** 2))

    explained_ratio = max(0.0, min(1.0, between_ss / total_ss))
    return float(np.sqrt(explained_ratio))
