from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from output_localization import OutputLocalizer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .base import GlobalSimilarityEvaluator
from .context import EvaluationContext
from .results import MetricResult
from .utils import (
    configure_matplotlib_for_chinese,
    normalize_categorical_dataframe,
    REAL_DATA_COLOR,
    SYNTHETIC_DATA_COLOR,
)

PRIMARY_COLOR = REAL_DATA_COLOR
SECONDARY_COLOR = SYNTHETIC_DATA_COLOR

def _resolve_feature_columns(context: EvaluationContext) -> Tuple[List[str], List[str], List[str]]:
    continuous_columns = list(context.valid_continuous_columns)
    categorical_columns = [
        col for col in context.schema.common_columns
        if col in context.comparison_real_data.columns
        and col in context.synthetic_data.columns
        and col not in continuous_columns
    ]
    feature_columns = continuous_columns + categorical_columns
    return feature_columns, continuous_columns, categorical_columns

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

def _prepare_global_features(
    df: pd.DataFrame,
    continuous_columns: List[str],
    categorical_columns: List[str],
) -> pd.DataFrame:
    prepared_df = df.copy()
    for column in continuous_columns:
        if column in prepared_df.columns:
            prepared_df[column] = pd.to_numeric(prepared_df[column], errors="coerce")
    prepared_df = normalize_categorical_dataframe(prepared_df, categorical_columns)
    return prepared_df

def save_umap_visualization_plots(
    embedding_df: pd.DataFrame,
    output_dir: str,
    localizer: OutputLocalizer | None = None,
) -> List[str]:
    if embedding_df.empty:
        return []

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("生成 UMAP 图需要 matplotlib，请先安装该依赖。") from exc
    localizer = localizer or OutputLocalizer()
    localizer.configure_matplotlib()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    saved_paths: List[str] = []
    real_df = embedding_df[embedding_df["dataset_label"] == "real"].copy()
    synthetic_df = embedding_df[embedding_df["dataset_label"] == "synthetic"].copy()

    def _save_plot(
        file_name: str,
        title: str,
        plot_df: pd.DataFrame | None = None,
        color: str = PRIMARY_COLOR,
        overlay: bool = False,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7.0, 6.0) if localizer.is_english else (7.5, 6.5))
        if overlay:
            ax.scatter(real_df["umap_1"], real_df["umap_2"], s=9, alpha=0.42, color=PRIMARY_COLOR, label="Real" if localizer.is_english else "real")
            ax.scatter(synthetic_df["umap_1"], synthetic_df["umap_2"], s=9, alpha=0.42, color=SECONDARY_COLOR, label="Synthetic" if localizer.is_english else "synthetic")
            ax.legend(frameon=False)
        else:
            if plot_df is None or plot_df.empty:
                plt.close(fig)
                return
            ax.scatter(plot_df["umap_1"], plot_df["umap_2"], s=9, alpha=0.45, color=color)
        ax.set_xlabel("Dimension 1" if localizer.is_english else "UMAP 1")
        ax.set_ylabel("Dimension 2" if localizer.is_english else "UMAP 2")
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()
        file_path = output_path / file_name
        fig.savefig(file_path, dpi=300 if localizer.is_english else 200, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(str(file_path))

    _save_plot(
        "real_umap.png",
        "UMAP: real data" if localizer.is_english else "UMAP: Real Data",
        plot_df=real_df,
        color=PRIMARY_COLOR,
    )
    _save_plot(
        "synthetic_umap.png",
        "UMAP: synthetic data" if localizer.is_english else "UMAP: Synthetic Data",
        plot_df=synthetic_df,
        color=SECONDARY_COLOR,
    )
    _save_plot(
        "overlay_umap.png",
        "UMAP: real and synthetic data" if localizer.is_english else "UMAP: Real vs Synthetic Overlay",
        overlay=True,
    )
    return saved_paths

class UMAPVisualizationEvaluator(GlobalSimilarityEvaluator):
    name = "umap_lisi"

    def _compute_mlisi(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
        n_neighbors: int = 30,
    ) -> Tuple[float, pd.DataFrame]:
        if len(embedding) < 3:
            return np.nan, pd.DataFrame()

        effective_neighbors = min(max(n_neighbors + 1, 2), len(embedding))
        neighbors = NearestNeighbors(n_neighbors=effective_neighbors, algorithm="auto")
        neighbors.fit(embedding)
        _, neighbor_indices = neighbors.kneighbors(embedding)

        rows: List[Dict[str, Any]] = []
        lisi_values: List[float] = []
        for point_index, row_indices in enumerate(neighbor_indices):
            local_indices = row_indices[row_indices != point_index]
            if len(local_indices) == 0:
                lisi = np.nan
                real_fraction = np.nan
                synthetic_fraction = np.nan
            else:
                neighbor_labels = labels[local_indices]
                real_fraction = float(np.mean(neighbor_labels == "real"))
                synthetic_fraction = float(np.mean(neighbor_labels == "synthetic"))
                simpson_index = (real_fraction ** 2) + (synthetic_fraction ** 2)
                lisi = float(1.0 / simpson_index) if simpson_index > 0 else np.nan

            lisi_values.append(lisi)
            rows.append({
                "point_index": int(point_index),
                "dataset_label": str(labels[point_index]),
                "neighbor_count": int(len(local_indices)),
                "real_neighbor_fraction": real_fraction,
                "synthetic_neighbor_fraction": synthetic_fraction,
                "lisi": lisi,
            })

        lisi_series = pd.Series(lisi_values, dtype=float)
        mlisi = float(lisi_series.mean()) if not lisi_series.dropna().empty else np.nan
        return mlisi, pd.DataFrame(rows)

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        feature_columns, continuous_columns, categorical_columns = _resolve_feature_columns(context)
        if not feature_columns:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"status": "skipped", "reason": "无共同特征列"},
                tables={},
                metadata={"method": "umap_lisi"},
                warnings=["无共同特征列，无法计算 UMAP 与 mLISI"],
            )

        real_features = context.comparison_real_data[feature_columns].copy()
        synthetic_features = context.synthetic_data[feature_columns].copy()
        real_features = _prepare_global_features(real_features, continuous_columns, categorical_columns)
        synthetic_features = _prepare_global_features(synthetic_features, continuous_columns, categorical_columns)
        if len(real_features) < 10 or len(synthetic_features) < 10:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"status": "skipped", "reason": "真实或合成样本不足，无法稳定计算 UMAP 与 mLISI"},
                tables={},
                metadata={"method": "umap_lisi"},
                warnings=["真实或合成样本不足，无法稳定计算 UMAP 与 mLISI"],
            )

        try:
            import umap
        except ImportError as exc:
            raise ImportError("计算 UMAP/mLISI 需要 umap-learn，请先安装该依赖。") from exc

        combined_features = pd.concat([real_features, synthetic_features], ignore_index=True)
        combined_labels = np.array(
            ["real"] * len(real_features) + ["synthetic"] * len(synthetic_features),
            dtype=object,
        )
        preprocessor = _build_preprocessor(continuous_columns, categorical_columns)
        real_feature_matrix = preprocessor.fit_transform(real_features)
        synthetic_feature_matrix = preprocessor.transform(synthetic_features)
        if hasattr(real_feature_matrix, "toarray"):
            real_feature_matrix = real_feature_matrix.toarray()
        if hasattr(synthetic_feature_matrix, "toarray"):
            synthetic_feature_matrix = synthetic_feature_matrix.toarray()
        real_feature_matrix = np.asarray(real_feature_matrix, dtype=float)
        synthetic_feature_matrix = np.asarray(synthetic_feature_matrix, dtype=float)

        umap_n_neighbors = int(min(30, max(2, len(real_features) - 1)))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=umap_n_neighbors,
            min_dist=0.1,
            metric="euclidean",
            random_state=context.random_state,
        )
        real_embedding = reducer.fit_transform(real_feature_matrix)
        synthetic_embedding = reducer.transform(synthetic_feature_matrix)
        embedding = np.vstack([real_embedding, synthetic_embedding])
        mlisi_mean, lisi_details_df = self._compute_mlisi(embedding, combined_labels, n_neighbors=30)

        embedding_df = pd.DataFrame({
            "dataset_label": combined_labels,
            "umap_1": embedding[:, 0].astype(float),
            "umap_2": embedding[:, 1].astype(float),
        })
        summary_row = {
            "status": "ok",
            "feature_count": int(len(feature_columns)),
            "continuous_feature_count": int(len(continuous_columns)),
            "categorical_feature_count": int(len(categorical_columns)),
            "real_reference_sample_count": int(len(real_features)),
            "synthetic_projected_sample_count": int(len(synthetic_features)),
            "umap_n_neighbors": umap_n_neighbors,
            "umap_min_dist": 0.1,
            "mlisi_mean": mlisi_mean,
            "mlisi_target_upper_bound": 2.0,
            "mlisi_gap_from_ideal": float(abs(2.0 - mlisi_mean)) if pd.notna(mlisi_mean) else np.nan,
        }
        summary_df = pd.DataFrame([summary_row])

        return MetricResult(
            name=self.name,
            domain=self.domain,
            level=self.level,
            summary=summary_row,
            tables={
                "summary": summary_df,
                "embedding": embedding_df,
                "lisi_details": lisi_details_df,
            },
            metadata={
                "method": "umap_lisi",
                "feature_columns": feature_columns,
                "categorical_normalization_applied": True,
                "umap_fit_reference": "real_only",
                "synthetic_embedding_mode": "transform_into_real_reference_space",
                "plot_file_patterns": [
                    "real_umap.png",
                    "synthetic_umap.png",
                    "overlay_umap.png",
                ],
            },
            warnings=[],
        )
