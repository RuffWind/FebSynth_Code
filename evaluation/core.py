from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional

import pandas as pd
from output_localization import OutputLocalizer

from .context import EvaluationContext
from .global_similarity import (
    UMAPVisualizationEvaluator,
    save_umap_visualization_plots,
)
from .ml_utility import MachineLearningUtilityTaskEvaluator
from .multivariate_similarity import (
    ConstraintViolationTestEvaluator,
    CrossTypeConditionalDistributionEvaluator,
    PairwiseCorrelationCoefficientEvaluator,
    save_pwcc_heatmap_triptych,
    save_ccd_distance_distribution_plots,
    save_ccd_dot_to_diagonal_plots,
    save_cvt_histogram_plots,
)
from .privacy import (
    AttributeInferenceEvaluator,
    DistanceToClosestRecordEvaluator,
    save_dcr_distribution_plot,
)
from .schema import EvaluationSchema
from .tasks import DEFAULT_ML_UTILITY_TASKS
from .univariate_similarity import (
    CategoricalUnivariateSimilarityEvaluator,
    ContinuousUnivariateSimilarityEvaluator,
    save_categorical_univariate_distribution_plots,
    save_continuous_univariate_distribution_plots,
)

def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def _write_csv(
    df: pd.DataFrame,
    output_dir: str,
    file_name: str,
    localizer: OutputLocalizer,
) -> None:
    if df.empty:
        return
    _ensure_dir(output_dir)
    localizer.save_csv(df, os.path.join(output_dir, file_name), index=False)

def _write_matrix_csv(
    df: pd.DataFrame,
    output_dir: str,
    file_name: str,
    localizer: OutputLocalizer,
) -> None:
    if df.empty:
        return
    _ensure_dir(output_dir)
    localizer.save_csv(df, os.path.join(output_dir, file_name), index=True, matrix=True)

class TabularDataEvaluator:

    def __init__(
        self,
        generation_real_data: pd.DataFrame,
        evaluation_real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        schema: EvaluationSchema,
        comparison_real_data: Optional[pd.DataFrame] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        context = EvaluationContext(
            generation_real_data=generation_real_data,
            evaluation_real_data=evaluation_real_data,
            synthetic_data=synthetic_data,
            schema=schema,
            comparison_real_data=comparison_real_data,
            metadata=metadata or {},
        )
        self.context = context
        self.generation_real_data = context.generation_real_data
        self.evaluation_real_data = context.evaluation_real_data
        self.comparison_real_data = context.comparison_real_data
        self.synthetic_data = context.synthetic_data
        self.schema = context.schema
        self.results: Dict[str, Any] = {
            "similarity": {
                "univariate": {},
                "multivariate": {},
                "global": {},
            },
            "ml_utility": {},
            "privacy": {},
        }

    def evaluate_continuous_distributions(self) -> pd.DataFrame:
        return self.evaluate_univariate_similarity()["continuous"]["metrics"]

    def evaluate_categorical_distributions(self) -> pd.DataFrame:
        return self.evaluate_univariate_similarity()["categorical"]["metrics"]

    def evaluate_univariate_similarity(self) -> Dict[str, Any]:
        cached = self.results.get("similarity", {}).get("univariate")
        if cached:
            return cached

        continuous_metric = ContinuousUnivariateSimilarityEvaluator().evaluate(self.context)
        categorical_metric = CategoricalUnivariateSimilarityEvaluator().evaluate(self.context)
        result = {
            "continuous": {
                "summary": continuous_metric.summary,
                "metrics": continuous_metric.tables.get("metrics", pd.DataFrame()),
                "summary_table": continuous_metric.tables.get("summary", pd.DataFrame()),
                "details": continuous_metric.metadata.get("details", {}),
                "warnings": continuous_metric.warnings,
            },
            "categorical": {
                "summary": categorical_metric.summary,
                "metrics": categorical_metric.tables.get("metrics", pd.DataFrame()),
                "summary_table": categorical_metric.tables.get("summary", pd.DataFrame()),
                "frequency_details": categorical_metric.tables.get("frequency_details", pd.DataFrame()),
                "details": categorical_metric.metadata.get("details", {}),
                "warnings": categorical_metric.warnings,
            },
        }
        self.results["similarity"]["univariate"] = result
        return result

    def evaluate_multivariate_similarity(self) -> Dict[str, Any]:
        cached = self.results.get("similarity", {}).get("multivariate")
        if cached:
            return cached

        pwcc_metric = PairwiseCorrelationCoefficientEvaluator().evaluate(self.context)
        ccd_metric = CrossTypeConditionalDistributionEvaluator().evaluate(self.context)
        cvt_metric = ConstraintViolationTestEvaluator().evaluate(self.context)
        result = {
            "pwcc": {
                "summary": pwcc_metric.summary,
                "summary_table": pwcc_metric.tables.get("summary", pd.DataFrame()),
                "real_matrix": pwcc_metric.tables.get("real_matrix", pd.DataFrame()),
                "synthetic_matrix": pwcc_metric.tables.get("synthetic_matrix", pd.DataFrame()),
                "absolute_diff_matrix": pwcc_metric.tables.get("absolute_diff_matrix", pd.DataFrame()),
                "pair_details": pwcc_metric.tables.get("pair_details", pd.DataFrame()),
                "variable_types": pwcc_metric.metadata.get("variable_types", {}),
                "warnings": pwcc_metric.warnings,
            },
            "ccd": {
                "summary": ccd_metric.summary,
                "summary_table": ccd_metric.tables.get("summary", pd.DataFrame()),
                "details": ccd_metric.tables.get("details", pd.DataFrame()),
                "categorical_grouping_columns": ccd_metric.metadata.get("categorical_grouping_columns", []),
                "continuous_grouping_columns": ccd_metric.metadata.get("continuous_grouping_columns", []),
                "plot_file_patterns": ccd_metric.metadata.get("plot_file_patterns", []),
                "warnings": ccd_metric.warnings,
            },
            "cvt": {
                "summary": cvt_metric.summary,
                "summary_table": cvt_metric.tables.get("summary", pd.DataFrame()),
                "range_columns": cvt_metric.metadata.get("range_columns", []),
                "difference_column_pairs": cvt_metric.metadata.get("difference_column_pairs", []),
                "plot_file_patterns": cvt_metric.metadata.get("plot_file_patterns", []),
                "warnings": cvt_metric.warnings,
            },
        }
        self.results["similarity"]["multivariate"] = result
        return result

    def evaluate_global_similarity(self) -> Dict[str, Any]:
        cached = self.results.get("similarity", {}).get("global")
        if cached:
            return cached

        umap_metric = UMAPVisualizationEvaluator().evaluate(self.context)
        result = {
            "umap_lisi": {
                "summary": umap_metric.summary,
                "summary_table": umap_metric.tables.get("summary", pd.DataFrame()),
                "embedding": umap_metric.tables.get("embedding", pd.DataFrame()),
                "lisi_details": umap_metric.tables.get("lisi_details", pd.DataFrame()),
                "feature_columns": umap_metric.metadata.get("feature_columns", []),
                "plot_file_patterns": umap_metric.metadata.get("plot_file_patterns", []),
                "warnings": umap_metric.warnings,
            },
        }
        self.results["similarity"]["global"] = result
        return result

    def evaluate_ml_utility(self) -> Dict[str, Any]:
        cached = self.results.get("ml_utility", {})
        if cached:
            return cached

        task_results: Dict[str, Any] = {}
        summary_frames = []
        detail_frames = []
        warnings: List[str] = []
        for task in DEFAULT_ML_UTILITY_TASKS:
            metric = MachineLearningUtilityTaskEvaluator(task).evaluate(self.context)
            task_results[task.name] = {
                "summary": metric.summary,
                "summary_table": metric.tables.get("summary", pd.DataFrame()),
                "details": metric.tables.get("details", pd.DataFrame()),
                "feature_columns": metric.metadata.get("feature_columns", []),
                "warnings": metric.warnings,
            }
            summary_table = metric.tables.get("summary", pd.DataFrame())
            details_table = metric.tables.get("details", pd.DataFrame())
            if not summary_table.empty:
                summary_frames.append(summary_table)
            if not details_table.empty:
                detail_frames.append(details_table)
            warnings.extend(metric.warnings)

        self.results["ml_utility"] = {
            "tasks": task_results,
            "summary_table": pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame(),
            "details": pd.concat(detail_frames, ignore_index=True) if detail_frames else pd.DataFrame(),
            "warnings": warnings,
        }
        return self.results["ml_utility"]

    def evaluate_privacy(self) -> Dict[str, Any]:
        cached = self.results.get("privacy", {})
        if cached:
            return cached

        dcr_metric = DistanceToClosestRecordEvaluator().evaluate(self.context)
        attribute_inference_metric = AttributeInferenceEvaluator().evaluate(self.context)
        self.results["privacy"] = {
            "dcr": {
                "summary": dcr_metric.summary,
                "summary_table": dcr_metric.tables.get("summary", pd.DataFrame()),
                "details": dcr_metric.tables.get("details", pd.DataFrame()),
                "feature_columns": dcr_metric.metadata.get("feature_columns", []),
                "plot_file_patterns": ["plots/dcr_distance_distribution.png"],
                "warnings": dcr_metric.warnings,
            },
            "attribute_inference": {
                "summary": attribute_inference_metric.summary,
                "summary_table": attribute_inference_metric.tables.get("summary", pd.DataFrame()),
                "details": attribute_inference_metric.tables.get("details", pd.DataFrame()),
                "feature_scores": attribute_inference_metric.tables.get("feature_scores", pd.DataFrame()),
                "encoded_features": attribute_inference_metric.tables.get("encoded_features", pd.DataFrame()),
                "feature_columns": attribute_inference_metric.metadata.get("feature_columns", []),
                "warnings": attribute_inference_metric.warnings,
            },
        }
        return self.results["privacy"]

    def run_all(
        self,
        output_dir: Optional[str] = None,
        random_state: int = 42,
        output_language: str = "chinese",
    ) -> Dict[str, Any]:
        self.context.random_state = random_state
        self.evaluate_univariate_similarity()
        self.evaluate_multivariate_similarity()
        self.evaluate_global_similarity()
        self.evaluate_ml_utility()
        self.evaluate_privacy()

        if output_dir:
            self.save_results(output_dir, output_language=output_language)

        return self.results

    def save_results(
        self,
        output_dir: str,
        output_language: str = "chinese",
    ) -> None:
        localizer = OutputLocalizer(output_language=output_language)
        _ensure_dir(output_dir)
        similarity_results = self.results.get("similarity", {})
        ml_utility_results = self.results.get("ml_utility", {})
        privacy_results = self.results.get("privacy", {})
        univariate_result = similarity_results.get("univariate", {})
        multivariate_result = similarity_results.get("multivariate", {})
        global_result = similarity_results.get("global", {})

        active_similarity_modules = ["similarity.univariate"]
        if multivariate_result.get("pwcc"):
            active_similarity_modules.append("similarity.multivariate.pwcc")
        ccd_result = multivariate_result.get("ccd", {})
        if ccd_result.get("categorical_grouping_columns") or ccd_result.get("continuous_grouping_columns"):
            active_similarity_modules.append("similarity.multivariate.ccd")
        cvt_result = multivariate_result.get("cvt", {})
        if cvt_result.get("range_columns") or cvt_result.get("difference_column_pairs"):
            active_similarity_modules.append("similarity.multivariate.cvt")
        if global_result.get("umap_lisi"):
            active_similarity_modules.append("similarity.global.umap_lisi")
        active_ml_utility_tasks = sorted(list(ml_utility_results.get("tasks", {}).keys()))
        active_privacy_modules = sorted(list(privacy_results.keys()))

        metadata = {
            "generation_real_shape": list(self.generation_real_data.shape),
            "evaluation_real_shape": list(self.evaluation_real_data.shape),
            "comparison_real_shape": list(self.comparison_real_data.shape),
            "synthetic_shape": list(self.synthetic_data.shape),
            "schema": asdict(self.schema),
            "active_similarity_modules": active_similarity_modules,
            "active_ml_utility_tasks": active_ml_utility_tasks,
            "active_privacy_modules": active_privacy_modules,
            "ccd_grouping_columns": (
                list(ccd_result.get("categorical_grouping_columns", []))
                + [str(spec.get("column")) for spec in ccd_result.get("continuous_grouping_columns", [])]
            ),
            "ccd_categorical_grouping_columns": ccd_result.get("categorical_grouping_columns", []),
            "ccd_continuous_grouping_columns": ccd_result.get("continuous_grouping_columns", []),
            "ccd_enabled": bool(
                ccd_result.get("categorical_grouping_columns") or ccd_result.get("continuous_grouping_columns")
            ),
        }
        localizer.save_json(metadata, os.path.join(output_dir, "evaluation_metadata.json"))

        similarity_root = _ensure_dir(os.path.join(output_dir, "similarity"))

        continuous_dir = _ensure_dir(os.path.join(similarity_root, "univariate", "continuous"))
        continuous_result = univariate_result.get("continuous", {})
        _write_csv(continuous_result.get("metrics", pd.DataFrame()), continuous_dir, "metrics.csv", localizer)
        _write_csv(continuous_result.get("summary_table", pd.DataFrame()), continuous_dir, "summary.csv", localizer)
        continuous_plot_dir = os.path.join(continuous_dir, "plots")
        continuous_plot_paths = save_continuous_univariate_distribution_plots(
            self.comparison_real_data,
            self.synthetic_data,
            list(self.schema.continuous_columns),
            continuous_plot_dir,
            localizer=localizer,
        )
        if continuous_plot_paths:
            self.results.setdefault("artifacts", {})["continuous_univariate_plot_paths"] = continuous_plot_paths

        categorical_dir = _ensure_dir(os.path.join(similarity_root, "univariate", "categorical"))
        categorical_result = univariate_result.get("categorical", {})
        _write_csv(categorical_result.get("metrics", pd.DataFrame()), categorical_dir, "metrics.csv", localizer)
        _write_csv(categorical_result.get("summary_table", pd.DataFrame()), categorical_dir, "summary.csv", localizer)
        _write_csv(categorical_result.get("frequency_details", pd.DataFrame()), categorical_dir, "frequency_details.csv", localizer)
        categorical_plot_dir = os.path.join(categorical_dir, "plots")
        categorical_plot_columns = list(self.schema.categorical_columns) + [
            column for column in self.schema.outcome_columns
            if column not in self.schema.categorical_columns
        ]
        categorical_plot_paths = save_categorical_univariate_distribution_plots(
            self.comparison_real_data,
            self.synthetic_data,
            categorical_plot_columns,
            categorical_plot_dir,
            localizer=localizer,
        )
        if categorical_plot_paths:
            self.results.setdefault("artifacts", {})["categorical_univariate_plot_paths"] = categorical_plot_paths

        pwcc_dir = _ensure_dir(os.path.join(similarity_root, "multivariate", "pwcc"))
        pwcc_result = multivariate_result.get("pwcc", {})
        _write_csv(pwcc_result.get("summary_table", pd.DataFrame()), pwcc_dir, "summary.csv", localizer)
        _write_matrix_csv(pwcc_result.get("real_matrix", pd.DataFrame()), pwcc_dir, "real_matrix.csv", localizer)
        _write_matrix_csv(pwcc_result.get("synthetic_matrix", pd.DataFrame()), pwcc_dir, "synthetic_matrix.csv", localizer)
        _write_matrix_csv(pwcc_result.get("absolute_diff_matrix", pd.DataFrame()), pwcc_dir, "absolute_diff_matrix.csv", localizer)
        _write_csv(pwcc_result.get("pair_details", pd.DataFrame()), pwcc_dir, "pair_details.csv", localizer)
        pwcc_plot_paths = save_pwcc_heatmap_triptych(
            pwcc_result.get("real_matrix", pd.DataFrame()),
            pwcc_result.get("synthetic_matrix", pd.DataFrame()),
            pwcc_result.get("absolute_diff_matrix", pd.DataFrame()),
            pwcc_dir,
            localizer=localizer,
        )
        if pwcc_plot_paths:
            self.results.setdefault("artifacts", {})["pwcc_plot_paths"] = pwcc_plot_paths

        ccd_dir = _ensure_dir(os.path.join(similarity_root, "multivariate", "ccd"))
        _write_csv(ccd_result.get("summary_table", pd.DataFrame()), ccd_dir, "summary.csv", localizer)
        ccd_details_df = ccd_result.get("details", pd.DataFrame())
        _write_csv(ccd_details_df, ccd_dir, "details.csv", localizer)
        if not ccd_details_df.empty:
            ccd_plot_dir = os.path.join(ccd_dir, "plots")
            ccd_plot_paths = save_ccd_dot_to_diagonal_plots(ccd_details_df, ccd_plot_dir, localizer=localizer)
            ccd_plot_paths.extend(save_ccd_distance_distribution_plots(ccd_details_df, ccd_plot_dir, localizer=localizer))
            self.results.setdefault("artifacts", {})["ccd_plot_paths"] = ccd_plot_paths

        cvt_dir = _ensure_dir(os.path.join(similarity_root, "multivariate", "cvt"))
        cvt_result = multivariate_result.get("cvt", {})
        _write_csv(cvt_result.get("summary_table", pd.DataFrame()), cvt_dir, "summary.csv", localizer)
        if cvt_result.get("range_columns") or cvt_result.get("difference_column_pairs"):
            cvt_plot_dir = os.path.join(cvt_dir, "plots")
            self.results.setdefault("artifacts", {})["cvt_plot_paths"] = save_cvt_histogram_plots(
                self.context.synthetic_data,
                list(cvt_result.get("range_columns", [])),
                [tuple(pair) for pair in cvt_result.get("difference_column_pairs", [])],
                cvt_plot_dir,
                localizer=localizer,
            )

        umap_dir = _ensure_dir(os.path.join(similarity_root, "global", "umap_lisi"))
        umap_result = global_result.get("umap_lisi", {})
        _write_csv(umap_result.get("summary_table", pd.DataFrame()), umap_dir, "summary.csv", localizer)
        embedding_df = umap_result.get("embedding", pd.DataFrame())
        _write_csv(embedding_df, umap_dir, "embedding.csv", localizer)
        _write_csv(umap_result.get("lisi_details", pd.DataFrame()), umap_dir, "lisi_details.csv", localizer)
        if not embedding_df.empty:
            umap_plot_dir = os.path.join(umap_dir, "plots")
            self.results.setdefault("artifacts", {})["umap_plot_paths"] = save_umap_visualization_plots(
                embedding_df,
                umap_plot_dir,
                localizer=localizer,
            )

        ml_utility_root = _ensure_dir(os.path.join(output_dir, "ml_utility"))
        _write_csv(ml_utility_results.get("summary_table", pd.DataFrame()), ml_utility_root, "summary.csv", localizer)
        _write_csv(ml_utility_results.get("details", pd.DataFrame()), ml_utility_root, "details.csv", localizer)
        for task_name, task_result in ml_utility_results.get("tasks", {}).items():
            task_dir = _ensure_dir(os.path.join(ml_utility_root, task_name))
            _write_csv(task_result.get("summary_table", pd.DataFrame()), task_dir, "summary.csv", localizer)
            _write_csv(task_result.get("details", pd.DataFrame()), task_dir, "details.csv", localizer)

        privacy_root = _ensure_dir(os.path.join(output_dir, "privacy"))
        dcr_result = privacy_results.get("dcr", {})
        dcr_dir = _ensure_dir(os.path.join(privacy_root, "dcr"))
        _write_csv(dcr_result.get("summary_table", pd.DataFrame()), dcr_dir, "summary.csv", localizer)
        dcr_details_df = dcr_result.get("details", pd.DataFrame())
        _write_csv(dcr_details_df, dcr_dir, "details.csv", localizer)
        if not dcr_details_df.empty:
            dcr_plot_dir = os.path.join(dcr_dir, "plots")
            self.results.setdefault("artifacts", {})["dcr_plot_paths"] = save_dcr_distribution_plot(
                dcr_details_df,
                dcr_plot_dir,
                localizer=localizer,
            )

        attribute_inference_result = privacy_results.get("attribute_inference", {})
        attribute_inference_dir = _ensure_dir(os.path.join(privacy_root, "attribute_inference"))
        _write_csv(
            attribute_inference_result.get("summary_table", pd.DataFrame()),
            attribute_inference_dir,
            "summary.csv",
            localizer,
        )
        _write_csv(
            attribute_inference_result.get("details", pd.DataFrame()),
            attribute_inference_dir,
            "details.csv",
            localizer,
        )
        _write_csv(
            attribute_inference_result.get("feature_scores", pd.DataFrame()),
            attribute_inference_dir,
            "feature_scores.csv",
            localizer,
        )
        _write_csv(
            attribute_inference_result.get("encoded_features", pd.DataFrame()),
            attribute_inference_dir,
            "encoded_features.csv",
            localizer,
        )
