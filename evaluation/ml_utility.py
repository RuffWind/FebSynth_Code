from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .base import MLUtilityEvaluator
from .context import EvaluationContext
from .ml_common import (
    build_binary_classification_pipelines,
    calculate_binary_classification_metrics,
    predict_with_youden_threshold,
    prepare_ml_feature_dataframe,
)
from .ml_task_utils import is_binary_dataset_usable, prepare_task_dataset
from .results import MetricResult
from .tasks import MachineLearningTask

class MachineLearningUtilityTaskEvaluator(MLUtilityEvaluator):
    def __init__(self, task: MachineLearningTask) -> None:
        self.task = task
        self.name = task.name

    def evaluate(self, context: EvaluationContext) -> MetricResult:
        def _retention_metric(trtr_value: float, tstr_value: float) -> float:
            return (
                float(tstr_value / trtr_value)
                if pd.notna(trtr_value) and pd.notna(tstr_value) and abs(trtr_value) > 1e-10
                else np.nan
            )

        def _gain_metric(trtr_value: float, mixed_value: float) -> float:
            return (
                float(mixed_value - trtr_value)
                if pd.notna(trtr_value) and pd.notna(mixed_value)
                else np.nan
            )

        base_feature_columns = [
            col for col in context.schema.feature_columns
            if col not in self.task.excluded_feature_columns
            and col in context.generation_real_data.columns
            and col in context.evaluation_real_data.columns
            and col in context.synthetic_data.columns
        ]
        continuous_columns = [col for col in base_feature_columns if col in context.schema.continuous_columns]
        categorical_columns = [col for col in base_feature_columns if col in context.schema.categorical_columns]
        warnings: List[str] = []

        if not base_feature_columns:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"status": "skipped", "reason": "无可用特征列"},
                tables={},
                metadata={"task": self.task.name},
                warnings=["无可用特征列，无法计算机器学习效用"],
            )

        real_train_X, real_train_y, real_train_meta = prepare_task_dataset(
            context.generation_real_data,
            self.task,
            base_feature_columns,
            "real_train",
        )
        synthetic_train_X, synthetic_train_y, synthetic_train_meta = prepare_task_dataset(
            context.synthetic_data,
            self.task,
            base_feature_columns,
            "synthetic_train",
        )
        real_test_X, real_test_y, real_test_meta = prepare_task_dataset(
            context.evaluation_real_data,
            self.task,
            base_feature_columns,
            "real_test",
        )

        test_usable, test_reason = is_binary_dataset_usable(
            real_test_X,
            real_test_y,
            self.task.minimum_rows,
            self.task.minimum_positive_rows,
        )
        if not test_usable:
            return MetricResult(
                name=self.name,
                domain=self.domain,
                level=self.level,
                summary={"status": "skipped", "reason": f"测试集不可用：{test_reason}"},
                tables={},
                metadata={
                    "task": self.task.name,
                    "target_column": self.task.target_column,
                    "subset_column": self.task.subset_column,
                    "subset_value": self.task.subset_value,
                },
                warnings=[f"测试集不可用：{test_reason}"],
            )

        mixed_train_X = pd.concat([real_train_X, synthetic_train_X], ignore_index=True)
        mixed_train_y = pd.concat([real_train_y, synthetic_train_y], ignore_index=True)
        scenario_datasets = {
            "trtr": (real_train_X, real_train_y, real_train_meta),
            "tstr": (synthetic_train_X, synthetic_train_y, synthetic_train_meta),
            "mixed": (
                mixed_train_X,
                mixed_train_y,
                {
                    "source": "mixed_train",
                    "rows_before_target_filter": int(len(real_train_X) + len(synthetic_train_X)),
                    "rows_after_target_filter": int(len(mixed_train_X)),
                    "positive_rows": int((mixed_train_y == 1).sum()),
                    "negative_rows": int((mixed_train_y == 0).sum()),
                    "has_target_column": True,
                },
            ),
        }

        model_pipelines = build_binary_classification_pipelines(
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            random_state=context.random_state,
        )
        detail_rows: List[Dict[str, Any]] = []
        summary_rows: List[Dict[str, Any]] = []
        scenario_metric_map: Dict[Tuple[str, str], Dict[str, float]] = {}

        for classifier_name, pipeline in model_pipelines.items():
            for scenario_name, (train_X, train_y, train_meta) in scenario_datasets.items():
                train_usable, train_reason = is_binary_dataset_usable(
                    train_X,
                    train_y,
                    self.task.minimum_rows,
                    self.task.minimum_positive_rows,
                )
                row = {
                    "task_name": self.task.name,
                    "target_column": self.task.target_column,
                    "subset_column": self.task.subset_column,
                    "subset_value": self.task.subset_value,
                    "classifier": classifier_name,
                    "scenario": scenario_name,
                    "train_rows": int(len(train_X)),
                    "train_positive_rows": int((train_y == 1).sum()),
                    "train_negative_rows": int((train_y == 0).sum()),
                    "test_rows": int(len(real_test_X)),
                    "test_positive_rows": int((real_test_y == 1).sum()),
                    "test_negative_rows": int((real_test_y == 0).sum()),
                    "status": "ok" if train_usable else "skipped",
                    "skip_reason": "" if train_usable else train_reason,
                    "auc": np.nan,
                    "precision": np.nan,
                    "recall": np.nan,
                    "specificity": np.nan,
                    "average_precision": np.nan,
                    "decision_threshold": np.nan,
                    "decision_threshold_source": "",
                }
                if train_usable:
                    prepared_test_X = prepare_ml_feature_dataframe(
                        real_test_X,
                        continuous_columns,
                        categorical_columns,
                    )
                    prepared_train_X = prepare_ml_feature_dataframe(
                        train_X,
                        continuous_columns,
                        categorical_columns,
                    )
                    probabilities, predictions, decision_threshold, decision_threshold_source = predict_with_youden_threshold(
                        pipeline,
                        prepared_train_X,
                        train_y,
                        prepared_test_X,
                        random_state=context.random_state,
                    )
                    metric_values = calculate_binary_classification_metrics(
                        y_true=real_test_y,
                        y_pred=predictions,
                        y_proba=probabilities,
                    )
                    row.update(metric_values)
                    row["decision_threshold"] = float(decision_threshold)
                    row["decision_threshold_source"] = decision_threshold_source
                    scenario_metric_map[(classifier_name, scenario_name)] = metric_values
                else:
                    warnings.append(
                        f"{self.task.name} / {classifier_name} / {scenario_name} 跳过：{train_reason}"
                    )
                detail_rows.append(row)

            trtr_metrics = scenario_metric_map.get((classifier_name, "trtr"), {})
            tstr_metrics = scenario_metric_map.get((classifier_name, "tstr"), {})
            mixed_metrics = scenario_metric_map.get((classifier_name, "mixed"), {})
            trtr_auc = trtr_metrics.get("auc", np.nan)
            tstr_auc = tstr_metrics.get("auc", np.nan)
            mixed_auc = mixed_metrics.get("auc", np.nan)
            trtr_precision = trtr_metrics.get("precision", np.nan)
            tstr_precision = tstr_metrics.get("precision", np.nan)
            mixed_precision = mixed_metrics.get("precision", np.nan)
            trtr_recall = trtr_metrics.get("recall", np.nan)
            tstr_recall = tstr_metrics.get("recall", np.nan)
            mixed_recall = mixed_metrics.get("recall", np.nan)
            trtr_specificity = trtr_metrics.get("specificity", np.nan)
            tstr_specificity = tstr_metrics.get("specificity", np.nan)
            mixed_specificity = mixed_metrics.get("specificity", np.nan)
            trtr_average_precision = trtr_metrics.get("average_precision", np.nan)
            tstr_average_precision = tstr_metrics.get("average_precision", np.nan)
            mixed_average_precision = mixed_metrics.get("average_precision", np.nan)
            summary_rows.append({
                "task_name": self.task.name,
                "target_column": self.task.target_column,
                "subset_column": self.task.subset_column,
                "subset_value": self.task.subset_value,
                "classifier": classifier_name,
                "feature_count": int(len(base_feature_columns)),
                "continuous_feature_count": int(len(continuous_columns)),
                "categorical_feature_count": int(len(categorical_columns)),
                "test_rows": int(len(real_test_X)),
                "test_positive_rows": int((real_test_y == 1).sum()),
                "test_negative_rows": int((real_test_y == 0).sum()),
                "trtr_auc": trtr_auc,
                "tstr_auc": tstr_auc,
                "mixed_auc": mixed_auc,
                "trtr_precision": trtr_precision,
                "tstr_precision": tstr_precision,
                "mixed_precision": mixed_precision,
                "trtr_recall": trtr_recall,
                "tstr_recall": tstr_recall,
                "mixed_recall": mixed_recall,
                "trtr_specificity": trtr_specificity,
                "tstr_specificity": tstr_specificity,
                "mixed_specificity": mixed_specificity,
                "trtr_average_precision": trtr_average_precision,
                "tstr_average_precision": tstr_average_precision,
                "mixed_average_precision": mixed_average_precision,
                "auc_retention": _retention_metric(trtr_auc, tstr_auc),
                "auc_gain": _gain_metric(trtr_auc, mixed_auc),
                "precision_retention": _retention_metric(trtr_precision, tstr_precision),
                "precision_gain": _gain_metric(trtr_precision, mixed_precision),
                "recall_retention": _retention_metric(trtr_recall, tstr_recall),
                "recall_gain": _gain_metric(trtr_recall, mixed_recall),
                "specificity_retention": _retention_metric(trtr_specificity, tstr_specificity),
                "specificity_gain": _gain_metric(trtr_specificity, mixed_specificity),
                "average_precision_retention": _retention_metric(trtr_average_precision, tstr_average_precision),
                "average_precision_gain": _gain_metric(trtr_average_precision, mixed_average_precision),
                "retention": _retention_metric(trtr_auc, tstr_auc),
                "gain": _gain_metric(trtr_auc, mixed_auc),
                "gain_ratio": (
                    float((mixed_auc / trtr_auc) - 1.0)
                    if pd.notna(trtr_auc) and pd.notna(mixed_auc) and abs(trtr_auc) > 1e-10
                    else np.nan
                ),
            })

        summary_df = pd.DataFrame(summary_rows)
        details_df = pd.DataFrame(detail_rows)
        summary = {
            "status": "ok",
            "task_name": self.task.name,
            "target_column": self.task.target_column,
            "subset_column": self.task.subset_column,
            "subset_value": self.task.subset_value,
            "feature_count": int(len(base_feature_columns)),
            "continuous_feature_count": int(len(continuous_columns)),
            "categorical_feature_count": int(len(categorical_columns)),
            "test_rows": int(len(real_test_X)),
            "test_positive_rows": int((real_test_y == 1).sum()),
            "test_negative_rows": int((real_test_y == 0).sum()),
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
                "task": self.task.name,
                "target_column": self.task.target_column,
                "subset_column": self.task.subset_column,
                "subset_value": self.task.subset_value,
                "feature_columns": base_feature_columns,
                "continuous_columns": continuous_columns,
                "categorical_columns": categorical_columns,
                "real_train_metadata": real_train_meta,
                "synthetic_train_metadata": synthetic_train_meta,
                "real_test_metadata": real_test_meta,
            },
            warnings=warnings,
        )
