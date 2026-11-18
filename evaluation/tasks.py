from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from .eval_config import ML_UTILITY_TASK_CONFIGS, NSAMPLE_SENSITIVITY_TASK_CONFIGS

@dataclass(frozen=True)
class MachineLearningTask:
    name: str
    target_column: str
    subset_column: Optional[str] = None
    subset_value: Optional[Any] = None
    excluded_feature_columns: Tuple[str, ...] = ()
    minimum_rows: int = 20
    minimum_positive_rows: int = 5

def build_machine_learning_task_from_config(config: dict[str, Any]) -> MachineLearningTask:
    if "name" not in config or "target_column" not in config:
        raise ValueError("机器学习任务配置至少需要包含 `name` 和 `target_column`。")

    excluded_feature_columns = tuple(str(column) for column in config.get("excluded_feature_columns", []))
    minimum_rows = int(config.get("minimum_rows", 20))
    minimum_positive_rows = int(config.get("minimum_positive_rows", 5))
    if minimum_rows <= 0:
        raise ValueError("机器学习任务配置中的 `minimum_rows` 必须大于 0。")
    if minimum_positive_rows <= 0:
        raise ValueError("机器学习任务配置中的 `minimum_positive_rows` 必须大于 0。")

    return MachineLearningTask(
        name=str(config["name"]),
        target_column=str(config["target_column"]),
        subset_column=None if config.get("subset_column") in [None, ""] else str(config.get("subset_column")),
        subset_value=config.get("subset_value"),
        excluded_feature_columns=excluded_feature_columns,
        minimum_rows=minimum_rows,
        minimum_positive_rows=minimum_positive_rows,
    )

DEFAULT_ML_UTILITY_TASKS: Tuple[MachineLearningTask, ...] = tuple(
    build_machine_learning_task_from_config(config)
    for config in ML_UTILITY_TASK_CONFIGS
)

DEFAULT_NSAMPLE_SENSITIVITY_TASKS: Tuple[MachineLearningTask, ...] = tuple(
    build_machine_learning_task_from_config(config)
    for config in NSAMPLE_SENSITIVITY_TASK_CONFIGS
)

DEFAULT_NSAMPLE_SENSITIVITY_TASK = DEFAULT_NSAMPLE_SENSITIVITY_TASKS[0]
