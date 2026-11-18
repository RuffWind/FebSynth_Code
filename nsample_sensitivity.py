from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import pandas as pd
from output_localization import OutputLocalizer, resolve_default_output_root
from sklearn.metrics import f1_score

from evaluation.eval_config import (
    NSAMPLE_SENSITIVITY_N_MLU,
    NSAMPLE_SENSITIVITY_N_REPEATS,
    NSAMPLE_SENSITIVITY_SYNTHETIC_ONLY_SAMPLE_SIZES,
    NSAMPLE_SENSITIVITY_SYNTHETIC_MULTIPLIERS,
)
from evaluation.ml_common import (
    build_binary_classification_pipelines,
    calculate_binary_classification_metrics,
    predict_with_youden_threshold,
    prepare_ml_feature_dataframe,
)
from evaluation.ml_task_utils import is_binary_dataset_usable, prepare_task_dataset
from evaluation.tasks import DEFAULT_NSAMPLE_SENSITIVITY_TASKS, MachineLearningTask
from shared_generation import (
    CONTINUOUS_COLS,
    CATEGORICAL_COLS,
    DEFAULT_INPUT,
    DEFAULT_PIPELINE_OUTPUT_DIR,
    OUTCOME_COLS,
    build_group_summary_df,
    generate_synthetic_from_training_data,
    make_json_safe,
    split_real_train_test,
)
from utils.outcome_utils import format_outcome_distribution, get_outcome_joint_distribution

DEFAULT_OUTPUT_ROOT = os.path.join(DEFAULT_PIPELINE_OUTPUT_DIR, "nsample_sensitivity")
EXPERIMENT_GROUPS: Dict[str, Dict[str, Any]] = {
    "mixed_training": {
        "scenario_types": {"real_only", "mixed"},
        "generator_branches": {"mixed_dataset_b"},
    },
    "synthetic_only_training": {
        "scenario_types": {"real_only", "synthetic_only"},
        "generator_branches": {"synthetic_only_full_train_pool"},
    },
}

@dataclass(frozen=True)
class NSampleSensitivityConfig:
    model: str
    input_path: str
    output_root: str
    constraints_path: Optional[str]
    grouping_path: Optional[str]
    split_assignment_path: Optional[str]
    test_size: float
    random_state: int
    epochs: Optional[int]
    batch_size: Optional[int]
    cpu: bool
    cuda: bool
    copula_default_distribution: Optional[str]
    copula_no_enforce_min_max_values: bool
    copula_no_enforce_rounding: bool
    patience: int
    val_split: float
    output_language: str

def _save_performance_plot(
    summary_df: pd.DataFrame,
    output_dir: str,
    localizer: OutputLocalizer,
) -> Optional[str]:
    if summary_df.empty:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    localizer.configure_matplotlib()
    x_column = "synthetic_rows_requested_mean" if "synthetic_rows_requested_mean" in summary_df.columns else "synthetic_multiplier"
    plot_df = summary_df.sort_values(x_column).copy()

    fig, axes = plt.subplots(3, 2, figsize=(13.5, 12.5) if localizer.is_english else (14, 14))
    metric_specs = [
        ("auc_mean", "auc_std", "ROC AUC", "tab:blue"),
        ("average_precision_mean", "average_precision_std", "AP (PR AUC)", "tab:purple"),
        ("precision_mean", "precision_std", "Precision", "tab:green"),
        ("recall_mean", "recall_std", "Recall", "tab:red"),
        ("specificity_mean", "specificity_std", "Specificity", "tab:brown"),
        ("f1_score_mean", "f1_score_std", "F1", "tab:orange"),
    ]
    for ax, (metric_mean_column, metric_std_column, title, color) in zip(axes.flatten(), metric_specs):
        x_values = plot_df[x_column].to_numpy(dtype=float)
        mean_values = plot_df[metric_mean_column].to_numpy(dtype=float)
        std_values = plot_df[metric_std_column].fillna(0.0).to_numpy(dtype=float)
        ax.plot(
            x_values,
            mean_values,
            marker="o",
            linewidth=1.6 if localizer.is_english else 2,
            color=color,
        )
        ax.fill_between(
            x_values,
            mean_values - std_values,
            mean_values + std_values,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
        ax.set_xlabel("Synthetic samples used for training" if localizer.is_english else "训练使用的合成样本数")
        ax.set_ylabel(title)
        ax.set_title(title if localizer.is_english else f"{title} 随合成样本数变化")
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    output_path = os.path.join(output_dir, "performance_curve.png")
    fig.savefig(output_path, dpi=300 if localizer.is_english else 200, bbox_inches="tight")
    plt.close(fig)
    return output_path

class NSampleSensitivityAnalyzer:
    def __init__(
        self,
        config: NSampleSensitivityConfig,
        tasks: List[MachineLearningTask],
        output_dirs: Dict[str, str],
    ) -> None:
        self.config = config
        self.tasks = list(tasks)
        self.output_dirs = dict(output_dirs)
        self.localizer = OutputLocalizer(output_language=config.output_language)

    def _sample_small_dataset_a_indices(self, train_df: pd.DataFrame, repeat_index: int) -> pd.Index:
        if NSAMPLE_SENSITIVITY_N_MLU >= len(train_df):
            raise ValueError(
                "NSAMPLE_SENSITIVITY_N_MLU 必须严格小于去掉 holdout 后的训练池样本数，"
                f"当前 N_MLU={NSAMPLE_SENSITIVITY_N_MLU}，训练池样本数={len(train_df)}。"
            )
        return train_df.sample(
            n=int(NSAMPLE_SENSITIVITY_N_MLU),
            replace=False,
            random_state=self.config.random_state + repeat_index,
        ).index

    def _resolve_feature_columns(
        self,
        real_subset_df: pd.DataFrame,
        test_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        task: MachineLearningTask,
    ) -> List[str]:
        candidate_columns = [column for column in real_subset_df.columns if column not in task.excluded_feature_columns]
        return [
            column for column in candidate_columns
            if column in test_df.columns and column in synthetic_df.columns
        ]

    def _generate_synthetic_pool(
        self,
        train_df: pd.DataFrame,
        n_samples: int,
        repeat_index: int,
        *,
        branch_name: str,
        random_state_offset: int,
    ) -> tuple[pd.DataFrame, Dict[str, Any], Any]:
        synthetic_pool_df, generation_metadata, group_diag = generate_synthetic_from_training_data(
            train_df=train_df,
            model=self.config.model,
            constraints_path=self.config.constraints_path,
            grouping_path=self.config.grouping_path,
            n_samples=n_samples,
            random_state=self.config.random_state + repeat_index + random_state_offset,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            cpu=self.config.cpu,
            cuda=self.config.cuda,
            load_model=False,
            model_save_dir=None,
            persist_model=False,
            copula_default_distribution=self.config.copula_default_distribution,
            copula_no_enforce_min_max_values=self.config.copula_no_enforce_min_max_values,
            copula_no_enforce_rounding=self.config.copula_no_enforce_rounding,
            patience=self.config.patience,
            val_split=self.config.val_split,
        )
        return synthetic_pool_df, {
            "repeat_index": int(repeat_index),
            "generator_branch": branch_name,
            "generator_training_rows": int(len(train_df)),
            "metadata": generation_metadata,
        }, group_diag

    def run(self) -> None:
        real_data = pd.read_csv(self.config.input_path, encoding="utf-8-sig")
        for output_dir in self.output_dirs.values():
            os.makedirs(output_dir, exist_ok=True)

        mixed_multipliers = [float(value) for value in NSAMPLE_SENSITIVITY_SYNTHETIC_MULTIPLIERS]
        synthetic_only_sample_sizes = [int(value) for value in NSAMPLE_SENSITIVITY_SYNTHETIC_ONLY_SAMPLE_SIZES]
        if any(float(value) <= 0 for value in mixed_multipliers):
            raise ValueError("NSAMPLE_SENSITIVITY_SYNTHETIC_MULTIPLIERS 中的倍数必须都大于 0。")
        if any(int(value) <= 0 for value in synthetic_only_sample_sizes):
            raise ValueError("NSAMPLE_SENSITIVITY_SYNTHETIC_ONLY_SAMPLE_SIZES 中的样本量必须都大于 0。")
        if any(int(round(float(value) * NSAMPLE_SENSITIVITY_N_MLU)) <= 0 for value in mixed_multipliers):
            raise ValueError(
                "NSAMPLE_SENSITIVITY_SYNTHETIC_MULTIPLIERS 中的每个倍数都必须能生成至少 1 条合成样本。"
            )

        train_pool_df, holdout_test_df, split_metadata = split_real_train_test(
            data=real_data,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            outcome_columns=OUTCOME_COLS,
            split_assignment_path=self.config.split_assignment_path,
        )
        if NSAMPLE_SENSITIVITY_N_REPEATS < 1:
            raise ValueError("NSAMPLE_SENSITIVITY_N_REPEATS 必须至少为 1。")
        max_multiplier = max(mixed_multipliers) if mixed_multipliers else 0.0
        max_mixed_synthetic_size = int(round(max_multiplier * NSAMPLE_SENSITIVITY_N_MLU)) if mixed_multipliers else 0
        max_synthetic_only_size = max(synthetic_only_sample_sizes) if synthetic_only_sample_sizes else 0

        print("\n" + "=" * 80)
        print("开始执行生成样本数量敏感性分析")
        print("=" * 80)
        print(f"真实数据总样本数: {len(real_data)}")
        print(f"holdout 测试集样本数: {len(holdout_test_df)}")
        print(f"训练池样本数: {len(train_pool_df)}")
        print(f"切分来源: {split_metadata.get('split_source', 'unknown')}")
        if split_metadata.get('split_assignment_path'):
            print(f"split_assignment 文件: {split_metadata['split_assignment_path']}")
        print(f"小数据集 A 样本数 N_MLU: {NSAMPLE_SENSITIVITY_N_MLU}")
        print(f"重复次数: {NSAMPLE_SENSITIVITY_N_REPEATS}")
        print(f"混合实验启用: {'是' if mixed_multipliers else '否'}")
        if mixed_multipliers:
            print(f"最大合成样本倍数: {max_multiplier}")
            print(f"混合实验一次性生成的最大合成样本池大小: {max_mixed_synthetic_size}")
        print(f"纯合成实验启用: {'是' if synthetic_only_sample_sizes else '否'}")
        if synthetic_only_sample_sizes:
            print(f"纯合成实验最大样本量: {max_synthetic_only_size}")

        detail_rows_by_task: Dict[str, List[Dict[str, Any]]] = {
            task.name: [] for task in self.tasks
        }
        generation_metadata_list: List[Dict[str, Any]] = []
        group_summary_frames: List[pd.DataFrame] = []
        repeat_metadata_rows: List[Dict[str, Any]] = []
        scenario_specs = [{
            "scenario": "real_only",
            "scenario_type": "real_only",
            "synthetic_multiplier": 0.0,
            "synthetic_rows": 0,
            "pool_key": None,
        }]
        scenario_specs.extend([
            {
                "scenario": f"mixed_x{multiplier:g}",
                "scenario_type": "mixed",
                "synthetic_multiplier": float(multiplier),
                "synthetic_rows": int(round(float(multiplier) * NSAMPLE_SENSITIVITY_N_MLU)),
                "pool_key": "mixed",
            }
            for multiplier in mixed_multipliers
        ])
        scenario_specs.extend([
            {
                "scenario": f"synthetic_only_n{sample_size}",
                "scenario_type": "synthetic_only",
                "synthetic_multiplier": float(sample_size) / float(NSAMPLE_SENSITIVITY_N_MLU),
                "synthetic_rows": int(sample_size),
                "pool_key": "synthetic_only",
            }
            for sample_size in synthetic_only_sample_sizes
        ])

        for repeat_index in range(1, NSAMPLE_SENSITIVITY_N_REPEATS + 1):
            dataset_a_indices = self._sample_small_dataset_a_indices(train_pool_df, repeat_index)
            dataset_a_df = train_pool_df.loc[dataset_a_indices].reset_index(drop=True)
            dataset_b_df = train_pool_df.drop(index=dataset_a_indices).reset_index(drop=True)
            if dataset_b_df.empty:
                raise ValueError(f"第 {repeat_index} 次重复中，大数据集 B 为空，无法训练生成模型。")

            print(f"\n--- Repeat {repeat_index}/{NSAMPLE_SENSITIVITY_N_REPEATS} ---")
            print(f"小数据集 A 样本数: {len(dataset_a_df)}")
            print(f"大数据集 B 样本数: {len(dataset_b_df)}")

            train_outcome_columns = [column for column in OUTCOME_COLS if column in dataset_a_df.columns]
            if train_outcome_columns:
                train_outcome_dist = get_outcome_joint_distribution(dataset_a_df, train_outcome_columns)
                print(f"小数据集 A 结局变量联合分布 ({train_outcome_columns}):")
                for line in format_outcome_distribution(train_outcome_columns, train_outcome_dist):
                    print(f"  {line}")
            if train_outcome_columns:
                generator_outcome_dist = get_outcome_joint_distribution(dataset_b_df, train_outcome_columns)
                print(f"大数据集 B 结局变量联合分布 ({train_outcome_columns}):")
                for line in format_outcome_distribution(train_outcome_columns, generator_outcome_dist):
                    print(f"  {line}")

            repeat_metadata_rows.append({
                "repeat_index": int(repeat_index),
                "dataset_a_rows": int(len(dataset_a_df)),
                "dataset_b_rows": int(len(dataset_b_df)),
                "mixed_enabled": bool(mixed_multipliers),
                "synthetic_only_enabled": bool(synthetic_only_sample_sizes),
            })

            synthetic_pools: Dict[str, pd.DataFrame] = {}
            if mixed_multipliers:
                synthetic_pool_df, generation_metadata, group_diag = self._generate_synthetic_pool(
                    dataset_b_df,
                    max_mixed_synthetic_size,
                    repeat_index,
                    branch_name="mixed_dataset_b",
                    random_state_offset=0,
                )
                synthetic_pools["mixed"] = synthetic_pool_df
                generation_metadata_list.append(generation_metadata)
                if group_diag is not None:
                    group_summary_df = build_group_summary_df(group_diag)
                    group_summary_df.insert(0, "repeat_index", int(repeat_index))
                    group_summary_df.insert(1, "generator_branch", "mixed_dataset_b")
                    group_summary_frames.append(group_summary_df)

            if synthetic_only_sample_sizes:
                synthetic_pool_df, generation_metadata, group_diag = self._generate_synthetic_pool(
                    train_pool_df,
                    max_synthetic_only_size,
                    repeat_index,
                    branch_name="synthetic_only_full_train_pool",
                    random_state_offset=100000,
                )
                synthetic_pools["synthetic_only"] = synthetic_pool_df
                generation_metadata_list.append(generation_metadata)
                if group_diag is not None:
                    group_summary_df = build_group_summary_df(group_diag)
                    group_summary_df.insert(0, "repeat_index", int(repeat_index))
                    group_summary_df.insert(1, "generator_branch", "synthetic_only_full_train_pool")
                    group_summary_frames.append(group_summary_df)

            for task in self.tasks:
                feature_reference_df = synthetic_pools.get("mixed", synthetic_pools.get("synthetic_only", train_pool_df))
                feature_columns = self._resolve_feature_columns(
                    dataset_a_df,
                    holdout_test_df,
                    feature_reference_df,
                    task,
                )
                continuous_columns = [column for column in feature_columns if column in CONTINUOUS_COLS]
                categorical_columns = [column for column in feature_columns if column in CATEGORICAL_COLS]
                if not feature_columns:
                    raise ValueError(
                        f"第 {repeat_index} 次重复中，任务 `{task.name}` 没有可用于训练下游模型的特征列。"
                    )

                real_train_X, real_train_y, _ = prepare_task_dataset(
                    dataset_a_df,
                    task,
                    feature_columns,
                    f"real_train_dataset_a_repeat_{repeat_index}",
                )
                test_X, test_y, _ = prepare_task_dataset(
                    holdout_test_df,
                    task,
                    feature_columns,
                    "holdout_test",
                )
                test_usable, test_reason = is_binary_dataset_usable(
                    test_X,
                    test_y,
                    task.minimum_rows,
                    task.minimum_positive_rows,
                )
                if not test_usable:
                    raise ValueError(f"任务 `{task.name}` 的 holdout 测试集不可用：{test_reason}")

                prepared_test_X = prepare_ml_feature_dataframe(test_X, continuous_columns, categorical_columns)

                for scenario_index, scenario in enumerate(scenario_specs):
                    synthetic_rows = int(scenario["synthetic_rows"])
                    pool_key = scenario["pool_key"]
                    synthetic_pool_df = synthetic_pools.get(pool_key) if pool_key is not None else None
                    if synthetic_pool_df is not None and synthetic_rows > len(synthetic_pool_df):
                        raise ValueError(
                            f"第 {repeat_index} 次重复的场景 {scenario['scenario']} 需要 {synthetic_rows} 条合成样本，但最大样本池只有 {len(synthetic_pool_df)} 条。"
                        )

                    if scenario["scenario_type"] == "real_only":
                        train_X = real_train_X.copy()
                        train_y = real_train_y.copy()
                        effective_synthetic_train_rows = 0
                        effective_real_train_rows = int(len(real_train_X))
                    else:
                        if synthetic_pool_df is None:
                            raise ValueError(f"场景 `{scenario['scenario']}` 所需的合成样本池不存在。")
                        synthetic_subset = synthetic_pool_df.sample(
                            n=synthetic_rows,
                            replace=False,
                            random_state=self.config.random_state + repeat_index * 1000 + scenario_index,
                        ).reset_index(drop=True)
                        synthetic_train_X, synthetic_train_y, _ = prepare_task_dataset(
                            synthetic_subset,
                            task,
                            feature_columns,
                            f"synthetic_repeat_{repeat_index}_x{scenario['synthetic_multiplier']:g}",
                        )
                        effective_synthetic_train_rows = int(len(synthetic_train_X))
                        if scenario["scenario_type"] == "mixed":
                            train_X = pd.concat([real_train_X, synthetic_train_X], ignore_index=True)
                            train_y = pd.concat([real_train_y, synthetic_train_y], ignore_index=True)
                            effective_real_train_rows = int(len(real_train_X))
                        elif scenario["scenario_type"] == "synthetic_only":
                            train_X = synthetic_train_X.copy()
                            train_y = synthetic_train_y.copy()
                            effective_real_train_rows = 0
                        else:
                            raise ValueError(f"未知场景类型: {scenario['scenario_type']}")

                    train_usable, train_reason = is_binary_dataset_usable(
                        train_X,
                        train_y,
                        task.minimum_rows,
                        task.minimum_positive_rows,
                    )
                    row = {
                        "task_name": task.name,
                        "target_column": task.target_column,
                        "subset_column": task.subset_column,
                        "subset_value": task.subset_value,
                        "repeat_index": int(repeat_index),
                        "scenario": scenario["scenario"],
                        "scenario_type": scenario["scenario_type"],
                        "synthetic_multiplier": float(scenario["synthetic_multiplier"]),
                        "synthetic_rows_requested": synthetic_rows,
                        "synthetic_rows_effective": effective_synthetic_train_rows,
                        "dataset_a_rows": int(len(real_train_X)),
                        "dataset_b_rows": int(len(dataset_b_df)),
                        "real_rows_in_train": int(effective_real_train_rows),
                        "train_rows": int(len(train_X)),
                        "train_positive_rows": int((train_y == 1).sum()),
                        "train_negative_rows": int((train_y == 0).sum()),
                        "test_rows": int(len(test_X)),
                        "test_positive_rows": int((test_y == 1).sum()),
                        "test_negative_rows": int((test_y == 0).sum()),
                        "auc": None,
                        "average_precision": None,
                        "precision": None,
                        "recall": None,
                        "specificity": None,
                        "f1_score": None,
                        "decision_threshold": None,
                        "decision_threshold_source": "",
                        "status": "ok" if train_usable else "skipped",
                        "skip_reason": "" if train_usable else train_reason,
                    }
                    if train_usable:
                        xgboost_pipeline = build_binary_classification_pipelines(
                            continuous_columns=continuous_columns,
                            categorical_columns=categorical_columns,
                            random_state=self.config.random_state + repeat_index,
                        )["xgboost"]
                        prepared_train_X = prepare_ml_feature_dataframe(train_X, continuous_columns, categorical_columns)
                        probabilities, predictions, decision_threshold, decision_threshold_source = predict_with_youden_threshold(
                            xgboost_pipeline,
                            prepared_train_X,
                            train_y,
                            prepared_test_X,
                            random_state=self.config.random_state + repeat_index,
                        )
                        row.update(calculate_binary_classification_metrics(
                            y_true=test_y,
                            y_pred=predictions,
                            y_proba=probabilities,
                        ))
                        row["f1_score"] = float(f1_score(test_y, predictions, zero_division=0))
                        row["decision_threshold"] = float(decision_threshold)
                        row["decision_threshold_source"] = decision_threshold_source
                    detail_rows_by_task[task.name].append(row)

        for task in self.tasks:
            output_dir = self.output_dirs[task.name]
            detail_df = pd.DataFrame(detail_rows_by_task[task.name])
            total_repeat_count = int(detail_df["repeat_index"].nunique())
            all_group_summary_df = pd.concat(group_summary_frames, ignore_index=True) if group_summary_frames else pd.DataFrame()

            task_manifest = {
                **asdict(self.config),
                "n_mlu": int(NSAMPLE_SENSITIVITY_N_MLU),
                "n_repeats": int(NSAMPLE_SENSITIVITY_N_REPEATS),
                "synthetic_multipliers": mixed_multipliers,
                "synthetic_only_sample_sizes": synthetic_only_sample_sizes,
                "task": {
                    "name": task.name,
                    "target_column": task.target_column,
                    "subset_column": task.subset_column,
                    "subset_value": task.subset_value,
                    "excluded_feature_columns": list(task.excluded_feature_columns),
                    "minimum_rows": int(task.minimum_rows),
                    "minimum_positive_rows": int(task.minimum_positive_rows),
                },
                "all_configured_tasks": [
                    {
                        "name": configured_task.name,
                        "target_column": configured_task.target_column,
                        "subset_column": configured_task.subset_column,
                        "subset_value": configured_task.subset_value,
                    }
                    for configured_task in self.tasks
                ],
                "experiment_output_dirs": {},
            }

            for experiment_name, experiment_spec in EXPERIMENT_GROUPS.items():
                experiment_detail_df = detail_df[
                    detail_df["scenario_type"].isin(experiment_spec["scenario_types"])
                ].copy()
                if experiment_detail_df.empty:
                    continue

                experiment_output_dir = os.path.join(output_dir, experiment_name)
                os.makedirs(experiment_output_dir, exist_ok=True)
                self.localizer.save_csv(experiment_detail_df, os.path.join(experiment_output_dir, "details.csv"), index=False)

                ok_detail_df = experiment_detail_df[experiment_detail_df["status"] == "ok"].copy()
                if ok_detail_df.empty:
                    raise ValueError(
                        f"任务 `{task.name}` / 实验 `{experiment_name}` 的所有重复和场景都未得到可用结果，请检查配置。"
                    )

                summary_df = (
                    ok_detail_df.groupby(["scenario", "scenario_type"], as_index=False)
                    .agg(
                        task_name=("task_name", "first"),
                        target_column=("target_column", "first"),
                        subset_column=("subset_column", "first"),
                        subset_value=("subset_value", "first"),
                        synthetic_multiplier=("synthetic_multiplier", "first"),
                        synthetic_rows_requested_mean=("synthetic_rows_requested", "mean"),
                        synthetic_rows_effective_mean=("synthetic_rows_effective", "mean"),
                        dataset_a_rows_mean=("dataset_a_rows", "mean"),
                        dataset_b_rows_mean=("dataset_b_rows", "mean"),
                        real_rows_in_train_mean=("real_rows_in_train", "mean"),
                        train_rows_mean=("train_rows", "mean"),
                        train_positive_rows_mean=("train_positive_rows", "mean"),
                        train_negative_rows_mean=("train_negative_rows", "mean"),
                        test_rows_mean=("test_rows", "mean"),
                        test_positive_rows_mean=("test_positive_rows", "mean"),
                        test_negative_rows_mean=("test_negative_rows", "mean"),
                        auc_mean=("auc", "mean"),
                        auc_std=("auc", "std"),
                        average_precision_mean=("average_precision", "mean"),
                        average_precision_std=("average_precision", "std"),
                        precision_mean=("precision", "mean"),
                        precision_std=("precision", "std"),
                        recall_mean=("recall", "mean"),
                        recall_std=("recall", "std"),
                        specificity_mean=("specificity", "mean"),
                        specificity_std=("specificity", "std"),
                        f1_score_mean=("f1_score", "mean"),
                        f1_score_std=("f1_score", "std"),
                        successful_repeat_count=("repeat_index", "nunique"),
                    )
                )
                skip_counts = (
                    experiment_detail_df[experiment_detail_df["status"] != "ok"]
                    .groupby(["scenario", "scenario_type"])
                    .size()
                    .reset_index(name="skipped_repeat_count")
                )
                summary_df = summary_df.merge(
                    skip_counts,
                    on=["scenario", "scenario_type"],
                    how="left",
                )
                summary_df["skipped_repeat_count"] = summary_df["skipped_repeat_count"].fillna(0).astype(int)
                summary_df["total_repeat_count"] = total_repeat_count
                self.localizer.save_csv(summary_df, os.path.join(experiment_output_dir, "summary.csv"), index=False)

                plot_path = _save_performance_plot(summary_df.copy(), experiment_output_dir, self.localizer)

                self.localizer.save_json(make_json_safe(split_metadata), os.path.join(experiment_output_dir, "split_metadata.json"))
                self.localizer.save_json(make_json_safe(repeat_metadata_rows), os.path.join(experiment_output_dir, "repeat_metadata.json"))
                filtered_generation_metadata = [
                    row for row in generation_metadata_list
                    if row.get("generator_branch") in experiment_spec["generator_branches"]
                ]
                self.localizer.save_json(
                    make_json_safe(filtered_generation_metadata),
                    os.path.join(experiment_output_dir, "generation_metadata.json"),
                )

                if not all_group_summary_df.empty:
                    filtered_group_summary_df = all_group_summary_df[
                        all_group_summary_df["generator_branch"].isin(experiment_spec["generator_branches"])
                    ].copy()
                    if not filtered_group_summary_df.empty:
                        self.localizer.save_csv(
                            filtered_group_summary_df,
                            os.path.join(experiment_output_dir, "group_summary.csv"),
                            index=False,
                        )

                task_manifest["experiment_output_dirs"][experiment_name] = os.path.abspath(experiment_output_dir)

                experiment_config = {
                    **task_manifest,
                    "experiment_name": experiment_name,
                    "included_scenario_types": sorted(experiment_spec["scenario_types"]),
                    "summary_file": os.path.abspath(os.path.join(experiment_output_dir, "summary.csv")),
                    "details_file": os.path.abspath(os.path.join(experiment_output_dir, "details.csv")),
                    "performance_plot": os.path.abspath(plot_path) if plot_path else None,
                }
                self.localizer.save_json(make_json_safe(experiment_config), os.path.join(experiment_output_dir, "nsample_config.json"))

            self.localizer.save_json(make_json_safe(task_manifest), os.path.join(output_dir, "nsample_config.json"))

            print(f"\n任务 `{task.name}` 的样本量敏感性分析完成，结果已保存到: {os.path.abspath(output_dir)}")

def parse_args() -> NSampleSensitivityConfig:
    parser = argparse.ArgumentParser(description="生成样本数量敏感性分析")
    parser.add_argument("--model", type=str, required=True, help="生成器名称")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="输入真实数据路径")
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    parser.add_argument("--output_language", type=str, default="chinese", help="输出结果语言: chinese/english")
    parser.add_argument("--constraints", type=str, default=None, help="约束 JSON 路径")
    parser.add_argument("--grouping", type=str, default=None, help="分组 JSON 路径")
    parser.add_argument("--split_assignment", type=str, default=None, help="显式指定 train/test 切分的 split_assignment CSV 路径")
    parser.add_argument("--test_size", type=float, default=0.2, help="holdout 测试集占比，默认 0.2")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖默认批次大小")
    parser.add_argument("--copula_default_distribution", type=str, default=None, help="Copula 默认分布")
    parser.add_argument("--copula_no_enforce_min_max_values", action="store_true", help="关闭 Copula 数值范围约束")
    parser.add_argument("--copula_no_enforce_rounding", action="store_true", help="关闭 Copula 取整约束")
    parser.add_argument("--patience", type=int, default=50, help="早停 patience（仅 TabDDPM 使用）")
    parser.add_argument("--val_split", type=float, default=0.1, help="验证集比例（仅 TabDDPM 使用）")

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument("--cpu", action="store_true", help="使用 CPU")
    device_group.add_argument("--cuda", action="store_true", help="使用 GPU（默认）")

    args = parser.parse_args()
    if not 0 < args.test_size < 1:
        parser.error("--test_size 必须在 0 和 1 之间")

    resolved_output_root = (
        resolve_default_output_root(args.output_root, args.output_language)
        if args.output_root == DEFAULT_OUTPUT_ROOT else args.output_root
    )

    return NSampleSensitivityConfig(
        model=args.model,
        input_path=args.input,
        output_root=resolved_output_root,
        constraints_path=args.constraints,
        grouping_path=args.grouping,
        split_assignment_path=args.split_assignment,
        test_size=args.test_size,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cpu=args.cpu,
        cuda=args.cuda,
        copula_default_distribution=args.copula_default_distribution,
        copula_no_enforce_min_max_values=args.copula_no_enforce_min_max_values,
        copula_no_enforce_rounding=args.copula_no_enforce_rounding,
        patience=args.patience,
        val_split=args.val_split,
        output_language=args.output_language,
    )

def main() -> None:
    config = parse_args()
    model_output_root = os.path.join(config.output_root, config.model.lower())
    os.makedirs(model_output_root, exist_ok=True)
    task_count = len(DEFAULT_NSAMPLE_SENSITIVITY_TASKS)
    if task_count <= 0:
        raise ValueError("NSAMPLE_SENSITIVITY_TASK_CONFIGS 不能为空。")
    task_names = [task.name for task in DEFAULT_NSAMPLE_SENSITIVITY_TASKS]
    if len(set(task_names)) != len(task_names):
        raise ValueError("NSAMPLE_SENSITIVITY_TASK_CONFIGS 中的任务名 `name` 必须唯一。")

    output_dirs: Dict[str, str] = {}
    for task in DEFAULT_NSAMPLE_SENSITIVITY_TASKS:
        task_output_dir = (
            model_output_root
            if task_count == 1
            else os.path.join(model_output_root, task.name)
        )
        output_dirs[task.name] = task_output_dir
        print("\n" + "#" * 80)
        print(f"开始任务: {task.name}")
        print(
            f"目标列: {task.target_column}"
            + (
                f"；子集条件: {task.subset_column}={task.subset_value}"
                if task.subset_column is not None else ""
            )
        )
        print(f"结果目录: {os.path.abspath(task_output_dir)}")
        print("#" * 80)
    analyzer = NSampleSensitivityAnalyzer(config, list(DEFAULT_NSAMPLE_SENSITIVITY_TASKS), output_dirs)
    analyzer.run()

if __name__ == "__main__":
    main()
