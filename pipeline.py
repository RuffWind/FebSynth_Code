from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd
from output_localization import OutputLocalizer, resolve_default_output_root

from evaluation import EvaluationSchema, TabularDataEvaluator
from shared_generation import (
    CATEGORICAL_COLS,
    CONTINUOUS_COLS,
    DEFAULT_INPUT,
    DEFAULT_PIPELINE_OUTPUT_DIR,
    OUTCOME_COLS,
    build_group_summary_df,
    generate_synthetic_from_training_data,
    make_json_safe,
    resolve_model_save_dir,
    split_real_train_test,
)
from utils.outcome_utils import (
    format_outcome_distribution,
    get_outcome_joint_distribution,
)

@dataclass(frozen=True)
class PipelineConfig:
    model: str
    input_path: str
    output_root: str
    constraints_path: Optional[str]
    grouping_path: Optional[str]
    split_assignment_path: Optional[str]
    test_size: float
    n_samples: Optional[int]
    random_state: int
    epochs: Optional[int]
    batch_size: Optional[int]
    cpu: bool
    cuda: bool
    load_model: bool
    model_path: Optional[str]
    copula_default_distribution: Optional[str]
    copula_no_enforce_min_max_values: bool
    copula_no_enforce_rounding: bool
    patience: int
    val_split: float
    output_language: str

class SyntheticDataPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.localizer = OutputLocalizer(output_language=config.output_language)

    def _model_dir(self) -> str:
        return resolve_model_save_dir(self.config.model, self.config.model_path)

    def _split_train_test(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        return split_real_train_test(
            data=data,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            outcome_columns=OUTCOME_COLS,
            split_assignment_path=self.config.split_assignment_path,
        )

    def _generate_synthetic_data(
        self,
        train_df: pd.DataFrame,
        output_dir: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], str]:
        os.makedirs(output_dir, exist_ok=True)

        n_samples = self.config.n_samples if self.config.n_samples is not None else len(train_df)
        synthetic_data, generation_metadata, group_diag = generate_synthetic_from_training_data(
            train_df=train_df,
            model=self.config.model,
            constraints_path=self.config.constraints_path,
            grouping_path=self.config.grouping_path,
            n_samples=n_samples,
            random_state=self.config.random_state,
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            cpu=self.config.cpu,
            cuda=self.config.cuda,
            load_model=self.config.load_model,
            model_save_dir=self._model_dir(),
            persist_model=True,
            copula_default_distribution=self.config.copula_default_distribution,
            copula_no_enforce_min_max_values=self.config.copula_no_enforce_min_max_values,
            copula_no_enforce_rounding=self.config.copula_no_enforce_rounding,
            patience=self.config.patience,
            val_split=self.config.val_split,
        )

        synthetic_path = os.path.join(output_dir, "synthetic_data.csv")
        self.localizer.save_csv(synthetic_data, synthetic_path, index=False)
        print(f"✓ 数据已保存: {synthetic_path}")
        print(f"  保存样本数: {len(synthetic_data)}")

        if group_diag is not None:
            group_summary_df = build_group_summary_df(group_diag)
            self.localizer.save_csv(group_summary_df, os.path.join(output_dir, "group_summary.csv"), index=False)

        self.localizer.save_json(make_json_safe(generation_metadata), os.path.join(output_dir, "generation_metadata.json"))

        return synthetic_data, generation_metadata, synthetic_path

    def run(self) -> None:
        real_data = pd.read_csv(self.config.input_path, encoding="utf-8-sig")
        model_output_dir = os.path.join(self.config.output_root, self.config.model.lower())
        os.makedirs(model_output_dir, exist_ok=True)

        train_df, test_df, split_metadata = self._split_train_test(real_data)
        print("\n" + "=" * 80)
        print("开始执行训练集生成 + 质量评估流水线")
        print("=" * 80)
        print(f"真实数据总样本数: {len(real_data)}")
        print(f"训练集样本数: {len(train_df)}")
        print(f"测试集样本数: {len(test_df)}")
        print(f"测试集占比: {split_metadata['actual_test_ratio']:.4f}")
        print(f"按结局联合状态分层随机切分: {split_metadata['stratified_by_outcome_joint_state']}")
        print(f"切分来源: {split_metadata.get('split_source', 'unknown')}")
        if split_metadata.get("split_assignment_path"):
            print(f"split_assignment 文件: {split_metadata['split_assignment_path']}")

        train_outcome_dist = get_outcome_joint_distribution(train_df, OUTCOME_COLS)
        print(f"\n训练集结局变量联合分布 ({OUTCOME_COLS}):")
        for line in format_outcome_distribution(OUTCOME_COLS, train_outcome_dist):
            print(f"  {line}")

        test_outcome_dist = get_outcome_joint_distribution(test_df, OUTCOME_COLS)
        print(f"\n测试集结局变量联合分布 ({OUTCOME_COLS}):")
        for line in format_outcome_distribution(OUTCOME_COLS, test_outcome_dist):
            print(f"  {line}")

        synthetic_df, generation_metadata, synthetic_path = self._generate_synthetic_data(
            train_df=train_df,
            output_dir=model_output_dir,
        )

        schema = EvaluationSchema.from_datasets(
            real_data=train_df,
            synthetic_data=synthetic_df,
            continuous_candidates=CONTINUOUS_COLS,
            categorical_candidates=CATEGORICAL_COLS,
            outcome_columns=OUTCOME_COLS,
        )
        evaluator = TabularDataEvaluator(
            generation_real_data=train_df,
            evaluation_real_data=test_df,
            comparison_real_data=train_df,
            synthetic_data=synthetic_df,
            schema=schema,
        )
        eval_output_dir = os.path.join(model_output_dir, "evaluation")
        evaluator.run_all(
            output_dir=eval_output_dir,
            random_state=self.config.random_state,
            output_language=self.config.output_language,
        )

        self.localizer.save_json(make_json_safe(split_metadata), os.path.join(model_output_dir, "split_metadata.json"))

        self.localizer.save_json(make_json_safe(asdict(self.config)), os.path.join(model_output_dir, "pipeline_config.json"))

        run_summary = {
            "real_data_size": int(len(real_data)),
            "train_size": int(len(train_df)),
            "test_size": int(len(test_df)),
            "synthetic_size": int(len(synthetic_df)),
            "synthetic_path": os.path.abspath(synthetic_path),
            "evaluation_output_dir": os.path.abspath(eval_output_dir),
            "comparison_real_data": "train_split",
            "ml_test_real_data": "test_split",
            "generation_metadata_file": os.path.abspath(os.path.join(model_output_dir, "generation_metadata.json")),
            "split_metadata_file": os.path.abspath(os.path.join(model_output_dir, "split_metadata.json")),
        }
        self.localizer.save_json(make_json_safe(run_summary), os.path.join(model_output_dir, "run_summary.json"))

        print(f"\n全部完成，结果已保存到: {os.path.abspath(model_output_dir)}")

def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="训练集生成 + 质量评估流水线")
    parser.add_argument("--model", type=str, required=True, help="生成器名称")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT, help="输入真实数据路径")
    parser.add_argument("--output_root", type=str, default=DEFAULT_PIPELINE_OUTPUT_DIR, help="输出根目录")
    parser.add_argument("--output_language", type=str, default="chinese", help="输出结果语言: chinese/english")
    parser.add_argument("--constraints", type=str, default=None, help="约束 JSON 路径")
    parser.add_argument("--grouping", type=str, default=None, help="分组 JSON 路径")
    parser.add_argument("--split_assignment", type=str, default=None, help="显式指定 train/test 切分的 split_assignment CSV 路径")
    parser.add_argument("--test_size", type=float, default=0.2, help="测试集占比，默认 0.2")
    parser.add_argument("--n_samples", type=int, default=None, help="生成样本数，默认等于训练集样本数")
    parser.add_argument("--random_state", type=int, default=42, help="随机种子")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认训练轮数")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖默认批次大小")
    parser.add_argument("--load_model", action="store_true", help="优先加载流水线已保存模型，若不存在则重新训练")
    parser.add_argument("--model_path", type=str, default=None, help="模型保存/加载目录（默认沿用 models/<model>/）")
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
        if args.output_root == DEFAULT_PIPELINE_OUTPUT_DIR else args.output_root
    )

    return PipelineConfig(
        model=args.model,
        input_path=args.input,
        output_root=resolved_output_root,
        constraints_path=args.constraints,
        grouping_path=args.grouping,
        split_assignment_path=args.split_assignment,
        test_size=args.test_size,
        n_samples=args.n_samples,
        random_state=args.random_state,
        epochs=args.epochs,
        batch_size=args.batch_size,
        cpu=args.cpu,
        cuda=args.cuda,
        load_model=args.load_model,
        model_path=args.model_path,
        copula_default_distribution=args.copula_default_distribution,
        copula_no_enforce_min_max_values=args.copula_no_enforce_min_max_values,
        copula_no_enforce_rounding=args.copula_no_enforce_rounding,
        patience=args.patience,
        val_split=args.val_split,
        output_language=args.output_language,
    )

def main() -> None:
    config = parse_args()
    pipeline = SyntheticDataPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()
