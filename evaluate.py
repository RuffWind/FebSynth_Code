import argparse
import os

import pandas as pd
from output_localization import resolve_default_output_root

from evaluation import EvaluationSchema, TabularDataEvaluator
from shared_generation import CATEGORICAL_COLS, CONTINUOUS_COLS, OUTCOME_COLS

DEFAULT_REAL_INPUT = "/path/to/real_data.csv"
DEFAULT_SYNTHETIC_INPUT = "/path/to/synthetic_data.csv"
DEFAULT_OUTPUT_DIR = "/path/to/evaluation_output"
def main() -> None:
    parser = argparse.ArgumentParser(description="合成数据评估入口")
    parser.add_argument("--real", type=str, default=DEFAULT_REAL_INPUT, help="真实数据路径")
    parser.add_argument("--synthetic", type=str, default=DEFAULT_SYNTHETIC_INPUT, help="合成数据路径")
    parser.add_argument(
        "--train_real",
        type=str,
        default=None,
        help="用于训练生成器/ML utility 训练侧的真实数据路径；若不提供，则默认与 --real 相同（不推荐用于 ML utility）",
    )
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR, help="评估结果输出目录")
    parser.add_argument("--output_language", type=str, default="chinese", help="输出结果语言: chinese/english")
    args = parser.parse_args()
    resolved_output_dir = (
        resolve_default_output_root(args.output_dir, args.output_language)
        if args.output_dir == DEFAULT_OUTPUT_DIR else args.output_dir
    )

    evaluation_real = pd.read_csv(args.real, encoding="utf-8-sig")
    generation_real = pd.read_csv(args.train_real, encoding="utf-8-sig") if args.train_real else evaluation_real.copy()
    synthetic = pd.read_csv(args.synthetic, encoding="utf-8-sig")

    schema = EvaluationSchema.from_datasets(
        real_data=evaluation_real,
        synthetic_data=synthetic,
        continuous_candidates=CONTINUOUS_COLS,
        categorical_candidates=CATEGORICAL_COLS,
        outcome_columns=OUTCOME_COLS,
    )
    evaluator = TabularDataEvaluator(
        generation_real_data=generation_real,
        evaluation_real_data=evaluation_real,
        synthetic_data=synthetic,
        schema=schema,
    )
    evaluator.run_all(output_dir=resolved_output_dir, output_language=args.output_language)

    print("\n评估完成")
    print(f"输出目录: {os.path.abspath(resolved_output_dir)}")
    print("当前输出包含：相似性评价、机器学习效用、隐私评估，并按层级目录保存")

if __name__ == "__main__":
    main()
