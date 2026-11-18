
import argparse
import os
import time

import pandas as pd
from output_localization import OutputLocalizer, resolve_default_output_root

from constraints import load_constraints_with_reference_data, generate_with_constraints
from grouping import load_grouping, generate_by_groups
from generators import get_generator, list_generators
from shared_generation import (
    CATEGORICAL_COLS,
    CONTINUOUS_COLS,
    DEFAULT_INPUT,
    DEFAULT_GENERATION_OUTPUT_DIR,
    OUTCOME_COLS,
    build_group_summary_df,
    prepare_model_config,
    resolve_model_save_dir,
    resolve_generation_columns,
    validate_outcome_constraint_columns,
)
from utils.data_processor import DataProcessor
from utils.outcome_utils import format_outcome_distribution, get_outcome_joint_distribution

def main():
    parser = argparse.ArgumentParser(
        description='合成数据生成系统（增强版：支持生成约束）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('--model', type=str, help='生成器名称 (copula, ctgan, tvae, tabddpm)')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT, help=f'输入数据路径 (默认: {DEFAULT_INPUT})')
    parser.add_argument('--output', type=str, help=f'输出数据路径 (默认: {DEFAULT_GENERATION_OUTPUT_DIR}/synthetic_model.csv)')
    parser.add_argument('--output_language', type=str, default='chinese', help='输出结果语言: chinese/english (默认: chinese)')
    parser.add_argument('--constraints', type=str, default=None, help='约束JSON路径（可选）')
    parser.add_argument('--grouping', type=str, default=None, help='分组条件生成JSON路径（可选）')
    parser.add_argument('--n_samples', type=int, default=None, help='生成样本数（默认与输入相同）')
    parser.add_argument('--epochs', type=int, default=None, help='训练轮数（覆盖默认配置）')
    parser.add_argument('--batch_size', type=int, default=None, help='批次大小（覆盖默认配置）')
    parser.add_argument('--copula_default_distribution', type=str, default=None, help='Copula数值列默认分布（仅copula有效）')
    parser.add_argument('--copula_no_enforce_min_max_values', action='store_true', help='关闭Copula数值范围约束（仅copula有效）')
    parser.add_argument('--copula_no_enforce_rounding', action='store_true', help='关闭Copula数值精度/取整约束（仅copula有效）')
    parser.add_argument('--list_models', action='store_true', help='列出所有可用模型')

    device_group = parser.add_mutually_exclusive_group()
    device_group.add_argument('--cuda', action='store_true', help='使用GPU（如果可用）')
    device_group.add_argument('--cpu', action='store_true', help='使用CPU')

    parser.add_argument('--load_model', action='store_true', help='加载已保存的模型（如果存在）')
    parser.add_argument('--model_path', type=str, default=None, help='模型保存/加载目录（默认沿用 models/<model>/）')
    parser.add_argument('--patience', type=int, default=50, help='早停耐心值（默认50）')
    parser.add_argument('--val_split', type=float, default=0.1, help='验证集比例（默认0.1）')

    args = parser.parse_args()
    localizer = OutputLocalizer(output_language=args.output_language)

    if args.list_models:
        list_generators()
        return

    if not args.model:
        parser.error("请指定模型名称 (--model)")

    print("\n" + "=" * 80)
    print("合成数据生成系统（增强版）")
    print("=" * 80)

    print("\n[步骤 1/5] 加载数据")
    print("-" * 80)
    processor = DataProcessor()
    data = processor.load_data(args.input)

    print("\n[步骤 2/5] 数据预处理")
    print("-" * 80)

    missing_outcomes = [column for column in OUTCOME_COLS if column not in data.columns]
    if missing_outcomes:
        raise ValueError(f"结局变量不存在: {missing_outcomes}")

    column_types = processor.identify_column_types(
        data,
        continuous_cols=CONTINUOUS_COLS,
        categorical_cols=CATEGORICAL_COLS,
        outcome_cols=OUTCOME_COLS,
    )
    generation_columns = resolve_generation_columns(
        data.columns,
        column_types,
        OUTCOME_COLS,
    )

    print(f"\n本次目标生成列: {len(generation_columns)}")
    preview_columns = generation_columns[:10]
    print(f"  前{len(preview_columns)}列: {preview_columns}")
    if len(generation_columns) > 10:
        print(f"  ... 以及其他 {len(generation_columns) - 10} 列")

    data_processed = processor.preprocess_data(data, fit=True)
    data_processed_for_generation = data_processed[generation_columns].copy()

    is_valid, issues = processor.validate_data(data_processed)
    if not is_valid:
        print("⚠ 警告: 数据验证未完全通过，但将继续执行")

    outcome_dist = get_outcome_joint_distribution(data_processed, OUTCOME_COLS)
    print(f"\n结局变量联合分布 ({OUTCOME_COLS}):")
    for line in format_outcome_distribution(OUTCOME_COLS, outcome_dist):
        print(f"  {line}")

    print("\n[步骤 3/5] 创建模型与训练/加载")
    print("-" * 80)

    try:
        resolved_model_dir = resolve_model_save_dir(args.model, args.model_path)
        config = prepare_model_config(
            model=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            cpu=args.cpu,
            cuda=args.cuda,
            copula_default_distribution=args.copula_default_distribution,
            copula_no_enforce_min_max_values=args.copula_no_enforce_min_max_values,
            copula_no_enforce_rounding=args.copula_no_enforce_rounding,
            save_dir=resolved_model_dir,
        )
    except ValueError as e:
        print(f"✗ {e}")
        print("\n使用 --list_models 查看可用模型")
        return

    n_samples = args.n_samples if args.n_samples is not None else len(data)
    print(f"\n目标生成样本数: {n_samples}")
    print(f"模型目录: {resolved_model_dir}")

    extra_kwargs = {
        'outcome_columns': OUTCOME_COLS,
        'outcome_distribution': outcome_dist,
        'correct_outcome_distribution': True,
        'load_model': args.load_model,

        'early_stopping': True,
        'patience': args.patience,
        'val_split': args.val_split,
        'save_best': True,
    }

    if args.model.lower() == 'tabddpm':
        extra_kwargs.update({
            'continuous_columns': column_types['continuous'],
            'categorical_columns': column_types['categorical'],
        })

    print("\n[步骤 4/5] 生成数据")
    print("-" * 80)

    generator = None
    info = None
    group_diag = None

    if args.grouping:
        grouping_spec = load_grouping(args.grouping)
        constraint_spec = (
            load_constraints_with_reference_data(
                args.constraints,
                data,
                continuous_columns=column_types["continuous"],
            )
            if args.constraints else None
        )
        validate_outcome_constraint_columns(constraint_spec, OUTCOME_COLS)

        if args.constraints:
            output_filename = f"synthetic_{args.model.lower()}_grouping_constraints.csv"
        else:
            output_filename = f"synthetic_{args.model.lower()}_grouping.csv"

        start_group = time.time()
        synthetic_data, group_diag = generate_by_groups(
            model_name=args.model,
            model_config=config,
            data_raw=data,
            data_processed=data_processed,
            grouping_spec=grouping_spec,
            n_samples=n_samples,
            fit_kwargs=extra_kwargs,
            generate_kwargs=extra_kwargs,
            constraints_spec=constraint_spec,
            outcome_distribution_real=outcome_dist,
            generation_columns=generation_columns,
            verbose=True,
        )
        total_group_time = time.time() - start_group
        info = {
            'name': f"{args.model.upper()}Grouped",
            'training_time': group_diag.get('training_time', 0.0),
            'generation_time': group_diag.get('generation_time', total_group_time),
        }

        if group_diag.get('warnings'):
            unique_warnings = sorted(set(group_diag['warnings']))
            print(f"\n[分组条件生成] 警告(去重):")
            for w in unique_warnings[:10]:
                print(f"  - {w}")
            if len(unique_warnings) > 10:
                print(f"  ... 以及其他 {len(unique_warnings)-10} 条")
    else:
        try:
            generator = get_generator(args.model, **config)
        except Exception as e:
            print(f"✗ 创建生成器失败: {str(e)}")
            return

        if args.load_model:
            try:
                generator.load(checkpoint='best')
            except FileNotFoundError:
                print("\n⚠ 未找到已保存的模型，将重新训练")
                start_fit = time.time()
                generator.fit(data_processed_for_generation, **extra_kwargs)
                generator.training_time = time.time() - start_fit
        else:
            start_fit = time.time()
            generator.fit(data_processed_for_generation, **extra_kwargs)
            generator.training_time = time.time() - start_fit

        if args.constraints:
            output_filename = f"synthetic_{args.model.lower()}_constraints.csv"
            spec = load_constraints_with_reference_data(
                args.constraints,
                data,
                continuous_columns=column_types["continuous"],
            )
            validate_outcome_constraint_columns(spec, OUTCOME_COLS)
            generate_kwargs = dict(extra_kwargs)
            if spec.outcome is not None:
                generate_kwargs['correct_outcome_distribution'] = False
            start_gen = time.time()
            synthetic_data, diag = generate_with_constraints(
                generator=generator,
                n_samples=n_samples,
                spec=spec,
                outcome_distribution_real=outcome_dist,
                rng_seed=42,
                verbose=True,
                generate_kwargs=generate_kwargs,
            )
            generator.generation_time = time.time() - start_gen
            if diag.get('warnings'):
                unique_warnings = sorted(set(diag['warnings']))
                print(f"\n[约束生成] 警告(去重):")
                for w in unique_warnings[:10]:
                    print(f"  - {w}")
                if len(unique_warnings) > 10:
                    print(f"  ... 以及其他 {len(unique_warnings)-10} 条")
        else:
            output_filename = f"synthetic_{args.model.lower()}.csv"
            start_gen = time.time()
            synthetic_data = generator.generate(n_samples, **extra_kwargs)
            generator.generation_time = time.time() - start_gen
        info = generator.get_info()

    print("\n[步骤 5/5] 保存结果")
    print("-" * 80)

    if args.output:
        output_path = args.output
    else:
        default_output_dir = resolve_default_output_root(DEFAULT_GENERATION_OUTPUT_DIR, args.output_language)
        os.makedirs(default_output_dir, exist_ok=True)
        output_path = os.path.join(default_output_dir, output_filename)

    processor.save_data(localizer.localize_dataframe(synthetic_data), output_path)

    if args.grouping and group_diag is not None:
        output_root, _ = os.path.splitext(output_path)
        group_summary_path = f"{output_root}_group_summary.csv"
        group_summary_df = build_group_summary_df(group_diag)
        localizer.save_csv(group_summary_df, group_summary_path, index=False)
        print(f"✓ 分组总结已保存: {group_summary_path}")

    print(f"\n生成器信息:")
    print(f"  模型: {info['name']}")
    print(f"  训练时间: {info['training_time']:.2f}秒")
    print(f"  生成时间: {info['generation_time']:.2f}秒")

    metadata = processor.get_metadata(synthetic_data)
    print(f"\n数据统计:")
    print(f"  生成样本数: {metadata['shape'][0]}")
    print(f"  特征数: {metadata['shape'][1]}")
    if metadata['missing_values']:
        print(f"  含缺失值的列数: {len(metadata['missing_values'])}")

    print("\n" + "=" * 80)
    print("✓ 完成！")
    print(f"✓ 合成数据已保存至: {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
