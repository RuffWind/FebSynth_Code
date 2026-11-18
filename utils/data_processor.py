
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:

    def __init__(self):
        self.scaler = None
        self.continuous_cols = []
        self.categorical_cols = []
        self.outcome_cols = []
        self.column_info = {}

    def load_data(self, file_path: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(file_path)
            print(f"✓ 成功加载数据: {file_path}")
            print(f"  数据形状: {data.shape}")
            print(f"  列数: {len(data.columns)}")
            return data
        except Exception as e:
            raise ValueError(f"✗ 加载数据失败: {str(e)}")

    def identify_column_types(self,
                            data: pd.DataFrame,
                            continuous_cols: Optional[List[str]] = None,
                            categorical_cols: Optional[List[str]] = None,
                            outcome_cols: Optional[List[str]] = None) -> Dict[str, List[str]]:
        data_cols = set(data.columns)

        valid_outcomes = []
        if outcome_cols:
            missing_outcomes = [col for col in outcome_cols if col not in data_cols]
            if missing_outcomes:
                raise ValueError(f"结局变量不存在: {missing_outcomes}")
            valid_outcomes = [col for col in outcome_cols if col in data_cols]
            self.outcome_cols = valid_outcomes
            print(f"✓ 结局变量: {valid_outcomes}")
        else:
            self.outcome_cols = []

        valid_continuous = []
        if continuous_cols:
            specified_continuous = set(continuous_cols)
            valid_continuous = list(specified_continuous & data_cols)
            missing_continuous = specified_continuous - data_cols

            if missing_continuous:
                print(f"⚠ 警告: 以下{len(missing_continuous)}个连续型列在数据中不存在，已忽略:")
                for col in sorted(list(missing_continuous)[:5]):
                    print(f"  - {col}")
                if len(missing_continuous) > 5:
                    print(f"  ... 以及其他{len(missing_continuous)-5}列")

        valid_categorical = []
        if categorical_cols:
            specified_categorical = set(categorical_cols)
            valid_categorical = list(specified_categorical & data_cols)
            missing_categorical = specified_categorical - data_cols

            if missing_categorical:
                print(f"⚠ 警告: 以下{len(missing_categorical)}个类别型列在数据中不存在，已忽略:")
                for col in sorted(list(missing_categorical)[:5]):
                    print(f"  - {col}")
                if len(missing_categorical) > 5:
                    print(f"  ... 以及其他{len(missing_categorical)-5}列")

        self.continuous_cols = valid_continuous
        self.categorical_cols = valid_categorical

        print(f"\n✓ 列类型识别完成:")
        print(f"  有效连续型变量: {len(valid_continuous)}列")
        print(f"  有效类别型变量: {len(valid_categorical)}列")

        return {
            'continuous': valid_continuous,
            'categorical': valid_categorical,
            'outcomes': valid_outcomes,
        }

    def preprocess_data(self,
                       data: pd.DataFrame,
                       fit: bool = True) -> pd.DataFrame:
        data_processed = data.copy()

        print(f"\n开始预处理数据...")

        if self.continuous_cols:
            print(f"\n处理 {len(self.continuous_cols)} 个连续型变量:")

            conversion_issues = []

            for col in self.continuous_cols:
                if col not in data_processed.columns:
                    continue

                original_dtype = data_processed[col].dtype
                is_numeric = pd.api.types.is_numeric_dtype(original_dtype)

                if not is_numeric:
                    print(f"  转换 '{col}' (原类型: {original_dtype})")

                    converted = pd.to_numeric(data_processed[col], errors='coerce')

                    original_nan_count = data_processed[col].isnull().sum()
                    converted_nan_count = converted.isnull().sum()
                    new_nan_count = converted_nan_count - original_nan_count

                    if new_nan_count > 0:
                        conversion_issues.append({
                            'column': col,
                            'failed_count': new_nan_count,
                            'total_count': len(data_processed),
                            'failed_ratio': new_nan_count / len(data_processed) * 100
                        })
                        print(f"    ⚠ 有 {new_nan_count} 个值无法转换为数值 ({new_nan_count/len(data_processed)*100:.2f}%)")
                    else:
                        print(f"    ✓ 成功转换为数值类型")

                    data_processed[col] = converted

                if data_processed[col].isnull().any():
                    nan_count = data_processed[col].isnull().sum()
                    nan_ratio = nan_count / len(data_processed) * 100

                    if fit:

                        mean_val = data_processed[col].mean()

                        if pd.isna(mean_val):
                            mean_val = 0.0
                            print(f"    ⚠ '{col}' 所有值都是NaN，使用0填充")

                        self.column_info[col] = {'mean': mean_val}
                        data_processed[col].fillna(mean_val, inplace=True)
                        print(f"    填充 {nan_count} 个缺失值 ({nan_ratio:.2f}%)，均值: {mean_val:.4f}")
                    else:

                        mean_val = self.column_info.get(col, {}).get('mean', 0)
                        data_processed[col].fillna(mean_val, inplace=True)
                        print(f"    填充 {nan_count} 个缺失值，使用训练时的均值: {mean_val:.4f}")

            if conversion_issues:
                print(f"\n  ⚠ 数值转换警告汇总:")
                print(f"  共 {len(conversion_issues)} 个列存在转换失败:")
                for issue in conversion_issues[:5]:
                    print(f"    - {issue['column']}: {issue['failed_count']}个值 ({issue['failed_ratio']:.2f}%)")
                if len(conversion_issues) > 5:
                    print(f"    ... 以及其他 {len(conversion_issues)-5} 列")

                high_failure = [i for i in conversion_issues if i['failed_ratio'] > 10]
                if high_failure:
                    print(f"\n  ⚠ 建议检查以下列，转换失败率超过10%:")
                    for issue in high_failure:
                        print(f"    - {issue['column']}: {issue['failed_ratio']:.2f}%")
                    print(f"  这些列可能包含非数值的文本，考虑将它们移到类别型变量")

        if self.categorical_cols:
            print(f"\n类别型变量 ({len(self.categorical_cols)}列) 保持原样")

            categorical_missing = []
            for col in self.categorical_cols:
                if col in data_processed.columns:
                    nan_count = data_processed[col].isnull().sum()
                    if nan_count > 0:
                        nan_ratio = nan_count / len(data_processed) * 100
                        categorical_missing.append({
                            'column': col,
                            'nan_count': nan_count,
                            'nan_ratio': nan_ratio
                        })

            if categorical_missing:
                print(f"  {len(categorical_missing)} 个类别型列包含缺失值:")
                for item in categorical_missing[:5]:
                    print(f"    - {item['column']}: {item['nan_count']}个 ({item['nan_ratio']:.2f}%)")
                if len(categorical_missing) > 5:
                    print(f"    ... 以及其他 {len(categorical_missing)-5} 列")

        print(f"\n✓ 数据预处理完成")

        return data_processed

    def save_data(self, data: pd.DataFrame, file_path: str):
        try:

            import os
            output_dir = os.path.dirname(file_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)

            data.to_csv(file_path, index=False, encoding='utf-8-sig')
            print(f"✓ 数据已保存: {file_path}")
            print(f"  保存样本数: {len(data)}")
        except Exception as e:
            print(f"✗ 保存数据失败: {str(e)}")

    def get_metadata(self, data: pd.DataFrame) -> Dict:
        metadata = {
            'columns': list(data.columns),
            'dtypes': {col: str(dtype) for col, dtype in data.dtypes.items()},
            'continuous_columns': self.continuous_cols,
            'categorical_columns': self.categorical_cols,
            'shape': data.shape,
            'missing_values': {
                col: int(data[col].isnull().sum())
                for col in data.columns
                if data[col].isnull().any()
            }
        }

        return metadata

    def validate_data(self, data: pd.DataFrame) -> Tuple[bool, List[str]]:
        issues = []

        empty_cols = data.columns[data.isnull().all()].tolist()
        if empty_cols:
            issues.append(f"完全为空的列: {empty_cols}")

        if len(data) == 0:
            issues.append("数据集为空")

        for col in self.continuous_cols:
            if col in data.columns:
                if not pd.api.types.is_numeric_dtype(data[col]):
                    issues.append(f"连续型变量'{col}'不是数值类型 (当前类型: {data[col].dtype})")
                else:

                    if data[col].isnull().all():
                        issues.append(f"连续型变量'{col}'全是缺失值")

                    elif np.isinf(data[col]).any():
                        inf_count = np.isinf(data[col]).sum()
                        issues.append(f"连续型变量'{col}'包含{inf_count}个无穷值")

        for col in self.categorical_cols:
            if col in data.columns:
                unique_count = data[col].nunique(dropna=True)
                total_count = len(data)

                if unique_count > total_count * 0.5:
                    issues.append(
                        f"类别型变量'{col}'的唯一值过多 ({unique_count}/{total_count}={unique_count/total_count*100:.1f}%)，"
                        "可能更适合作为连续型变量或ID列"
                    )

        is_valid = len(issues) == 0

        if is_valid:
            print("\n✓ 数据验证通过")
        else:
            print("\n⚠ 数据验证发现以下问题:")
            for issue in issues:
                print(f"  - {issue}")

        return is_valid, issues
