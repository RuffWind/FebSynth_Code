from __future__ import annotations

import os
import tempfile
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.model_config import get_config
from constraints import generate_with_constraints, load_constraints_with_reference_data
from generators import get_generator
from grouping import generate_by_groups, load_grouping
from utils.data_processor import DataProcessor
from utils.outcome_utils import (
    get_outcome_joint_distribution,
)
from utils.split_assignment import (
    apply_split_assignment,
    generate_split_assignment,
    load_split_assignment,
)

DEFAULT_INPUT = "/path/to/real_data.csv"
DEFAULT_GENERATION_OUTPUT_DIR = "/path/to/generation_output"
DEFAULT_PIPELINE_OUTPUT_DIR = "/path/to/pipeline_output"

OUTCOME_COLS = ["住院", "icu"]

CONTINUOUS_COLS = [
    "C反应蛋白", "嗜碱性粒细胞百分数", "嗜酸性粒细胞百分数", "中性粒细胞百分数", "中性粒细胞绝对值",
    "单核细胞百分数", "单核细胞绝对值", "平均红细胞体积", "平均血红蛋白浓度", "平均血红蛋白量", "淋巴细胞百分数",
    "淋巴细胞绝对值", "白细胞计数", "血小板比容", "血小板计数", "血红蛋白", "血细胞比容", "体重",
    "热峰", "发热时间", "计算年龄",
]

CATEGORICAL_COLS = [
    "乏力", "低体温", "体重下降", "便秘", "便脓", "便血", "入睡困难", "关节痛", "关节红肿", "冷汗", "凝视",
    "反跳痛", "发绀", "口吐白沫", "口腔疱疹", "吞咽困难", "吸气性呼吸困难", "呃逆", "呕吐", "呕血", "呛奶",
    "呻吟", "呼吸暂停", "呼吸窘迫", "呼吸音异常", "呼气性呼吸困难", "咯血", "咳嗽", "咽干/咽喉痛", "咽部充血",
    "唇干", "唇红", "啰音", "喷射性呕吐", "嗜睡", "囟门凹陷", "囟门隆起", "声嘶", "多汗/盗汗", "多食", "多饮",
    "大便失禁", "头痛", "头颅血肿", "寒战", "小便失禁", "少尿", "尖叫", "尿急", "尿痛", "尿色异常", "尿量增多",
    "尿频", "局部皮温升高", "心律失常", "心悸", "心脏杂音", "恶心", "惊厥", "意识丧失", "意识模糊", "意识障碍",
    "手足硬肿", "手部皮疹", "打鼾", "拒绝进食", "指趾端脱皮", "排尿困难", "收缩期杂音", "斜视", "无尿", "昏迷",
    "晕眩", "气促", "水肿", "泡沫尿", "流泪", "流涎", "浅部淋巴结肿大", "消瘦", "激惹", "牙关紧闭", "牙龈出血",
    "犬吠样咳嗽", "畏光", "畏寒", "痉挛样咳嗽", "瘙痒", "白陶土样便", "皮下血肿", "皮疹", "皮肤创口", "皮肤干燥",
    "皮肤色素沉着", "皮肤苍白", "皮肤黏膜出血", "眼痛", "眼睑水肿", "神萎", "结膜充血", "耳痛", "肌张力减弱",
    "肌张力增高", "肌紧张", "肌肉酸痛", "肝大", "肢端凉", "胸闷/胸痛", "脐周红肿", "脑膜刺激征阳性", "脓尿",
    "脓痰", "脚部皮疹", "脱水", "脾大", "腮腺颌下腺肿大", "腹水", "腹泻", "腹痛", "腹胀", "腹部包块", "腹部压痛",
    "腿痛", "臀部皮疹", "草莓舌/杨梅舌", "血尿", "角弓反张", "运动障碍", "里急后重", "陈旧性皮疹", "震颤", "颈部包块",
    "颈部疼痛", "颤抖", "食纳减少/厌食", "鸡鸣样回声", "麻疹黏膜斑", "黄疸", "黑便", "鼻翼扇动", "鼻衄", "性别",
]

def resolve_generation_columns(
    data_columns: List[str] | pd.Index,
    column_types: Dict[str, List[str]],
    outcome_cols: List[str],
) -> List[str]:
    selected = set(column_types["continuous"]) | set(column_types["categorical"])
    for outcome_col in outcome_cols:
        if outcome_col in data_columns:
            selected.add(outcome_col)

    generation_columns = [col for col in data_columns if col in selected]
    if not generation_columns:
        raise ValueError("未找到任何可用于生成的列，请检查 CONTINUOUS_COLS / CATEGORICAL_COLS 配置")
    return generation_columns

def validate_outcome_constraint_columns(spec: Any, expected_outcome_cols: List[str]) -> None:
    if spec is None or spec.outcome is None:
        return

    actual = list(spec.outcome.columns)
    expected = list(expected_outcome_cols)
    if actual != expected:
        raise ValueError(
            "outcome.columns 必须与主程序中的 OUTCOME_COLS 完全一致，"
            f"当前约束为 {actual}，主程序为 {expected}"
        )

def build_group_summary_df(group_diag: Optional[Dict[str, Any]]) -> pd.DataFrame:
    groups = group_diag.get("groups", []) if group_diag else []
    summary_rows = []
    for group in groups:
        summary_rows.append({
            "group_name": group.get("group_name"),
            "provider": group.get("provider"),
            "real_count": group.get("real_count"),
            "target_count": group.get("target_count"),
            "pooled_target_count": group.get("pooled_target_count"),
            "fixed_exact_columns": ", ".join(group.get("fixed_exact_columns", []) or []),
            "generated_columns_count": group.get("generated_columns_count"),
            "pool_size": group.get("pool_size"),
            "rounds": group.get("rounds"),
            "rejected_rows_total": group.get("rejected_rows_total"),
            "warning_count": len(group.get("warnings", [])),
        })
    return pd.DataFrame(summary_rows)

def make_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): make_json_safe(sub_value) for key, sub_value in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value

def resolve_model_save_dir(
    model: str,
    model_path: Optional[str] = None,
) -> str:
    if model_path:
        return model_path
    return os.path.join("models", model.lower())

def split_real_train_test(
    data: pd.DataFrame,
    test_size: float,
    random_state: int,
    outcome_columns: Optional[List[str]] = None,
    split_assignment_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    outcome_columns = outcome_columns or []
    if split_assignment_path:
        assignment_df, assignment_metadata = load_split_assignment(split_assignment_path)
        train_df, test_df, split_metadata = apply_split_assignment(
            data=data,
            assignment_df=assignment_df,
            outcome_columns=outcome_columns,
            split_metadata_override=assignment_metadata,
        )
        split_metadata["split_assignment_path"] = os.path.abspath(split_assignment_path)
    else:
        train_df, test_df, split_metadata, _ = generate_split_assignment(
            data=data,
            test_size=test_size,
            random_state=random_state,
            outcome_columns=outcome_columns,
        )
    return train_df, test_df, split_metadata

def prepare_model_config(
    model: str,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    cpu: bool = False,
    cuda: bool = False,
    copula_default_distribution: Optional[str] = None,
    copula_no_enforce_min_max_values: bool = False,
    copula_no_enforce_rounding: bool = False,
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    model_config = get_config(model)
    if epochs is not None:
        model_config["epochs"] = epochs
    if batch_size is not None:
        model_config["batch_size"] = batch_size
    if copula_default_distribution is not None:
        model_config["default_distribution"] = copula_default_distribution
    if copula_no_enforce_min_max_values:
        model_config["enforce_min_max_values"] = False
    if copula_no_enforce_rounding:
        model_config["enforce_rounding"] = False

    if cpu:
        if "cuda" in model_config:
            model_config["cuda"] = False
        if "device" in model_config:
            model_config["device"] = "cpu"
    elif cuda:
        if "cuda" in model_config:
            model_config["cuda"] = True
        if "device" in model_config:
            model_config["device"] = "cuda"

    if save_dir is not None:
        model_config["save_dir"] = save_dir
    return model_config

def generate_synthetic_from_training_data(
    train_df: pd.DataFrame,
    *,
    model: str,
    constraints_path: Optional[str],
    grouping_path: Optional[str],
    n_samples: int,
    random_state: int,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    cpu: bool = False,
    cuda: bool = False,
    load_model: bool = False,
    model_save_dir: Optional[str] = None,
    persist_model: bool = True,
    copula_default_distribution: Optional[str] = None,
    copula_no_enforce_min_max_values: bool = False,
    copula_no_enforce_rounding: bool = False,
    patience: int = 50,
    val_split: float = 0.1,
) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[Dict[str, Any]]]:
    processor = DataProcessor()
    column_types = processor.identify_column_types(
        train_df,
        continuous_cols=CONTINUOUS_COLS,
        categorical_cols=CATEGORICAL_COLS,
        outcome_cols=OUTCOME_COLS,
    )
    generation_columns = resolve_generation_columns(train_df.columns, column_types, OUTCOME_COLS)
    data_processed = processor.preprocess_data(train_df, fit=True)
    data_processed_for_generation = data_processed[generation_columns].copy()
    outcome_dist = get_outcome_joint_distribution(data_processed, OUTCOME_COLS)

    def _run_generation(resolved_model_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any], Optional[Dict[str, Any]]]:
        model_config = prepare_model_config(
            model=model,
            epochs=epochs,
            batch_size=batch_size,
            cpu=cpu,
            cuda=cuda,
            copula_default_distribution=copula_default_distribution,
            copula_no_enforce_min_max_values=copula_no_enforce_min_max_values,
            copula_no_enforce_rounding=copula_no_enforce_rounding,
            save_dir=resolved_model_dir,
        )
        extra_kwargs: Dict[str, Any] = {
            "outcome_columns": OUTCOME_COLS,
            "outcome_distribution": outcome_dist,
            "correct_outcome_distribution": True,
            "load_model": load_model,
            "early_stopping": True,
            "patience": patience,
            "val_split": val_split,
            "save_best": True,
        }
        if model.lower() == "tabddpm":
            extra_kwargs.update({
                "continuous_columns": column_types["continuous"],
                "categorical_columns": column_types["categorical"],
            })

        generation_metadata: Dict[str, Any] = {
            "generation_columns": generation_columns,
            "comparison_real_data": "train_split",
            "ml_test_real_data": "test_split",
            "outcome_distribution": {str(k): float(v) for k, v in outcome_dist.items()},
            "n_samples": int(n_samples),
            "model_dir": os.path.abspath(resolved_model_dir),
            "load_model": bool(load_model),
            "persist_model": bool(persist_model),
        }
        group_diag: Optional[Dict[str, Any]] = None

        if grouping_path:
            grouping_spec = load_grouping(grouping_path)
            constraint_spec = (
                load_constraints_with_reference_data(
                    constraints_path,
                    train_df,
                    continuous_columns=column_types["continuous"],
                )
                if constraints_path else None
            )
            validate_outcome_constraint_columns(constraint_spec, OUTCOME_COLS)
            synthetic_data, group_diag = generate_by_groups(
                model_name=model,
                model_config=model_config,
                data_raw=train_df,
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
            if group_diag is not None:
                generation_metadata["group_diag"] = group_diag
        else:
            generator = get_generator(model, **model_config)
            if load_model:
                try:
                    generator.load(checkpoint="best")
                except FileNotFoundError:
                    generator.fit(data_processed_for_generation, **extra_kwargs)
            else:
                generator.fit(data_processed_for_generation, **extra_kwargs)

            if constraints_path:
                constraint_spec = load_constraints_with_reference_data(
                    constraints_path,
                    train_df,
                    continuous_columns=column_types["continuous"],
                )
                validate_outcome_constraint_columns(constraint_spec, OUTCOME_COLS)
                generate_kwargs = dict(extra_kwargs)
                if constraint_spec.outcome is not None:
                    generate_kwargs["correct_outcome_distribution"] = False
                synthetic_data, diag = generate_with_constraints(
                    generator=generator,
                    n_samples=n_samples,
                    spec=constraint_spec,
                    outcome_distribution_real=outcome_dist,
                    rng_seed=random_state,
                    verbose=True,
                    generate_kwargs=generate_kwargs,
                )
                generation_metadata["constraint_diag"] = diag
            else:
                synthetic_data = generator.generate(n_samples, **extra_kwargs)
            generation_metadata["generator_info"] = generator.get_info()

        return synthetic_data, generation_metadata, group_diag

    if persist_model:
        resolved_model_dir = resolve_model_save_dir(model, model_save_dir)
        Path(resolved_model_dir).mkdir(parents=True, exist_ok=True)
        return _run_generation(resolved_model_dir)

    with tempfile.TemporaryDirectory(prefix=f"{model.lower()}_tmp_") as temp_dir:
        return _run_generation(temp_dir)
