from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .utils import normalize_categorical_dataframe

def build_ml_preprocessor(
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

def prepare_ml_feature_dataframe(
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

def build_binary_classification_pipelines(
    continuous_columns: List[str],
    categorical_columns: List[str],
    random_state: int,
) -> Dict[str, Pipeline]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError(
            "机器学习二分类评估需要安装 `xgboost` 包。"
            " 如果你使用 conda 环境，建议运行：`conda install -c conda-forge xgboost`"
        ) from exc

    return {
        "logistic_regression": Pipeline([
            ("preprocessor", build_ml_preprocessor(continuous_columns, categorical_columns)),
            ("model", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "random_forest": Pipeline([
            ("preprocessor", build_ml_preprocessor(continuous_columns, categorical_columns)),
            ("model", RandomForestClassifier(
                n_estimators=300,
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
        "xgboost": Pipeline([
            ("preprocessor", build_ml_preprocessor(continuous_columns, categorical_columns)),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )),
        ]),
    }

def calculate_binary_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series | List[int],
    y_proba: pd.Series | List[float],
) -> Dict[str, float]:
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(recall_score(y_true, y_pred, pos_label=0, zero_division=0))
    average_precision = float(average_precision_score(y_true, y_proba))
    roc_auc = float(roc_auc_score(y_true, y_proba))
    return {
        "auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "average_precision": average_precision,
    }

def select_youden_threshold(
    y_true: pd.Series | List[int],
    y_proba: pd.Series | List[float],
    *,
    default_threshold: float = 0.5,
) -> float:
    y_true_array = np.asarray(y_true, dtype=int)
    y_proba_array = np.asarray(y_proba, dtype=float)
    if y_true_array.size == 0 or np.unique(y_true_array).size < 2:
        return float(default_threshold)

    fpr, tpr, thresholds = roc_curve(y_true_array, y_proba_array)
    finite_mask = np.isfinite(thresholds)
    if not finite_mask.any():
        return float(default_threshold)

    candidate_fpr = fpr[finite_mask]
    candidate_tpr = tpr[finite_mask]
    candidate_thresholds = thresholds[finite_mask]
    youden_scores = candidate_tpr - candidate_fpr
    best_index = int(np.argmax(youden_scores))
    return float(candidate_thresholds[best_index])

def probabilities_to_predictions(
    y_proba: pd.Series | List[float],
    threshold: float,
) -> np.ndarray:
    y_proba_array = np.asarray(y_proba, dtype=float)
    return (y_proba_array >= float(threshold)).astype(int)

def predict_with_youden_threshold(
    pipeline: Pipeline,
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    *,
    random_state: int,
    validation_size: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, float, str]:
    decision_threshold = 0.5
    decision_threshold_source = "default_0_5"
    try:
        split_train_X, validation_X, split_train_y, validation_y = train_test_split(
            train_X,
            train_y,
            test_size=validation_size,
            random_state=random_state,
            stratify=train_y,
        )
        threshold_pipeline = clone(pipeline).fit(split_train_X, split_train_y)
        validation_probabilities = threshold_pipeline.predict_proba(validation_X)[:, 1]
        decision_threshold = select_youden_threshold(
            validation_y,
            validation_probabilities,
            default_threshold=0.5,
        )
        decision_threshold_source = "youden_validation"
    except ValueError:
        decision_threshold = 0.5
        decision_threshold_source = "default_0_5_fallback"

    fitted_pipeline = clone(pipeline).fit(train_X, train_y)
    probabilities = fitted_pipeline.predict_proba(test_X)[:, 1]
    predictions = probabilities_to_predictions(probabilities, decision_threshold)
    return probabilities, predictions, float(decision_threshold), decision_threshold_source
