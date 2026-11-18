from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import pandas as pd

@dataclass(frozen=True)
class EvaluationSchema:
    common_columns: List[str]
    continuous_columns: List[str]
    categorical_columns: List[str]
    outcome_columns: List[str]

    @property
    def feature_columns(self) -> List[str]:
        return [col for col in self.common_columns if col not in self.outcome_columns]

    @classmethod
    def from_datasets(
        cls,
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        continuous_candidates: Sequence[str],
        categorical_candidates: Sequence[str],
        outcome_columns: Sequence[str],
    ) -> "EvaluationSchema":
        common_columns = [col for col in synthetic_data.columns if col in real_data.columns]
        candidate_continuous = set(continuous_candidates)
        candidate_categorical = set(categorical_candidates)
        candidate_outcomes = set(outcome_columns)

        continuous = [col for col in common_columns if col in candidate_continuous]
        categorical = [col for col in common_columns if col in candidate_categorical]
        outcomes = [col for col in common_columns if col in candidate_outcomes]

        classified = set(continuous) | set(categorical) | set(outcomes)
        extras = [col for col in common_columns if col not in classified]
        for col in extras:
            series = synthetic_data[col] if col in synthetic_data.columns else real_data[col]
            if pd.api.types.is_numeric_dtype(series) and series.nunique(dropna=True) > 10:
                continuous.append(col)
            else:
                categorical.append(col)

        return cls(
            common_columns=common_columns,
            continuous_columns=continuous,
            categorical_columns=categorical,
            outcome_columns=outcomes,
        )
