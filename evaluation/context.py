from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from .schema import EvaluationSchema

@dataclass
class EvaluationContext:
    generation_real_data: pd.DataFrame
    evaluation_real_data: pd.DataFrame
    synthetic_data: pd.DataFrame
    schema: EvaluationSchema
    comparison_real_data: Optional[pd.DataFrame] = None
    random_state: int = 42
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.generation_real_data = self.generation_real_data.copy()
        self.evaluation_real_data = self.evaluation_real_data.copy()
        self.synthetic_data = self.synthetic_data.copy()
        self.comparison_real_data = (
            self.comparison_real_data.copy()
            if self.comparison_real_data is not None
            else self.evaluation_real_data.copy()
        )

    @property
    def valid_continuous_columns(self) -> List[str]:
        return [
            col for col in self.schema.continuous_columns
            if col in self.comparison_real_data.columns and col in self.synthetic_data.columns
        ]

    @property
    def valid_categorical_columns(self) -> List[str]:
        return [
            col for col in self.schema.categorical_columns
            if col in self.comparison_real_data.columns and col in self.synthetic_data.columns
        ]
