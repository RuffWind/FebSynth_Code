from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .spec import ConstraintSpec, parse_constraint_spec, resolve_constraints_with_reference_data

def load_constraints(path: str) -> ConstraintSpec:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("约束JSON顶层必须是对象(dict)")
    return parse_constraint_spec(raw)

def load_constraints_from_obj(obj: Dict[str, Any]) -> ConstraintSpec:
    return parse_constraint_spec(obj)

def load_constraints_with_reference_data(
    path: str,
    reference_df: pd.DataFrame,
    continuous_columns: Optional[List[str]] = None,
) -> ConstraintSpec:
    return resolve_constraints_with_reference_data(
        load_constraints(path),
        reference_df,
        continuous_columns=continuous_columns,
    )
