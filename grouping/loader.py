from __future__ import annotations

import json
from typing import Any, Dict

from .spec import GroupingSpec, parse_grouping_spec

def load_grouping(path: str) -> GroupingSpec:
    with open(path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("分组JSON顶层必须是对象(dict)")
    return parse_grouping_spec(raw)

def load_grouping_from_obj(obj: Dict[str, Any]) -> GroupingSpec:
    return parse_grouping_spec(obj)
