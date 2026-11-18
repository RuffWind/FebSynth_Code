from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass(frozen=True)
class GroupFeatureSpec:
    column: str
    thresholds: List[float] = field(default_factory=list)
    include_null: bool = False
    missing_to_other: bool = False

@dataclass(frozen=True)
class GroupingSpec:
    version: int = 1
    features: List[GroupFeatureSpec] = field(default_factory=list)
    rare_group_min_ratio: float = 0.0
    oversample_factor: float = 2.0
    max_rounds: int = 20
    min_group_size: int = 20
    fallback_to_bootstrap: bool = True
    random_seed: int = 42

    def validate(self) -> None:
        if self.version != 1:
            raise ValueError(f"不支持的分组配置版本: {self.version}")
        if not self.features:
            raise ValueError("分组配置至少需要1个特征")
        if self.rare_group_min_ratio < 0 or self.rare_group_min_ratio >= 1:
            raise ValueError("grouping.rare_group_min_ratio 必须满足 0 <= ratio < 1")
        if self.oversample_factor <= 1.0:
            raise ValueError("grouping.oversample_factor 必须 > 1.0")
        if self.max_rounds <= 0:
            raise ValueError("grouping.max_rounds 必须 > 0")
        if self.min_group_size <= 0:
            raise ValueError("grouping.min_group_size 必须 > 0")

        for feature in self.features:
            if not feature.column:
                raise ValueError("分组特征 column 不能为空")
            if feature.thresholds:
                normalized = [float(x) for x in feature.thresholds]
                if normalized != sorted(normalized):
                    raise ValueError(f"分组特征阈值必须升序: {feature.column}")
                if len(set(normalized)) != len(normalized):
                    raise ValueError(f"分组特征阈值不能重复: {feature.column}")

def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]

def parse_grouping_spec(raw: Dict[str, Any]) -> GroupingSpec:
    version = int(raw.get('version', 1))

    features_raw = raw.get('features', raw.get('group_features', []))
    features: List[GroupFeatureSpec] = []
    if isinstance(features_raw, list):
        for item in features_raw:
            if not isinstance(item, dict):
                continue
            column = str(item.get('column', item.get('name', ''))).strip()
            thresholds_raw = _as_list(item.get('thresholds', []))
            thresholds = [float(x) for x in thresholds_raw]
            include_null = bool(item.get('include_null', False))
            missing_to_other = bool(item.get('missing_to_other', False))
            features.append(
                GroupFeatureSpec(
                    column=column,
                    thresholds=thresholds,
                    include_null=include_null,
                    missing_to_other=missing_to_other,
                )
            )

    spec = GroupingSpec(
        version=version,
        features=features,
        rare_group_min_ratio=float(raw.get('rare_group_min_ratio', 0.0)),
        oversample_factor=float(raw.get('oversample_factor', 2.0)),
        max_rounds=int(raw.get('max_rounds', 20)),
        min_group_size=int(raw.get('min_group_size', 20)),
        fallback_to_bootstrap=bool(raw.get('fallback_to_bootstrap', True)),
        random_seed=int(raw.get('random_seed', 42)),
    )
    spec.validate()
    return spec
