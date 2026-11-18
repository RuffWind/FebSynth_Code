from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

@dataclass
class MetricResult:
    name: str
    domain: str
    level: str
    summary: Dict[str, Any] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

@dataclass
class DomainEvaluationResult:
    domain: str
    metrics: List[MetricResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, metric: MetricResult) -> None:
        self.metrics.append(metric)

@dataclass
class EvaluationReport:
    domains: Dict[str, DomainEvaluationResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_metric(self, metric: MetricResult) -> None:
        domain_result = self.domains.setdefault(
            metric.domain,
            DomainEvaluationResult(domain=metric.domain),
        )
        domain_result.add_metric(metric)

    def get_domain(self, domain: str) -> DomainEvaluationResult:
        return self.domains.setdefault(domain, DomainEvaluationResult(domain=domain))
