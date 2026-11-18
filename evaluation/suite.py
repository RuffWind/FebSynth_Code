from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from .base import BaseMetricEvaluator
from .context import EvaluationContext
from .results import EvaluationReport

@dataclass
class EvaluationSuite:

    evaluators: List[BaseMetricEvaluator] = field(default_factory=list)

    def register(self, evaluator: BaseMetricEvaluator) -> None:
        self.evaluators.append(evaluator)

    def extend(self, evaluators: Iterable[BaseMetricEvaluator]) -> None:
        self.evaluators.extend(evaluators)

    def run(
        self,
        context: EvaluationContext,
        evaluators: Optional[Iterable[BaseMetricEvaluator]] = None,
    ) -> EvaluationReport:
        report = EvaluationReport(metadata={"random_state": context.random_state})
        active_evaluators = list(evaluators) if evaluators is not None else self.evaluators
        for evaluator in active_evaluators:
            metric = evaluator.evaluate(context)
            report.add_metric(metric)
        return report
