from __future__ import annotations

from abc import ABC, abstractmethod

from .context import EvaluationContext
from .results import MetricResult

class BaseMetricEvaluator(ABC):

    domain: str = "generic"
    level: str = "generic"
    name: str = "base_metric"

    @abstractmethod
    def evaluate(self, context: EvaluationContext) -> MetricResult:
        raise NotImplementedError

class SimilarityEvaluator(BaseMetricEvaluator, ABC):
    domain = "similarity"

class UnivariateSimilarityEvaluator(SimilarityEvaluator, ABC):
    level = "univariate"

class MultivariateSimilarityEvaluator(SimilarityEvaluator, ABC):
    level = "multivariate"

class GlobalSimilarityEvaluator(SimilarityEvaluator, ABC):
    level = "global"

class MLUtilityEvaluator(BaseMetricEvaluator, ABC):
    domain = "ml_utility"
    level = "task"

class PrivacyEvaluator(BaseMetricEvaluator, ABC):
    domain = "privacy"
    level = "sample"
