
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import time
import os
import json
import pickle
from pathlib import Path

from utils.outcome_utils import enforce_joint_outcome_distribution, format_outcome_distribution

class BaseSyntheticDataGenerator(ABC):

    def __init__(self, name: str = "BaseGenerator", save_dir: Optional[str] = None):
        self.name = name
        self.is_fitted = False
        self.metadata: Dict[str, Any] = {}
        self.training_time = 0
        self.generation_time = 0
        self.outcome_columns: List[str] = []
        self.outcome_distribution = None

        if save_dir is None:

            self.save_dir = f"models/{name.lower()}"
        else:
            self.save_dir = save_dir

        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'epochs': 0,
            'best_epoch': 0,
            'best_loss': float('inf')
        }

    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> None:
        pass

    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        pass

    def fit_generate(self, data: pd.DataFrame, n_samples: int, **kwargs) -> pd.DataFrame:
        print(f"\n{'='*80}")
        print(f"使用 {self.name} 生成器")
        print(f"{'='*80}")

        load_model = kwargs.get('load_model', False)

        if load_model:
            try:
                self.load(checkpoint='best')
                print(f"\n✓ 成功加载已保存的模型")
            except FileNotFoundError:
                print(f"\n⚠ 未找到已保存的模型，将重新训练")
                load_model = False

        if not load_model:
            start_time = time.time()
            self.fit(data, **kwargs)
            self.training_time = time.time() - start_time

        start_time = time.time()
        synthetic_data = self.generate(n_samples, **kwargs)
        self.generation_time = time.time() - start_time

        return synthetic_data

    def save(self, checkpoint: str = 'best') -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        filename = f"model_{checkpoint}.pkl"
        filepath = os.path.join(self.save_dir, filename)

        save_dict = {
            'model_state': self._get_model_state(),
            'metadata': self.metadata,
            'training_history': self.training_history,
            'outcome_columns': self.outcome_columns,
            'outcome_distribution': self.outcome_distribution,
            'is_fitted': self.is_fitted
        }

        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)

        history_file = os.path.join(self.save_dir, 'training_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        print(f"  ✓ 模型已保存到: {filepath}")

    def load(self, checkpoint: str = 'best') -> None:
        filename = f"model_{checkpoint}.pkl"
        filepath = os.path.join(self.save_dir, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")

        with open(filepath, 'rb') as f:
            save_dict = pickle.load(f)

        self._set_model_state(save_dict['model_state'])

        self.metadata = save_dict.get('metadata', {})
        self.training_history = save_dict.get('training_history', {})
        legacy_outcome_column = save_dict.get('outcome_column', None)
        self.outcome_columns = save_dict.get('outcome_columns', None) or ([legacy_outcome_column] if legacy_outcome_column else [])
        self.outcome_distribution = save_dict.get('outcome_distribution', None)
        self.is_fitted = True

        print(f"  ✓ 模型已加载: {filepath}")
        print(f"    训练轮数: {self.training_history.get('epochs', 0)}")
        print(f"    最佳轮次: {self.training_history.get('best_epoch', 0)}")

    @abstractmethod
    def _get_model_state(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def _set_model_state(self, state: Dict[str, Any]) -> None:
        pass

    def get_info(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'is_fitted': self.is_fitted,
            'training_time': self.training_time,
            'generation_time': self.generation_time,
            'metadata': self.metadata,
            'training_history': self.training_history
        }

    def _normalize_binary_outcome_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        result = data.copy()
        for outcome_col in self.outcome_columns:
            if outcome_col not in result.columns:
                continue

            outcome_values = result[outcome_col]
            if outcome_values.dtype == 'object' or outcome_values.dtype.name == 'category':
                outcome_values = pd.to_numeric(outcome_values, errors='coerce')
            else:
                outcome_values = outcome_values.astype(float)

            if outcome_values.isnull().any():
                print(f"    ⚠ 警告: 结局变量'{outcome_col}'发现{outcome_values.isnull().sum()}个无效值")
                mode_value = outcome_values.mode()[0] if len(outcome_values.mode()) > 0 else 0
                outcome_values.fillna(mode_value, inplace=True)

            outcome_values = np.round(outcome_values).astype(float)
            outcome_values = np.clip(outcome_values, 0, 1)
            result[outcome_col] = outcome_values.fillna(0)

        return result

    def _stratified_resample(self,
                            data: pd.DataFrame,
                            outcome_columns: List[str],
                            target_distribution: Dict[Any, float],
                            n_samples: int) -> pd.DataFrame:
        print(f"\n  开始分层重采样...")

        for line in format_outcome_distribution(outcome_columns, target_distribution):
            print(f"    目标: {line}")

        result = enforce_joint_outcome_distribution(
            df=data,
            outcome_columns=outcome_columns,
            target_dist=target_distribution,
            n_samples=n_samples,
            rng=np.random.default_rng(42),
        )
        print(f"  ✓ 重采样完成，最终样本数: {len(result)}")
        return result

class EarlyStopping:

    def __init__(self, patience: int = 50, min_delta: float = 1e-4, verbose: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose

        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:

            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            if self.verbose:
                print(f"  → 验证损失改进: {val_loss:.6f} (最佳)")
        else:

            self.counter += 1
            if self.verbose:
                print(f"  → 验证损失: {val_loss:.6f} (无改进 {self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f"\n  ⚠ 早停触发！已连续{self.patience}轮无改进")
                    print(f"  最佳轮次: Epoch {self.best_epoch}, 损失: {self.best_loss:.6f}")

        return self.early_stop
