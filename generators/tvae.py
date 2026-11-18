
import pandas as pd
import numpy as np
from sdv.single_table import TVAESynthesizer
from sdv.metadata import SingleTableMetadata
from .base import BaseSyntheticDataGenerator, EarlyStopping

from utils.outcome_utils import format_outcome_distribution, get_outcome_joint_distribution

class TVAEGenerator(BaseSyntheticDataGenerator):

    def __init__(self,
                 epochs: int = 300,
                 batch_size: int = 500,
                 embedding_dim: int = 128,
                 compress_dims: tuple = (128, 128),
                 decompress_dims: tuple = (128, 128),
                 l2scale: float = 1e-5,
                 loss_factor: int = 2,
                 verbose: bool = True,
                 cuda: bool = True,
                 save_dir: str = None):
        super().__init__(name="TVAE", save_dir=save_dir)

        self.epochs = epochs
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.compress_dims = compress_dims
        self.decompress_dims = decompress_dims
        self.l2scale = l2scale
        self.loss_factor = loss_factor
        self.verbose = verbose
        self.cuda = cuda

        self.model = None
        self.metadata_obj = None

    def fit(self, data: pd.DataFrame, **kwargs) -> None:

        self.outcome_columns = list(kwargs.get('outcome_columns', []) or [])
        self.outcome_distribution = kwargs.get('outcome_distribution', None)
        save_best = kwargs.get('save_best', True)

        missing_outcome_columns = [col for col in self.outcome_columns if col not in data.columns]
        if missing_outcome_columns:
            print(f"  ⚠ 警告: 以下结局变量不在数据中，已忽略: {missing_outcome_columns}")
        self.outcome_columns = [col for col in self.outcome_columns if col in data.columns]
        if self.outcome_columns:
            print(f"  结局变量: {self.outcome_columns}")
            for outcome_col in self.outcome_columns:
                print(f"    {outcome_col}: 类型={data[outcome_col].dtype}, 唯一值={sorted(data[outcome_col].dropna().unique().tolist())}")
            if self.outcome_distribution:
                print(f"  结局联合分布:")
                for line in format_outcome_distribution(self.outcome_columns, self.outcome_distribution):
                    print(f"    {line}")

        self.metadata_obj = SingleTableMetadata()
        self.metadata_obj.detect_from_dataframe(data)

        for outcome_col in self.outcome_columns:
            self.metadata_obj.update_column(outcome_col, sdtype='categorical')
            if self.verbose:
                print(f"  ✓ 已将结局变量'{outcome_col}'设置为类别型")

        self.model = TVAESynthesizer(
            self.metadata_obj,
            epochs=self.epochs,
            batch_size=self.batch_size,
            embedding_dim=self.embedding_dim,
            compress_dims=self.compress_dims,
            decompress_dims=self.decompress_dims,
            l2scale=self.l2scale,
            loss_factor=self.loss_factor,
            verbose=self.verbose,
            cuda=self.cuda
        )

        print(f"  训练参数:")
        print(f"    Epochs: {self.epochs}")
        print(f"    Batch Size: {self.batch_size}")
        print(f"    Embedding Dim: {self.embedding_dim}")
        print(f"    Compress Dims: {self.compress_dims}")
        print(f"    Decompress Dims: {self.decompress_dims}")
        print(f"    使用GPU: {self.cuda}")
        print(f"  ⚠ 注意: SDV的TVAE不支持早停，将训练完整个epochs")

        self.model.fit(data)
        self.is_fitted = True

        self.training_history['epochs'] = self.epochs

        self.metadata = {
            'columns': list(data.columns),
            'shape': data.shape,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'outcome_columns': self.outcome_columns,
            'outcome_distribution': self.outcome_distribution
        }

        if save_best:
            self.save(checkpoint='last')
            self.save(checkpoint='best')

    def _get_model_state(self):
        return {
            'model': self.model,
            'metadata_obj': self.metadata_obj,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'embedding_dim': self.embedding_dim,
            'compress_dims': self.compress_dims,
            'decompress_dims': self.decompress_dims
        }

    def _set_model_state(self, state):
        self.model = state['model']
        self.metadata_obj = state['metadata_obj']
        self.epochs = state.get('epochs', self.epochs)
        self.batch_size = state.get('batch_size', self.batch_size)
        self.embedding_dim = state.get('embedding_dim', self.embedding_dim)
        self.compress_dims = state.get('compress_dims', self.compress_dims)
        self.decompress_dims = state.get('decompress_dims', self.decompress_dims)

    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit()方法")

        correct_distribution = kwargs.get('correct_outcome_distribution', True)

        if correct_distribution and self.outcome_columns and self.outcome_distribution:
            n_samples_to_generate = int(n_samples * 1.5)
            print(f"  为了校正结局分布，先生成 {n_samples_to_generate} 个样本...")
        else:
            n_samples_to_generate = n_samples

        synthetic_data = self.model.sample(num_rows=n_samples_to_generate)

        outcome_columns_present = [
            col for col in self.outcome_columns
            if col in synthetic_data.columns
        ]
        if outcome_columns_present:
            print(f"\n  后处理结局变量: {outcome_columns_present}...")
            synthetic_data = self._normalize_binary_outcome_columns(synthetic_data)
            generated_dist_before = get_outcome_joint_distribution(synthetic_data, outcome_columns_present)
            print(f"  生成的结局联合分布（校正前）:")
            for line in format_outcome_distribution(outcome_columns_present, generated_dist_before):
                print(f"    {line}")

            if correct_distribution and self.outcome_distribution:
                synthetic_data = self._stratified_resample(
                    synthetic_data,
                    outcome_columns_present,
                    self.outcome_distribution,
                    n_samples
                )
                generated_dist_after = get_outcome_joint_distribution(synthetic_data, outcome_columns_present)
                print(f"\n  生成的结局联合分布（校正后）:")
                for line in format_outcome_distribution(outcome_columns_present, generated_dist_after):
                    print(f"    {line}")

            if self.outcome_distribution:
                print(f"  原始结局联合分布:")
                for line in format_outcome_distribution(outcome_columns_present, self.outcome_distribution):
                    print(f"    {line}")

        return synthetic_data
