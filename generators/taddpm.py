
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List
from .base import BaseSyntheticDataGenerator, EarlyStopping
from utils.outcome_utils import format_outcome_distribution, get_outcome_joint_distribution

class TabDDPMGenerator(BaseSyntheticDataGenerator):

    def __init__(self,
                 epochs: int = 1000,
                 batch_size: int = 256,
                 lr: float = 0.002,
                 num_timesteps: int = 1000,
                 d_in: Optional[int] = None,
                 num_classes: Optional[int] = None,
                 is_y_cond: bool = False,
                 rtdl_params: Optional[Dict] = None,
                 verbose: bool = True,
                 device: str = 'cuda',
                 save_dir: Optional[str] = None):
        super().__init__(name="TabDDPM", save_dir=save_dir)

        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.num_timesteps = num_timesteps
        self.d_in = d_in
        self.num_classes = num_classes
        self.is_y_cond = is_y_cond
        self.rtdl_params = rtdl_params or {'d_layers': [256, 256, 256], 'dropout': 0.0}
        self.verbose = verbose

        if device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            if self.verbose:
                print(f"✓ 使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            if self.verbose:
                print("✓ 使用CPU")

        self.continuous_columns = []
        self.categorical_columns = []
        self.all_columns = []
        self.column_info = {}

        self.outcome_columns = []
        self.outcome_distribution = None

    @staticmethod
    def _choose_missing_sentinel(unique_vals: np.ndarray) -> float:
        used_values = {float(v) for v in unique_vals.tolist()}
        sentinel = -1.0
        if sentinel not in used_values:
            return sentinel

        if used_values:
            return float(np.floor(min(used_values)) - 1.0)

        return sentinel

    @staticmethod
    def _map_to_nearest_allowed(values: np.ndarray, allowed_values: List[float]) -> np.ndarray:
        allowed = np.asarray(sorted(float(v) for v in allowed_values), dtype=float)
        raw = np.asarray(values, dtype=float)

        if allowed.size == 0:
            return np.round(raw)

        distances = np.abs(raw[:, None] - allowed[None, :])
        nearest_idx = distances.argmin(axis=1)
        return allowed[nearest_idx]

    def _prepare_data(self, data: pd.DataFrame,
                     continuous_cols: List[str],
                     categorical_cols: List[str]) -> torch.Tensor:
        from sklearn.preprocessing import StandardScaler

        data_cols = set(data.columns)
        self.continuous_columns = [col for col in continuous_cols if col in data_cols]
        self.categorical_columns = [col for col in categorical_cols if col in data_cols]

        for outcome_col in self.outcome_columns:
            if outcome_col in data_cols and outcome_col not in self.categorical_columns:
                self.categorical_columns.append(outcome_col)
                if self.verbose:
                    print(f"  ✓ 自动将结局变量'{outcome_col}'添加到类别型变量")

        self.all_columns = [col for col in data.columns
                           if col in self.continuous_columns or col in self.categorical_columns]

        if self.verbose:
            print(f"\n  数据列信息:")
            print(f"    数据中的总列数: {len(data.columns)}")
            print(f"    实际使用的列数: {len(self.all_columns)}")
            print(f"    连续型变量: {len(self.continuous_columns)}")
            print(f"    类别型变量: {len(self.categorical_columns)}")
            if self.outcome_columns:
                print(f"    结局变量: {self.outcome_columns} ✓")

        data_to_use = data[self.all_columns].copy()

        processed_data = pd.DataFrame()

        if self.continuous_columns:
            self.scaler = StandardScaler()
            continuous_data = data_to_use[self.continuous_columns].copy()

            self.continuous_missing_mask = continuous_data.isnull()

            continuous_data_filled = continuous_data.fillna(continuous_data.mean())

            continuous_data_scaled = pd.DataFrame(
                self.scaler.fit_transform(continuous_data_filled),
                columns=self.continuous_columns,
                index=data_to_use.index
            )

            processed_data = pd.concat([processed_data, continuous_data_scaled], axis=1)

        if self.categorical_columns:
            categorical_data = data_to_use[self.categorical_columns].copy()

            print_count = 0
            for col in self.categorical_columns:
                unique_vals = categorical_data[col].dropna().unique()
                is_outcome = (col in self.outcome_columns)
                has_nan = categorical_data[col].isnull().any()
                missing_sentinel = (
                    self._choose_missing_sentinel(unique_vals)
                    if has_nan else None
                )
                allowed_values = sorted(float(v) for v in unique_vals.tolist())
                if missing_sentinel is not None:
                    allowed_values.append(float(missing_sentinel))
                    allowed_values = sorted(set(allowed_values))

                self.column_info[col] = {
                    'type': 'categorical',
                    'unique_values': sorted(unique_vals.tolist()),
                    'has_nan': has_nan,
                    'nan_count': categorical_data[col].isnull().sum(),
                    'is_outcome': is_outcome,
                    'missing_sentinel': missing_sentinel,
                    'allowed_values': allowed_values,
                }

                should_print = is_outcome or print_count < 5

                if self.verbose and should_print:
                    if len(unique_vals) <= 10:
                        nan_info = f", NaN数={self.column_info[col]['nan_count']}" if self.column_info[col]['has_nan'] else ""
                        outcome_mark = " [结局变量]" if is_outcome else ""
                        sentinel_info = (
                            f", 缺失哨兵={missing_sentinel:g}"
                            if missing_sentinel is not None else ""
                        )
                        print(f"    {col}{outcome_mark}: 唯一值 = {sorted(unique_vals.tolist())}{nan_info}{sentinel_info}")
                    else:
                        nan_info = f", NaN数={self.column_info[col]['nan_count']}" if self.column_info[col]['has_nan'] else ""
                        outcome_mark = " [结局变量]" if is_outcome else ""
                        sentinel_info = (
                            f", 缺失哨兵={missing_sentinel:g}"
                            if missing_sentinel is not None else ""
                        )
                        print(f"    {col}{outcome_mark}: 唯一值数量 = {len(unique_vals)}{nan_info}{sentinel_info}")

                    if not is_outcome:
                        print_count += 1

            for col in self.categorical_columns:
                missing_sentinel = self.column_info[col].get('missing_sentinel')
                if missing_sentinel is not None:
                    categorical_data.loc[categorical_data[col].isnull(), col] = missing_sentinel

            categorical_data_float = categorical_data.astype(float)

            processed_data = pd.concat([processed_data, categorical_data_float], axis=1)

        processed_data = processed_data[self.all_columns]

        X = torch.FloatTensor(processed_data.values)

        if torch.isnan(X).any():
            nan_cols = processed_data.columns[processed_data.isnull().any()].tolist()
            raise ValueError(
                f"TabDDPM训练输入仍包含未处理的NaN，列: {nan_cols}"
            )

        return X

    def _build_model(self, d_in: int):
        layers = []
        prev_dim = d_in + 1

        for hidden_dim in self.rtdl_params['d_layers']:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.rtdl_params['dropout'])
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, d_in))

        self.model = nn.Sequential(*layers).to(self.device)

        if self.verbose:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"  模型参数量: {total_params:,}")

    def _save_checkpoint_as_fitted(self, checkpoint: str) -> None:
        previous_is_fitted = self.is_fitted
        self.is_fitted = True
        try:
            self.save(checkpoint=checkpoint)
        finally:
            self.is_fitted = previous_is_fitted

    def _gaussian_diffusion_forward(self, x, t):

        beta_start = 0.0001
        beta_end = 0.02
        betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        alpha_t = alphas_cumprod[t].view(-1, 1)

        noise = torch.randn_like(x)
        x_noisy = torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

        return x_noisy, noise

    def fit(self, data: pd.DataFrame, **kwargs) -> None:

        self.outcome_columns = list(kwargs.get('outcome_columns', []) or [])
        self.outcome_distribution = kwargs.get('outcome_distribution', None)

        val_split = kwargs.get('val_split', 0.1)
        use_early_stopping = kwargs.get('early_stopping', True)
        patience = kwargs.get('patience', 50)
        save_best = kwargs.get('save_best', True)

        print(f"  训练参数:")
        print(f"    训练轮数: {self.epochs}")
        print(f"    批次大小: {self.batch_size}")
        print(f"    学习率: {self.lr}")
        print(f"    扩散步数: {self.num_timesteps}")
        print(f"    验证集比例: {val_split*100:.1f}%")
        print(f"    早停: {'启用' if use_early_stopping else '禁用'} (patience={patience})")

        if self.outcome_columns:
            print(f"  结局变量: {self.outcome_columns}")
            if self.outcome_distribution:
                print(f"  结局联合分布:")
                for line in format_outcome_distribution(self.outcome_columns, self.outcome_distribution):
                    print(f"    {line}")

        continuous_cols = kwargs.get('continuous_columns', [])
        categorical_cols = kwargs.get('categorical_columns', [])

        if not continuous_cols and not categorical_cols:
            print("\n  ⚠ 警告: 未提供列类型信息，将使用数据中的所有列")
            continuous_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = []

        try:

            print("\n  准备数据...")
            X = self._prepare_data(data, continuous_cols, categorical_cols)
            self.d_in = X.shape[1]

            print(f"  训练数据维度: {X.shape}")

            from sklearn.model_selection import train_test_split
            if val_split > 0:
                X_train, X_val = train_test_split(
                    X, test_size=val_split, random_state=42
                )
                print(f"  训练集: {X_train.shape[0]} 样本")
                print(f"  验证集: {X_val.shape[0]} 样本")
            else:
                X_train = X
                X_val = None
                print(f"  不使用验证集")

            print("\n  构建模型...")
            self._build_model(self.d_in)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            criterion = nn.MSELoss()

            train_dataset = torch.utils.data.TensorDataset(X_train)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True
            )

            if X_val is not None:
                val_dataset = torch.utils.data.TensorDataset(X_val)
                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False
                )

            if use_early_stopping and X_val is not None:
                early_stopping = EarlyStopping(
                    patience=patience,
                    min_delta=1e-4,
                    verbose=self.verbose
                )
            else:
                early_stopping = None

            print(f"\n  开始训练...")
            self.model.train()

            for epoch in range(self.epochs):

                epoch_train_loss = 0.0
                num_batches = 0

                for batch in train_dataloader:
                    x_batch = batch[0].to(self.device)

                    t = torch.randint(
                        0, self.num_timesteps,
                        (x_batch.shape[0],),
                        device=self.device
                    )

                    x_noisy, noise = self._gaussian_diffusion_forward(x_batch, t)

                    t_input = t.float().view(-1, 1) / self.num_timesteps
                    model_input = torch.cat([x_noisy, t_input], dim=1)
                    noise_pred = self.model(model_input)

                    loss = criterion(noise_pred, noise)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_train_loss += loss.item()
                    num_batches += 1

                avg_train_loss = epoch_train_loss / num_batches
                self.training_history['train_loss'].append(avg_train_loss)

                if X_val is not None:
                    self.model.eval()
                    epoch_val_loss = 0.0
                    num_val_batches = 0

                    with torch.no_grad():
                        for batch in val_dataloader:
                            x_batch = batch[0].to(self.device)

                            t = torch.randint(
                                0, self.num_timesteps,
                                (x_batch.shape[0],),
                                device=self.device
                            )

                            x_noisy, noise = self._gaussian_diffusion_forward(x_batch, t)

                            t_input = t.float().view(-1, 1) / self.num_timesteps
                            model_input = torch.cat([x_noisy, t_input], dim=1)
                            noise_pred = self.model(model_input)

                            loss = criterion(noise_pred, noise)
                            epoch_val_loss += loss.item()
                            num_val_batches += 1

                    avg_val_loss = epoch_val_loss / num_val_batches
                    self.training_history['val_loss'].append(avg_val_loss)

                    self.model.train()
                else:
                    avg_val_loss = avg_train_loss

                if self.verbose and (epoch + 1) % 50 == 0:
                    print(f"  Epoch [{epoch+1}/{self.epochs}]", end="")
                    print(f", Train Loss: {avg_train_loss:.6f}", end="")
                    if X_val is not None:
                        print(f", Val Loss: {avg_val_loss:.6f}", end="")
                    print()

                if early_stopping is not None:
                    if early_stopping(avg_val_loss, epoch + 1):
                        print(f"\n  早停触发，停止训练")
                        self.training_history['best_epoch'] = early_stopping.best_epoch
                        self.training_history['best_loss'] = early_stopping.best_loss
                        break

                    if save_best and early_stopping.counter == 0:
                        if self.verbose:
                            print(f"  → 保存最佳模型（Epoch {epoch+1}）")
                        self._save_checkpoint_as_fitted(checkpoint='best')

            self.training_history['epochs'] = epoch + 1
            if early_stopping is not None:
                self.training_history['best_epoch'] = early_stopping.best_epoch
                self.training_history['best_loss'] = early_stopping.best_loss

            self.is_fitted = True

            self.metadata = {
                'columns': self.all_columns,
                'shape': (len(data), len(self.all_columns)),
                'epochs': self.training_history['epochs'],
                'batch_size': self.batch_size,
                'num_timesteps': self.num_timesteps,
                'd_in': self.d_in,
                'continuous_columns': self.continuous_columns,
                'categorical_columns': self.categorical_columns,
                'outcome_columns': self.outcome_columns,
                'outcome_distribution': self.outcome_distribution
            }

            if save_best:
                self.save(checkpoint='last')
                if early_stopping is None:

                    self.save(checkpoint='best')

            print("\n  ✓ 训练完成！")
            print(f"  实际训练轮数: {self.training_history['epochs']}")
            if 'best_epoch' in self.training_history:
                print(f"  最佳轮次: {self.training_history['best_epoch']}")
                print(f"  最佳损失: {self.training_history['best_loss']:.6f}")

        except Exception as e:
            print(f"✗ TabDDPM训练失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _get_model_state(self):
        import torch
        return {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler if hasattr(self, 'scaler') else None,
            'all_columns': self.all_columns,
            'continuous_columns': self.continuous_columns,
            'categorical_columns': self.categorical_columns,
            'column_info': self.column_info,
            'd_in': self.d_in,
            'lr': self.lr,
            'num_classes': self.num_classes,
            'is_y_cond': self.is_y_cond,
            'rtdl_params': self.rtdl_params,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'num_timesteps': self.num_timesteps
        }

    def _set_model_state(self, state):
        import torch

        self.all_columns = state['all_columns']
        self.continuous_columns = state['continuous_columns']
        self.categorical_columns = state['categorical_columns']
        self.column_info = state['column_info']
        self.d_in = state['d_in']
        self.scaler = state.get('scaler', None)

        self.lr = state.get('lr', self.lr)
        self.num_classes = state.get('num_classes', self.num_classes)
        self.is_y_cond = state.get('is_y_cond', self.is_y_cond)
        self.rtdl_params = state.get('rtdl_params', self.rtdl_params)
        self.epochs = state.get('epochs', self.epochs)
        self.batch_size = state.get('batch_size', self.batch_size)
        self.num_timesteps = state.get('num_timesteps', self.num_timesteps)

        self._build_model(self.d_in)

        self.model.load_state_dict(state['model_state_dict'])
        self.model.eval()

    def _gaussian_diffusion_reverse(self, num_samples: int) -> torch.Tensor:

        x = torch.randn(num_samples, self.d_in, device=self.device)

        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((num_samples,), t, device=self.device, dtype=torch.long)
            t_input = t_tensor.float().view(-1, 1) / self.num_timesteps

            with torch.no_grad():
                model_input = torch.cat([x, t_input], dim=1)
                noise_pred = self.model(model_input)

            beta_start = 0.0001
            beta_end = 0.02
            betas = torch.linspace(beta_start, beta_end, self.num_timesteps, device=self.device)
            alphas = 1.0 - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)

            alpha_t = alphas_cumprod[t]
            beta_t = betas[t]

            x = (x - beta_t / torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(1 - beta_t)

            if t > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(beta_t) * noise

        return x

    def generate(self, n_samples: int, **kwargs) -> pd.DataFrame:
        if not self.is_fitted:
            raise ValueError("模型尚未训练，请先调用fit()方法")

        try:
            correct_distribution = kwargs.get('correct_outcome_distribution', True)
            if correct_distribution and self.outcome_columns and self.outcome_distribution:
                n_samples_to_generate = int(n_samples * 1.5)
                print(f"  为了校正结局联合分布，先生成 {n_samples_to_generate} 个样本...")
            else:
                n_samples_to_generate = n_samples

            print(f"  正在生成 {n_samples_to_generate} 个样本...")

            self.model.eval()

            batch_size = min(self.batch_size, n_samples_to_generate)
            num_batches = (n_samples_to_generate + batch_size - 1) // batch_size

            generated_data = []

            for i in range(num_batches):
                current_batch_size = min(batch_size, n_samples_to_generate - i * batch_size)

                if self.verbose and num_batches > 1:
                    print(f"  生成批次 {i+1}/{num_batches}...")

                with torch.no_grad():
                    batch_data = self._gaussian_diffusion_reverse(current_batch_size)
                    generated_data.append(batch_data.cpu().numpy())

            X_generated = np.vstack(generated_data)

            synthetic_data = pd.DataFrame(
                X_generated,
                columns=self.all_columns
            )

            print(f"\n  后处理数据...")

            if self.continuous_columns and hasattr(self, 'scaler'):
                print(f"    反标准化 {len(self.continuous_columns)} 个连续型变量...")
                synthetic_data[self.continuous_columns] = self.scaler.inverse_transform(
                    synthetic_data[self.continuous_columns]
                )

            if self.categorical_columns:
                print(f"    处理 {len(self.categorical_columns)} 个类别型变量...")
                for col in self.categorical_columns:
                    if col in self.column_info:
                        unique_vals = self.column_info[col]['unique_values']
                        missing_sentinel = self.column_info[col].get('missing_sentinel')
                        allowed_values = self.column_info[col].get(
                            'allowed_values',
                            [float(v) for v in unique_vals],
                        )

                        mapped_values = self._map_to_nearest_allowed(
                            synthetic_data[col].values,
                            allowed_values,
                        )
                        synthetic_data[col] = mapped_values

                        if missing_sentinel is not None:
                            synthetic_data.loc[synthetic_data[col] == missing_sentinel, col] = np.nan

                for col in self.categorical_columns:
                    if col in synthetic_data.columns:

                        synthetic_data[col] = synthetic_data[col].astype(float)

            outcome_columns_present = [
                col for col in self.outcome_columns
                if col in synthetic_data.columns
            ]
            if outcome_columns_present:
                print(f"    后处理结局变量: {outcome_columns_present}")
                synthetic_data = self._normalize_binary_outcome_columns(synthetic_data)
                generated_dist_before = get_outcome_joint_distribution(synthetic_data, outcome_columns_present)
                print(f"    生成的结局联合分布（校正前）:")
                for line in format_outcome_distribution(outcome_columns_present, generated_dist_before):
                    print(f"      {line}")

                if correct_distribution and self.outcome_distribution:
                    synthetic_data = self._stratified_resample(
                        synthetic_data,
                        outcome_columns_present,
                        self.outcome_distribution,
                        n_samples,
                    )
                    generated_dist_after = get_outcome_joint_distribution(synthetic_data, outcome_columns_present)
                    print(f"    生成的结局联合分布（校正后）:")
                    for line in format_outcome_distribution(outcome_columns_present, generated_dist_after):
                        print(f"      {line}")

                if self.outcome_distribution:
                    print(f"    原始结局联合分布:")
                    for line in format_outcome_distribution(outcome_columns_present, self.outcome_distribution):
                        print(f"      {line}")

            print(f"\n  ✓ 生成完成！实际生成 {len(synthetic_data)} 个样本，{len(synthetic_data.columns)} 个特征")

            if self.verbose and self.categorical_columns:
                print(f"\n  类别型变量统计（前5个非结局变量）:")
                print_count = 0
                for col in self.categorical_columns:
                    if col in synthetic_data.columns and col not in set(self.outcome_columns) and print_count < 5:
                        unique_vals = synthetic_data[col].dropna().unique()
                        print(f"    {col}: 唯一值 = {sorted(unique_vals.tolist())}")
                        print(f"      NaN数量 = {synthetic_data[col].isnull().sum()}")
                        print_count += 1

            return synthetic_data

        except Exception as e:
            print(f"✗ TabDDPM生成失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
