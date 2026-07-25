from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn

from bias_variance.models.fnn.FnnArchitecture import FnnArchitecture
from bias_variance.models.fnn.FnnBuilder import FnnBuilder, FnnConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    optimizer: str = 'adam'
    learning_rate: float = 1e-3
    loss: str = 'mse'
    metrics: tuple[str, ...] = (
        'rmse',
        'r2',
        'mse',
        'mae'
    )
    epochs: int = 100
    batch_size: int = 10
    device: str = 'auto'

    @property
    def resolved_device(self) -> torch.device:
        if self.device != 'auto':
            return torch.device(self.device)
        
        if torch.cuda.is_available():
            return torch.device('cuda')
            
        if torch.backends.mps.is_available():
            return torch.device('mps')
        
        return torch.device('cpu')

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError(
                'learning_rate must be greater than 0.'
            )
        
        if self.epochs < 0:
            raise ValueError(
                'epochs cannot be negative.'
            )
        
        if (
            not isinstance(self.epochs, int)
            or isinstance(self.epochs, bool)
        ):
            raise TypeError('epochs must be an integer.')
        
        if self.batch_size <= 0:
            raise ValueError(
                'batch_size must be greater than 0.'
            )
        
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
        ):
            raise TypeError('batch_size must be an integer.')



class Trainer:
    def __init__(self, config: TrainingConfig, model_builder: FnnBuilder | None = None) -> None:
        self.config = config
        self.model_builder = model_builder

    def _build_model(self, architecture: FnnArchitecture) -> nn.Sequential:
        model = self.model_builder.build(architecture)
        return model.to(self.training_config.resolved_device)

    def train(
        self,
        architecture: FnnArchitecture,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        random_state: int | None,
    ) -> nn.Module:
        pass

    def predict(
        self,
        model: nn.Module,
        x_test: pd.DataFrame,
    ) -> np.ndarray[float]:
        pass

    def set_model_builder(self, input_size, output_size) -> None:
        config = FnnConfig(input_size, output_size)
        self.model_builder = FnnBuilder(config)
