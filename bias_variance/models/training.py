from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from bias_variance.models.fnn import FnnArchitecture, FnnBuilder, FnnConfig


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    optimizer: str = 'adam'
    learning_rate: float = 1e-3
    loss: str = 'mse'
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
        if not isinstance(self.optimizer, str):
            raise TypeError('optimizer must be a string.')
        if self.optimizer not in {'adam', 'sgd'}:
            raise ValueError('optimizer must be one of: adam, sgd.')
        if (
            not isinstance(self.learning_rate, Real)
            or isinstance(self.learning_rate, bool)
        ):
            raise TypeError('learning_rate must be a real number.')
        if not np.isfinite(self.learning_rate):
            raise ValueError('learning_rate must be finite.')
        if self.learning_rate <= 0:
            raise ValueError(
                'learning_rate must be greater than 0.'
            )
        if (
            not isinstance(self.epochs, int)
            or isinstance(self.epochs, bool)
        ):
            raise TypeError('epochs must be an integer.')
        if self.epochs < 0:
            raise ValueError(
                'epochs cannot be negative.'
            )
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
        ):
            raise TypeError('batch_size must be an integer.')
        if self.batch_size <= 0:
            raise ValueError(
                'batch_size must be greater than 0.'
            )
        if not isinstance(self.loss, str):
            raise TypeError('loss must be a string.')
        if self.loss not in {'mse', 'mae'}:
            raise ValueError('loss must be one of: mae, mse.')
        if not isinstance(self.device, str):
            raise TypeError('device must be a string.')
        if self.device != 'auto':
            try:
                torch.device(self.device)
            except (RuntimeError, ValueError) as error:
                raise ValueError(f'Invalid device: {self.device!r}.') from error


class Trainer:
    def __init__(
        self,
        config: TrainingConfig,
        model_builder: FnnBuilder | None = None
    ) -> None:
        self.config = config
        self.model_builder = model_builder

    def set_fnn_model_builder(self, input_size, output_size) -> None:
        config = FnnConfig(input_size, output_size)
        self.model_builder = FnnBuilder(config)

    def train(
        self,
        architecture: FnnArchitecture,
        x_train: torch.Tensor | pd.DataFrame,
        y_train: torch.Tensor | pd.DataFrame,
        random_state: int | None,
    ) -> nn.Sequential:
        if self.model_builder is None:
            raise RuntimeError(
                'A model builder must be configured before training.'
            )

        if not isinstance(x_train, torch.Tensor):
            x_train = torch.as_tensor(
                x_train.to_numpy(dtype=np.float32, copy=True),
                dtype=torch.float32,
            )
        else:
            x_train = x_train.to(dtype=torch.float32)

        if not isinstance(y_train, torch.Tensor):
            y_train = torch.as_tensor(
                y_train.to_numpy(dtype=np.float32, copy=True),
                dtype=torch.float32,
            )
        else:
            y_train = y_train.to(dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)

        forked_devices: list[int] = []

        if self.config.resolved_device.type == 'cuda':
            device_index = self.config.resolved_device.index
            forked_devices.append(
                torch.cuda.current_device()
                if device_index is None
                else device_index
            )

        with torch.random.fork_rng(devices=forked_devices):
            if random_state is not None:
                torch.manual_seed(random_state)

            model = _build_model(
                architecture,
                self.model_builder,
                self.config.resolved_device
            )

            loader_generator = torch.Generator()

            if random_state is not None:
                loader_generator.manual_seed(random_state)

            loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                generator=loader_generator
            )

            criterion = _build_loss(self.config.loss)
            optimizer = _build_optimizer(
                model,
                self.config.optimizer,
                self.config.learning_rate
            )
            
            model.train()
            
            for _ in np.arange(self.config.epochs):
                for batch_x, batch_y in loader:
                    batch_x = batch_x.to(self.config.resolved_device)
                    batch_y = batch_y.to(self.config.resolved_device)
                    optimizer.zero_grad()
                    predictions  = model(batch_x)
                    loss = criterion(predictions, batch_y)
                    loss.backward()
                    optimizer.step()

        return model


# Helper functions

def _build_model(
    architecture: FnnArchitecture,
    model_builder: FnnBuilder,
    resolved_device: torch.device,
) -> nn.Sequential:
    model = model_builder.build(architecture)
    return model.to(resolved_device)

def _build_optimizer(
    model: nn.Module,
    optimizer: str,
    learning_rate: float,
) ->  optim.Optimizer:
    optimizers: dict[str, type[optim.Optimizer]] = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
    }

    try:
        optimizer_type = optimizers[optimizer]

    except KeyError:
        raise ValueError(
            f'Unsupported optimizer: {optimizer!r}.'
            f'Expected one of {sorted(optimizers)}.'
        ) from None
    
    return optimizer_type(
        model.parameters(),
        lr=learning_rate,
    )

def _build_loss(loss: str) -> nn.Module:
    losses: dict[str, type[nn.Module]] = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
    }

    try:
        loss_type = losses[loss]

    except KeyError:
        raise ValueError(
            f'Unsupported loss: {loss!r}.'
            f'Expected one of {sorted(losses)}.'
        ) from None
    
    return loss_type()
