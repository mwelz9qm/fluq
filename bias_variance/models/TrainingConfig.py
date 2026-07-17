from dataclasses import dataclass


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