from collections.abc import Callable
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class FnnArchitecture:
    hidden_layers: tuple[int, ...]

    def __post_init__(self) -> None:
        for size in self.hidden_layers:
            if (
                not isinstance(size, int)
                or isinstance(size, bool)
                or size <= 0
            ):
                raise ValueError(
                    'Hidden-layer sizes must be positive integers.'
                )


@dataclass(frozen=True, slots=True)
class FnnConfig:
    input_size: int
    output_size: int
    activation_factory: Callable[[], nn.Module] = nn.ReLU
    bias: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.input_size, int)
            or isinstance(self.input_size, bool)
        ):
            raise TypeError('input_size must be an integer.')
        if self.input_size <= 0:
            raise ValueError('input_size must be greater than 0.')
        if (
            not isinstance(self.output_size, int)
            or isinstance(self.output_size, bool)
        ):
            raise TypeError('output_size must be an integer.')
        if self.output_size <= 0:
            raise ValueError('output_size must be greater than 0.')
        if not callable(self.activation_factory):
            raise TypeError('activation_factory must be callable.')
        if not isinstance(self.bias, bool):
            raise TypeError('bias must be a boolean.')


class FnnBuilder:
    def __init__(self, config: FnnConfig) -> None:
        self.config = config

    def build(
        self,
        architecture: FnnArchitecture,
    ) -> nn.Sequential:
        layers: list[nn.Module] = []
        previous_size = self.config.input_size

        for hidden_size in architecture.hidden_layers:
            layers.extend(
                [
                    nn.Linear(
                        previous_size,
                        hidden_size,
                        bias=self.config.bias,
                    ),
                    self.config.activation_factory(),
                ]
            )
            previous_size = hidden_size

        layers.append(
            nn.Linear(
                previous_size,
                self.config.output_size,
                bias=self.config.bias,
            )
        )

        return nn.Sequential(*layers)
