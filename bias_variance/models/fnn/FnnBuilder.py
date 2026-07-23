from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from .FnnArchitecture import FnnArchitecture


@dataclass(frozen=True, slots=True)
class FnnConfig:
    input_size: int
    output_size: int
    activation_factory: Callable[[], nn.Module] = nn.ReLU
    bias: bool = True

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError('input_size must be greater than 0.')
        
        if self.output_size <= 0:
            raise ValueError('output_size must be greater than 0.')


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