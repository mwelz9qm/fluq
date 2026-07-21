from dataclasses import dataclass


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