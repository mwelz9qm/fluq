from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FnnArchitecture:
    hidden_layers: tuple[int, ...]