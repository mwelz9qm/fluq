from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class VariationGeneratorConfig(ABC):

    @property
    @abstractmethod
    def variation_labels(self) -> tuple[str, ...]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Variation[VaritionGeneratedT]:
    label: str
    random_state: int | None
    generated: VaritionGeneratedT

    @staticmethod
    def _validate_label(label: str) -> None:
        if not isinstance(label, str):
            raise TypeError(
                'label must be a string.'
            )

        if label.strip() == '':
            raise ValueError(
                'label cannot be empty string or whitespace.'
            )

    @staticmethod
    def _validate_random_state(random_state: int | None) -> None:
        if (
            not isinstance(random_state, int)
            or isinstance(random_state, bool)
        ) and random_state is not None:
            raise TypeError('random_state must be an integer or None.')

        if random_state is not None and not (0 <= random_state < 2**32):
            raise ValueError(
                'random_state out of bounds.'
            )

    def __post_init__(self) -> None:
        self._validate_label(self.label)
        self._validate_random_state(self.random_state)


class VariationGenerator[VaritionGeneratedT](ABC):

    @property
    @abstractmethod
    def variation_labels(self) -> tuple[str, ...]:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> Iterable[Variation[VaritionGeneratedT]]:
        raise NotImplementedError

    @staticmethod
    def _validate_dataset(dataset: pd.DataFrame) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Base dataset must be a pandas DataFrame.")

        if dataset.empty:
            raise ValueError(
                "Base dataset must contain at least one row and column."
            )

        invalid_columns = [
            column
            for column in dataset.columns
            if (
                not pd.api.types.is_numeric_dtype(dataset[column])
                or pd.api.types.is_bool_dtype(dataset[column])
                or pd.api.types.is_complex_dtype(dataset[column])
            )
        ]
        if invalid_columns:
            raise TypeError(
                "Noise can only be applied to real numeric columns. "
                f"Invalid columns: {invalid_columns!r}"
            )

        values = dataset.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                "Base dataset must contain only finite values."
            )