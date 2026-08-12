from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class VariationGeneratorConfig(ABC):
    '''Configures generator settings.'''

    @property
    @abstractmethod
    def variation_labels(self) -> tuple[str, ...]:
        '''Returns all variation labels.'''
        raise NotImplementedError

    @staticmethod
    def _validate_dataset(dataset: pd.DataFrame) -> None:
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("Base dataset must be a pandas DataFrame.")

        if dataset.empty:
            raise ValueError("Base dataset must contain at least one row and column.")

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
            raise ValueError("Base dataset must contain only finite values.")


@dataclass(frozen=True, slots=True)
class Variation[VaritionGeneratedT]:
    '''
    Represents the base variation returned by any Generator.

    Attributes:
        label: The identifier of the generated variation.
        random_state: The random seed used if provided.
        generated: The value produced by the generator.
    '''
    label: str
    random_state: int | None
    generated: VaritionGeneratedT


class VariationGenerator[VaritionGeneratedT](ABC):
    '''Produces labeled variations for one study iteration.'''

    @property
    @abstractmethod
    def variation_labels(self) -> tuple[str, ...]:
        '''Returns all variation labels registered with ```VariationGeneratorConfig.```'''
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> list[Variation[VaritionGeneratedT]]:
        '''Generates one variation per configured label.'''
        raise NotImplementedError
