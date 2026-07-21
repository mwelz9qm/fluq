from .Generator import Generator
from collections.abc import Iterable
from dataclasses import dataclass
import numpy as np
import pandas as pd



@dataclass(frozen=True)
class NoiseVariation:
    label: str
    dataset: pd.DataFrame
    standard_deviation: float
    random_state: int | None = None


class NoiseGenerator(Generator[pd.DataFrame]):
    def __init__(
        self,
        dataset: pd.DataFrame,
        standard_deviations: Iterable[float] = (
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ),
    ) -> None:
        self._dataset = dataset.copy()
        self._standard_deviations = self._validate_standard_deviations(
            standard_deviations
        )

        non_numeric = self._dataset.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        if non_numeric:
            raise TypeError(
                f"Noise can only be applied to numeric columns: {non_numeric}"
            )

    @staticmethod
    def _validate_standard_deviations(
        values: Iterable[float],
    ) -> tuple[float, ...]:
        standard_deviations = tuple(float(value) for value in values)

        if not standard_deviations:
            raise ValueError(
                "At least one standard deviation must be configured."
            )

        if any(value <= 0 for value in standard_deviations):
            raise ValueError(
                "Standard deviations must be greater than zero."
            )

        if len(set(standard_deviations)) != len(standard_deviations):
            raise ValueError(
                "Standard deviations must be unique."
            )

        return standard_deviations

    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> dict[str, NoiseVariation]:
        rng = np.random.default_rng(random_state)
        generated = {}

        for standard_deviation in self._standard_deviations:
            scale_factors = rng.normal(
                loc=1.0,
                scale=standard_deviation,
                size=self._dataset.shape,
            )

            noisy_dataset = self._dataset.mul(
                scale_factors,
                axis="columns",
            )

            # Explicitly preserve pandas metadata.
            noisy_dataset.index = self._dataset.index
            noisy_dataset.columns = self._dataset.columns

            label = f"std_{standard_deviation:g}"
            generated[label] = NoiseVariation(
                dataset=noisy_dataset,
                standard_deviation=standard_deviation,
                random_state=random_state,
            )

        return generated