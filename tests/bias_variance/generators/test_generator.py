from collections.abc import Iterable

import pytest

from bias_variance.generators.base import Variation, VariationGenerator


def test_variation_generator_requires_generate_implementation():
    class IncompleteGenerator(VariationGenerator[int]):
        @property
        def variation_labels(self) -> tuple[str, ...]:
            return ("seed",)

    with pytest.raises(TypeError):
        IncompleteGenerator()


def test_variation_generator_requires_variation_labels_implementation():
    class IncompleteGenerator(VariationGenerator[int]):
        def generate(
            self,
            *,
            random_state: int | None = None,
        ) -> Iterable[Variation[int]]:
            return (Variation("seed", random_state, 42),)

    with pytest.raises(TypeError):
        IncompleteGenerator()


def test_variation_generator_subclass_can_implement_contract():
    class FixedGenerator(VariationGenerator[int]):
        @property
        def variation_labels(self) -> tuple[str, ...]:
            return ("seed",)

        def generate(
            self,
            *,
            random_state: int | None = None,
        ) -> Iterable[Variation[int]]:
            return (
                Variation(
                    "seed",
                    random_state,
                    -1 if random_state is None else random_state,
                ),
            )

    assert tuple(FixedGenerator().generate(random_state=42)) == (
        Variation("seed", 42, 42),
    )