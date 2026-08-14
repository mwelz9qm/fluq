from collections.abc import Mapping

import pytest

from bias_variance.generators.base import Generator


def test_generator_requires_generate_implementation():
    class IncompleteGenerator(Generator[int]):
        pass

    with pytest.raises(TypeError):
        IncompleteGenerator()


def test_generator_subclass_can_implement_contract():
    class FixedGenerator(Generator[int]):
        def generate(
            self,
            *,
            random_state: int | None = None,
        ) -> Mapping[str, int]:
            return {"seed": -1 if random_state is None else random_state}

    assert FixedGenerator().generate(random_state=42) == {"seed": 42}