import pandas as pd
import pytest
import numpy as np

from bias_variance.analyzer import BiasAnalyzer
from bias_variance.config import RunConfigBuilder
from bias_variance.generators.noise import NoiseGenerator, NoiseGeneratorConfig
from bias_variance.seeding import derive_keyed_seed


def _builder() -> RunConfigBuilder:
    return (
        RunConfigBuilder()
        .set_X(pd.DataFrame({'x': range(10)}))
        .set_Y(pd.DataFrame({'y': range(10)}))
    )


def test_run_seed_plan_is_reproducible_and_uses_independent_branches() -> None:
    first = _builder().set_random_state(42).build()
    second = _builder().set_random_state(42).build()

    assert first.seed_plan == second.seed_plan
    assert first.seed_plan is not None
    assert len({
        first.seed_plan.baseline_split_seed,
        first.seed_plan.tuning_seed,
        first.seed_plan.workflow_seed,
    }) == 3
    pd.testing.assert_frame_equal(
        first.baseline.X_train,
        second.baseline.X_train,
    )


@pytest.mark.parametrize(
    ('random_state', 'error'),
    (
        (True, TypeError),
        (1.5, TypeError),
        (-1, ValueError),
        (2**32, ValueError),
    ),
)
def test_run_config_validates_random_state(random_state, error) -> None:
    with pytest.raises(error):
        _builder().set_random_state(random_state).build()


def test_noise_variation_seeds_are_keyed_by_label() -> None:
    dataset = pd.DataFrame({'x': [1.0, 2.0, 3.0]})

    def generate(order: tuple[float, ...]):
        generator = NoiseGenerator(NoiseGeneratorConfig(order))
        generator.base_dataset = dataset
        return {
            variation.label: variation
            for variation in generator.generate(random_state=17)
        }

    forward = generate((0.1, 0.2))
    reverse = generate((0.2, 0.1))

    assert {
        label: variation.variation_seed
        for label, variation in forward.items()
    } == {
        label: variation.variation_seed
        for label, variation in reverse.items()
    }
    for label, variation in forward.items():
        assert variation.variation_seed == derive_keyed_seed(
            17,
            'noise',
            label,
        )
        pd.testing.assert_frame_equal(
            variation.generated,
            reverse[label].generated,
        )


def test_each_iteration_and_action_receives_an_independent_seed() -> None:
    first = BiasAnalyzer._iteration_action_seeds(
        np.random.SeedSequence(42),
        4,
    )
    second = BiasAnalyzer._iteration_action_seeds(
        np.random.SeedSequence(42),
        4,
    )

    assert first == second
    assert len(first) == 4
    assert all(len(set(iteration)) == 3 for iteration in first)
    assert len({seed for iteration in first for seed in iteration}) == 12
