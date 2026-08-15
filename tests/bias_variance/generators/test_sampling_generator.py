import pandas as pd
import pytest

from bias_variance.generators.sampling import (
    SamplingGenerator,
    SamplingGeneratorConfig,
    SamplingStrategy,
    SamplingStrategyName,
    SamplingVariation,
)


@pytest.fixture
def dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": range(20),
            "target": range(100, 120),
        }
    )


def _take_rows(
    frame: pd.DataFrame,
    *,
    random_state: int | None = None,
    n_rows: int = 1,
) -> pd.DataFrame:
    return frame.head(n_rows).copy()


def _generate_by_label(
    dataset: pd.DataFrame,
    strategies,
    *,
    random_state: int | None = 42,
) -> dict[str, SamplingVariation]:
    generator = SamplingGenerator(
        SamplingGeneratorConfig(strategies)
    )
    generator.base_dataset = dataset

    return {
        variation.label: variation
        for variation in generator.generate(random_state=random_state)
    }


def test_generate_returns_labeled_reproducible_samples(dataset):
    strategies = {
        SamplingStrategyName.BOOTSTRAP: SamplingStrategy(
            function=_take_rows,
            kwargs={"n_rows": 20},
        ),
        SamplingStrategyName.STRATIFIED: SamplingStrategy(
            function=_take_rows,
            kwargs={"n_rows": 8},
        ),
    }

    first = _generate_by_label(dataset, strategies, random_state=42)
    second = _generate_by_label(dataset, strategies, random_state=42)

    assert set(first) == {"bootstrap", "stratified"}
    assert len(first["bootstrap"].dataset) == 20
    assert len(first["stratified"].dataset) == 8

    assert isinstance(first["bootstrap"], SamplingVariation)
    assert first["bootstrap"].random_state == 42
    pd.testing.assert_frame_equal(
        first["bootstrap"].dataset,
        second["bootstrap"].dataset,
    )
    pd.testing.assert_frame_equal(
        first["stratified"].dataset,
        second["stratified"].dataset,
    )


def test_default_strategy_labels_match_config():
    generator = SamplingGenerator()

    assert generator.variation_labels == (
        "bootstrap",
        "stratified",
        "lhs",
    )


def test_dataset_property_returns_copy(dataset):
    generator = SamplingGenerator()
    generator.base_dataset = dataset

    copied_dataset = generator.dataset
    copied_dataset.loc[:, "feature"] = -1

    pd.testing.assert_frame_equal(generator.base_dataset, dataset)


def test_each_strategy_receives_an_independent_dataset_copy(dataset):
    def mutate(
        frame: pd.DataFrame,
        *,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        frame.loc[:, "feature"] = -1
        return frame

    def observe(
        frame: pd.DataFrame,
        *,
        random_state: int | None = None,
    ) -> pd.DataFrame:
        return frame

    strategies = {
        SamplingStrategyName.BOOTSTRAP: SamplingStrategy(function=mutate),
        SamplingStrategyName.STRATIFIED: SamplingStrategy(function=observe),
    }

    generated = _generate_by_label(dataset, strategies, random_state=42)

    assert (generated["bootstrap"].dataset["feature"] == -1).all()
    pd.testing.assert_frame_equal(generated["stratified"].dataset, dataset)
    pd.testing.assert_frame_equal(dataset, pd.DataFrame(
        {
            "feature": range(20),
            "target": range(100, 120),
        }
    ))


def test_strategy_kwargs_are_passed_to_function(dataset):
    strategies = {
        SamplingStrategyName.BOOTSTRAP: SamplingStrategy(
            function=_take_rows,
            kwargs={"n_rows": 4},
        ),
    }

    generated = _generate_by_label(dataset, strategies, random_state=42)

    pd.testing.assert_frame_equal(
        generated["bootstrap"].dataset,
        dataset.head(4),
    )


def test_generate_without_base_dataset_raises_value_error():
    strategies = {
        SamplingStrategyName.BOOTSTRAP: SamplingStrategy(
            function=_take_rows,
            kwargs={"n_rows": 4},
        ),
    }
    generator = SamplingGenerator(
        SamplingGeneratorConfig(strategies)
    )

    with pytest.raises(ValueError, match="Base dataset is not set"):
        tuple(generator.generate(random_state=42))