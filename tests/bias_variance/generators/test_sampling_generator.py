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


def _generate_by_label(
    dataset: pd.DataFrame,
    settings,
    *,
    random_state: int | None = 42,
) -> dict[str, SamplingVariation]:
    generator = SamplingGenerator(
        settings
    )
    generator.base_dataset = dataset

    return {
        variation.label: variation
        for variation in generator.generate(random_state=random_state)
    }


def test_generate_returns_labeled_reproducible_samples(dataset):
    settings = {
        SamplingStrategyName.BOOTSTRAP: {
            "sample_fraction": 1.0,
            "with_replacement": True,
        },
        SamplingStrategyName.STRATIFIED: {
            "stratify_col_index": 0,
            "sample_fraction": 0.4,
        },
    }

    first = _generate_by_label(dataset, settings, random_state=42)
    second = _generate_by_label(dataset, settings, random_state=42)

    assert set(first) == {"bootstrap", "stratified"}
    assert len(first["bootstrap"].dataset) == 20
    assert len(first["stratified"].dataset) == 8

    assert isinstance(first["bootstrap"], SamplingVariation)
    assert isinstance(first["bootstrap"].variation_seed, int)
    assert (
        first["bootstrap"].variation_seed
        != first["stratified"].variation_seed
    )
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
    settings = {
        SamplingStrategyName.BOOTSTRAP: {},
        SamplingStrategyName.STRATIFIED: {},
    }

    generated = _generate_by_label(dataset, settings, random_state=42)

    generated["bootstrap"].dataset.loc[:, "feature"] = -1
    assert not (generated["stratified"].dataset["feature"] == -1).all()
    pd.testing.assert_frame_equal(dataset, pd.DataFrame(
        {
            "feature": range(20),
            "target": range(100, 120),
        }
    ))


def test_strategy_kwargs_are_passed_to_function(dataset):
    settings = {
        SamplingStrategyName.BOOTSTRAP: {
            "sample_fraction": 0.2,
            "with_replacement": False,
        },
    }

    generated = _generate_by_label(dataset, settings, random_state=42)

    assert len(generated["bootstrap"].dataset) == 4


def test_generate_without_base_dataset_raises_value_error():
    generator = SamplingGenerator({'bootstrap': {}})

    with pytest.raises(ValueError, match="Base dataset is not set"):
        tuple(generator.generate(random_state=42))


def test_strategy_name_rejects_a_mismatched_function():
    with pytest.raises(ValueError, match='must use get_random_samples'):
        SamplingGeneratorConfig({
            SamplingStrategyName.BOOTSTRAP: SamplingStrategy(
                function=SamplingStrategyName.LHS.function,
            ),
        })
