import numpy as np
import pandas as pd
import pytest

from bias_variance.generators.NoiseGenerator import (
    NoiseGenerator,
    NoiseVariation,
)


@pytest.fixture
def dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_a": [1.0, 2.0, 3.0, 4.0],
            "feature_b": [10.0, 20.0, 30.0, 40.0],
        },
        index=["row_a", "row_b", "row_c", "row_d"],
    )

def test_generate_returns_labeled_noise_variations(dataset):
    generator = NoiseGenerator(
        dataset,
        standard_deviations=[0.1, 0.25],
    )

    generated = generator.generate(random_state=42)

    assert set(generated) == {"std_0.1", "std_0.25"}

    first = generated["std_0.1"]
    second = generated["std_0.25"]

    assert isinstance(first, NoiseVariation)
    assert first.label == "std_0.1"
    assert first.standard_deviation == 0.1
    assert first.random_state == 42

    assert isinstance(second, NoiseVariation)
    assert second.label == "std_0.25"
    assert second.standard_deviation == 0.25
    assert second.random_state == 42

def test_default_standard_deviations_generate_expected_labels(dataset):
    generated = NoiseGenerator(dataset).generate(random_state=42)

    assert set(generated) == {
        "std_0.1",
        "std_0.2",
        "std_0.3",
        "std_0.4",
        "std_0.5",
    }

def test_generation_is_reproducible_for_same_seed(dataset):
    generator = NoiseGenerator(dataset, [0.1, 0.2])

    first = generator.generate(random_state=42)
    second = generator.generate(random_state=42)

    assert set(first) == set(second)

    for label in first:
        pd.testing.assert_frame_equal(
            first[label].dataset,
            second[label].dataset,
        )

def test_different_seeds_generate_different_noise(dataset):
    generator = NoiseGenerator(dataset, [0.1])

    first = generator.generate(random_state=42)["std_0.1"].dataset
    second = generator.generate(random_state=43)["std_0.1"].dataset

    assert not first.equals(second)

def test_generate_applies_expected_multiplicative_noise(dataset):
    generator = NoiseGenerator(dataset, [0.1])

    generated = generator.generate(random_state=42)["std_0.1"].dataset

    rng = np.random.default_rng(42)
    scale_factors = rng.normal(
        loc=1.0,
        scale=0.1,
        size=dataset.shape,
    )
    expected = dataset.mul(scale_factors, axis="columns")

    pd.testing.assert_frame_equal(generated, expected)

def test_each_standard_deviation_uses_next_rng_values(dataset):
    generator = NoiseGenerator(dataset, [0.1, 0.2])

    generated = generator.generate(random_state=42)

    rng = np.random.default_rng(42)

    expected_01 = dataset.mul(
        rng.normal(1.0, 0.1, size=dataset.shape),
        axis="columns",
    )
    expected_02 = dataset.mul(
        rng.normal(1.0, 0.2, size=dataset.shape),
        axis="columns",
    )

    pd.testing.assert_frame_equal(
        generated["std_0.1"].dataset,
        expected_01,
    )
    pd.testing.assert_frame_equal(
        generated["std_0.2"].dataset,
        expected_02,
    )

def test_generate_preserves_dataframe_metadata(dataset):
    generated = NoiseGenerator(dataset, [0.1]).generate(
        random_state=42
    )["std_0.1"].dataset

    assert generated.shape == dataset.shape
    assert generated.index.equals(dataset.index)
    assert generated.columns.equals(dataset.columns)

def test_generate_does_not_modify_source_dataset(dataset):
    original = dataset.copy(deep=True)
    generator = NoiseGenerator(dataset, [0.1])

    generator.generate(random_state=42)

    pd.testing.assert_frame_equal(dataset, original)

def test_constructor_copies_source_dataset(dataset):
    original = dataset.copy(deep=True)
    generator = NoiseGenerator(dataset, [0.1])

    dataset.loc[:, :] = -999.0

    actual = generator.generate(
        random_state=42
    )["std_0.1"].dataset

    expected = NoiseGenerator(original, [0.1]).generate(
        random_state=42
    )["std_0.1"].dataset

    pd.testing.assert_frame_equal(actual, expected)

def test_non_numeric_columns_are_rejected():
    dataset = pd.DataFrame(
        {
            "numeric": [1.0, 2.0],
            "category": ["a", "b"],
        }
    )

    with pytest.raises(
        TypeError,
        match="Noise can only be applied to numeric columns",
    ):
        NoiseGenerator(dataset)

def test_non_numeric_error_identifies_columns():
    dataset = pd.DataFrame(
        {
            "numeric": [1.0, 2.0],
            "category": ["a", "b"],
            "enabled": [True, False],
        }
    )

    with pytest.raises(TypeError) as error:
        NoiseGenerator(dataset)

    message = str(error.value)
    assert "category" in message
    assert "enabled" in message

@pytest.mark.parametrize(
    "standard_deviations",
    [
        [0.0],
        [-0.1],
        [0.1, -0.2],
    ],
)
def test_non_positive_standard_deviations_are_rejected(
    dataset,
    standard_deviations,
):
    with pytest.raises(
        ValueError,
        match="greater than zero",
    ):
        NoiseGenerator(dataset, standard_deviations)

def test_empty_standard_deviations_are_rejected(dataset):
    with pytest.raises(
        ValueError,
        match="At least one standard deviation",
    ):
        NoiseGenerator(dataset, [])

def test_duplicate_standard_deviations_are_rejected(dataset):
    with pytest.raises(
        ValueError,
        match="must be unique",
    ):
        NoiseGenerator(dataset, [0.1, 0.2, 0.1])

@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_non_finite_standard_deviations_are_rejected(dataset, value):
    with pytest.raises(
        ValueError,
        match="must be finite",
    ):
        NoiseGenerator(dataset, [value])