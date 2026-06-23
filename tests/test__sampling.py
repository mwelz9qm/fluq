import numpy as np
import pandas as pd
import pytest

from common.sampling._sampling import (
    generate_latin_hypercube_samples,
    get_random_samples,
    get_stratified_random_samples,
)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "feature_1": np.arange(12),
            "feature_2": np.arange(100, 112),
            "stratum": np.repeat(["A", "B", "C"], 4),
        }
    )


@pytest.fixture
def regressor_data():
    return pd.DataFrame(
        {
            "feature_1": np.arange(10, dtype=float),
            "feature_2": np.linspace(-5, 5, 10),
            "constant": np.full(10, 3.5),
        }
    )


def test_get_random_samples_size(sample_data):
    samples = get_random_samples(sample_data, n_samples=5, random_state=42)

    assert len(samples) == 5


def test_get_random_samples_fraction(sample_data):
    samples = get_random_samples(sample_data, sample_fraction=0.5, random_state=42)

    assert len(samples) == 6


def test_get_random_samples_rows_are_from_dataset(sample_data):
    samples = get_random_samples(sample_data, n_samples=5, random_state=42)

    assert samples.index.isin(sample_data.index).all()
    pd.testing.assert_frame_equal(samples, sample_data.loc[samples.index])


def test_get_random_samples_is_reproducible(sample_data):
    first = get_random_samples(sample_data, n_samples=5, random_state=17)
    second = get_random_samples(sample_data, n_samples=5, random_state=17)

    pd.testing.assert_frame_equal(first, second)


def test_get_random_samples_without_replacement_has_unique_rows(sample_data):
    samples = get_random_samples(
        sample_data, n_samples=8, random_state=42, with_replacement=False
    )

    assert not samples.index.duplicated().any()


def test_get_random_samples_with_replacement_can_exceed_dataset_size():
    data = pd.DataFrame({"value": [10]})

    samples = get_random_samples(
        data, n_samples=4, random_state=42, with_replacement=True
    )

    assert len(samples) == 4
    assert samples["value"].tolist() == [10, 10, 10, 10]


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({}, TypeError),
        ({"n_samples": 2, "sample_fraction": 0.5}, TypeError),
        ({"n_samples": 0}, ValueError),
        ({"n_samples": 13}, ValueError),
        ({"sample_fraction": 0}, ValueError),
        ({"sample_fraction": 1.1}, ValueError),
    ],
)
def test_get_random_samples_rejects_invalid_arguments(sample_data, kwargs, exception):
    with pytest.raises(exception):
        get_random_samples(sample_data, **kwargs)


def test_get_stratified_random_samples_returns_requested_size_and_balanced_strata(
    sample_data,
):
    samples = get_stratified_random_samples(
        sample_data,
        stratified_column_name="stratum",
        n_samples=7,
        random_state=42,
    )

    counts = samples["stratum"].value_counts()
    assert len(samples) == 7
    assert set(counts.index) == {"A", "B", "C"}
    assert counts.max() - counts.min() <= 1


def test_get_stratified_random_samples_fraction(sample_data):
    samples = get_stratified_random_samples(
        sample_data,
        stratified_column_name="stratum",
        sample_fraction=0.5,
        random_state=42,
    )

    assert len(samples) == 6
    assert samples["stratum"].value_counts().to_dict() == {"A": 2, "B": 2, "C": 2}


def test_get_stratified_random_samples_is_reproducible(sample_data):
    kwargs = {
        "stratified_column_name": "stratum",
        "n_samples": 7,
        "random_state": 17,
    }

    first = get_stratified_random_samples(sample_data, **kwargs)
    second = get_stratified_random_samples(sample_data, **kwargs)

    pd.testing.assert_frame_equal(first, second)


def test_get_stratified_random_samples_rows_are_from_dataset(sample_data):
    samples = get_stratified_random_samples(
        sample_data,
        stratified_column_name="stratum",
        n_samples=6,
        random_state=42,
    )

    pd.testing.assert_frame_equal(samples, sample_data.loc[samples.index])


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({}, TypeError),
        ({"n_samples": 2, "sample_fraction": 0.5}, TypeError),
        ({"n_samples": 0}, ValueError),
        ({"n_samples": 13}, ValueError),
        ({"sample_fraction": 0}, ValueError),
        ({"sample_fraction": 1.1}, ValueError),
    ],
)
def test_get_stratified_random_samples_rejects_invalid_arguments(
    sample_data, kwargs, exception
):
    with pytest.raises(exception):
        get_stratified_random_samples(
            sample_data, stratified_column_name="stratum", **kwargs
        )


def test_get_stratified_random_samples_rejects_unknown_column(sample_data):
    with pytest.raises(KeyError):
        get_stratified_random_samples(
            sample_data,
            stratified_column_name="missing",
            n_samples=3,
            random_state=42,
        )


def test_generate_latin_hypercube_samples_shape_and_columns(regressor_data):
    samples = generate_latin_hypercube_samples(
        regressor_data, n_samples=5, random_state=42
    )

    assert samples.shape == (5, 3)
    assert samples.columns.tolist() == regressor_data.columns.tolist()


def test_generate_latin_hypercube_samples_fraction(regressor_data):
    samples = generate_latin_hypercube_samples(
        regressor_data, sample_fraction=0.4, random_state=42
    )

    assert len(samples) == 4


def test_generate_latin_hypercube_samples_values_stay_within_column_bounds(
    regressor_data,
):
    samples = generate_latin_hypercube_samples(
        regressor_data, n_samples=6, random_state=42
    )

    assert samples.ge(regressor_data.min()).all().all()
    assert samples.le(regressor_data.max()).all().all()
    assert samples["constant"].eq(3.5).all()


def test_generate_latin_hypercube_samples_uses_each_quantile_interval_once(
    regressor_data,
):
    n_samples = 5
    samples = generate_latin_hypercube_samples(
        regressor_data, n_samples=n_samples, random_state=42
    )
    quantiles = regressor_data.quantile(
        np.linspace(0, 1, n_samples + 1), interpolation="midpoint"
    )

    for column in regressor_data.columns:
        sorted_samples = np.sort(samples[column].to_numpy())
        lower_bounds = quantiles[column].to_numpy()[:-1]
        upper_bounds = quantiles[column].to_numpy()[1:]
        assert np.all(sorted_samples >= lower_bounds)
        assert np.all(sorted_samples <= upper_bounds)


def test_generate_latin_hypercube_samples_is_reproducible(regressor_data):
    first = generate_latin_hypercube_samples(
        regressor_data, n_samples=5, random_state=17
    )
    second = generate_latin_hypercube_samples(
        regressor_data, n_samples=5, random_state=17
    )

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    ("kwargs", "exception"),
    [
        ({}, TypeError),
        ({"n_samples": 2, "sample_fraction": 0.5}, TypeError),
        ({"n_samples": 0}, ValueError),
        ({"n_samples": 11}, ValueError),
        ({"sample_fraction": 0}, ValueError),
        ({"sample_fraction": 1.1}, ValueError),
    ],
)
def test_generate_latin_hypercube_samples_rejects_invalid_arguments(
    regressor_data, kwargs, exception
):
    with pytest.raises(exception):
        generate_latin_hypercube_samples(regressor_data, **kwargs)
