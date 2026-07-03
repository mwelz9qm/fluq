import numpy as np
import pandas as pd
import pytest

from common.sampling._sampling import (
    generate_latin_hypercube_samples,
    get_quantile_stratified_random_samples,
    get_random_samples,
    get_stratified_random_samples,
)


@pytest.fixture
def sample_data():
    '''12x3 sample dataset'''
    return pd.DataFrame(
        {
            "feature_1": np.arange(12),
            "feature_2": np.arange(100, 112),
            "stratum": np.repeat(["A", "B", "C"], 4),
        }
    )


@pytest.fixture
def regressor_data():
    '''10x3 regressor sample dataset'''
    return pd.DataFrame(
        {
            "feature_1": np.arange(10, dtype=float),
            "feature_2": np.linspace(-5, 5, 10),
            "constant": np.full(10, 3.5),
        }
    )


###############################################
#       TESTS FOR get_random_samples()        #
###############################################

def test_get_random_samples_size(sample_data):
    '''Test n_samples arg for get_random_samples()'''
    samples = get_random_samples(sample_data, n_samples=5, random_state=42)

    assert len(samples) == 5


def test_get_random_samples_fraction(sample_data):
    '''Test sample_fraction arg for get_random_samples()'''
    samples = get_random_samples(sample_data, sample_fraction=0.5, random_state=42)

    assert len(samples) == 6


def test_get_random_samples_rows_are_from_dataset(sample_data):
    '''Test row data integrity for get_random_samples()'''
    samples = get_random_samples(sample_data, n_samples=5, random_state=42)

    assert samples.index.isin(sample_data.index).all()
    pd.testing.assert_frame_equal(samples, sample_data.loc[samples.index])


def test_get_random_samples_is_reproducible(sample_data):
    '''Test random_state arg for get_random_samples()'''
    first = get_random_samples(sample_data, n_samples=5, random_state=17)
    second = get_random_samples(sample_data, n_samples=5, random_state=17)

    pd.testing.assert_frame_equal(first, second)


def test_get_random_samples_without_replacement_has_unique_rows(sample_data):
    '''Test when with_replacement=False for get_random_samples()'''
    samples = get_random_samples(
        sample_data, n_samples=8, random_state=42, with_replacement=False
    )

    assert not samples.index.duplicated().any()


def test_get_random_samples_with_replacement_can_exceed_dataset_size():
    '''Test when with_replacement=True for get_random_samples()'''
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
    '''Test invalid arguments in get_random_samples()'''
    with pytest.raises(exception):
        get_random_samples(sample_data, **kwargs)


##########################################################
#       TESTS FOR get_stratified_random_samples()        #
##########################################################

def test_get_stratified_random_samples_returns_requested_size_and_balanced_strata(
    sample_data,
):
    '''Test n_samples arg and if stratas are balanced in get_stratified_random_samples()'''
    samples = get_stratified_random_samples(
        sample_data,
        stratified_column_name="stratum",
        n_samples=7,
        random_state=42,
    )

    counts = samples["stratum"].value_counts()
    assert len(samples) == 7
    assert set(counts.index) == {"A", "B", "C"}
    assert counts.max() - counts.min() <= 1 # strata counts are at most 1 count from each other


def test_get_stratified_random_samples_fraction(sample_data):
    '''Test sample_fraction arg for get_stratified_random_samples()'''
    samples = get_stratified_random_samples(
        sample_data,
        stratified_column_name="stratum",
        sample_fraction=0.5,
        random_state=42,
    )

    assert len(samples) == 6
    assert samples["stratum"].value_counts().to_dict() == {"A": 2, "B": 2, "C": 2}


def test_get_stratified_random_samples_is_reproducible(sample_data):
    '''Test random_state arg for get_stratified_random_samples()'''
    kwargs = {
        "stratified_column_name": "stratum",
        "n_samples": 7,
        "random_state": 17,
    }

    first = get_stratified_random_samples(sample_data, **kwargs)
    second = get_stratified_random_samples(sample_data, **kwargs)

    pd.testing.assert_frame_equal(first, second)


def test_get_stratified_random_samples_rows_are_from_dataset(sample_data):
    '''Test row data integrity for get_stratified_random_samples()'''
    samples = get_stratified_random_samples(
        sample_data,
        stratified_column_name="stratum",
        n_samples=6,
        random_state=42,
    )

    pd.testing.assert_frame_equal(samples, sample_data.loc[samples.index]) # compares indexed rows from samples to original dataset


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
    '''Test invalid arguments in get_stratified_random_samples()'''
    with pytest.raises(exception):
        get_stratified_random_samples(
            sample_data, stratified_column_name="stratum", **kwargs
        )


def test_get_stratified_random_samples_rejects_unknown_column(sample_data):
    '''Test invalid column name in get_stratified_random_samples()'''
    with pytest.raises(KeyError):
        get_stratified_random_samples(
            sample_data,
            stratified_column_name="missing",
            n_samples=3,
            random_state=42,
        )


###################################################################
#       TESTS FOR get_quantile_stratified_random_samples()        #
###################################################################

@pytest.mark.parametrize(
    "stratify_by",
    [
        {"stratify_col_name": "feature_1"},
        {"stratify_col_index": 0},
    ],
)
def test_get_quantile_stratified_random_samples_returns_balanced_quantiles(
    sample_data, stratify_by
):
    '''Test stratas are balanced in get_quantile_stratified_random_samples()'''
    samples = get_quantile_stratified_random_samples(
        sample_data,
        n_bins=3,
        n_samples=6,
        random_state=42,
        **stratify_by,
    )
    quantile_labels = pd.qcut(
        sample_data["feature_1"], q=3, labels=False, duplicates="drop"
    )

    assert len(samples) == 6
    assert quantile_labels.loc[samples.index].value_counts().to_dict() == {
        0: 2,
        1: 2,
        2: 2,
    }
    pd.testing.assert_frame_equal(samples, sample_data.loc[samples.index])


def test_get_quantile_stratified_random_samples_fraction(sample_data):
    '''Test sample_fraction arg for get_quantile_stratified_random_samples()'''
    samples = get_quantile_stratified_random_samples(
        sample_data,
        stratify_col_name="feature_1",
        n_bins=3,
        sample_fraction=0.5,
        random_state=42,
    )

    assert len(samples) == 6


def test_get_quantile_stratified_random_samples_is_reproducible(sample_data):
    '''Test random_state arg for get_quantile_stratified_random_samples()'''
    kwargs = {
        "stratify_col_name": "feature_1",
        "n_bins": 3,
        "n_samples": 7,
        "random_state": 17,
    }

    first = get_quantile_stratified_random_samples(sample_data, **kwargs)
    second = get_quantile_stratified_random_samples(sample_data, **kwargs)

    pd.testing.assert_frame_equal(first, second)


@pytest.mark.parametrize(
    "stratify_by",
    [
        {},
        {"stratify_col_index": 0, "stratify_col_name": "feature_1"},
    ],
)
def test_get_quantile_stratified_random_samples_requires_one_stratify_column(
    sample_data, stratify_by
):
    '''Test stratify column identifier is unique for get_quantile_stratified_random_samples()'''
    with pytest.raises(ValueError, match="Provide exactly one"):
        get_quantile_stratified_random_samples(
            sample_data, n_samples=3, **stratify_by
        )


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
def test_get_quantile_stratified_random_samples_rejects_invalid_sample_arguments(
    sample_data, kwargs, exception
):
    '''Test invalid sample arguments for get_quantile_stratified_random_samples()'''
    with pytest.raises(exception):
        get_quantile_stratified_random_samples(
            sample_data, stratify_col_name="feature_1", n_bins=3, **kwargs
        )


def test_get_quantile_stratified_random_samples_rejects_temporary_column_name():
    '''Test stratify column name is not equal to the temporary column name for
    get_quantile_stratified_random_samples()'''
    data = pd.DataFrame(
        {
            "value": np.arange(8),
            "__quantile_strata__": np.arange(8),
        }
    )

    with pytest.raises(ValueError, match="Temporary column name"):
        get_quantile_stratified_random_samples(
            data, stratify_col_name="value", n_samples=4, random_state=42
        )


#############################################################
#       TESTS FOR generate_latin_hypercube_samples()        #
#############################################################

def test_generate_latin_hypercube_samples_shape_and_columns(regressor_data):
    '''Test n_samples arg and columns in generate_latin_hypercube_samples()'''
    samples = generate_latin_hypercube_samples(
        regressor_data, n_samples=5, random_state=42
    )

    assert samples.shape == (5, 3)
    assert samples.columns.tolist() == regressor_data.columns.tolist()


def test_generate_latin_hypercube_samples_fraction(regressor_data):
    '''Test sample_fraction arg for generate_latin_hypercube_samples()'''
    samples = generate_latin_hypercube_samples(
        regressor_data, sample_fraction=0.4, random_state=42
    )

    assert len(samples) == 4


def test_generate_latin_hypercube_samples_values_stay_within_column_bounds(
    regressor_data,
):
    '''Test the samples are bounded by the max and min values in the
    respective column for generate_latin_hypercube_samples().'''
    samples = generate_latin_hypercube_samples(
        regressor_data, n_samples=6, random_state=42
    )

    assert samples.ge(regressor_data.min()).all().all()
    assert samples.le(regressor_data.max()).all().all()
    assert samples["constant"].eq(3.5).all()


def test_generate_latin_hypercube_samples_uses_each_quantile_interval_once(
    regressor_data,
):
    '''Test samples are in distinct stratas/quantile intervals for
    generate_latin_hypercube_samples()'''
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
    '''Test random_state arg for generate_latin_hypercube_samples()'''
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
    '''Test invalid arguments for generate_latin_hypercube_samples()'''
    with pytest.raises(exception):
        generate_latin_hypercube_samples(regressor_data, **kwargs)
