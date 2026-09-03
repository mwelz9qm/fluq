'''
NOTE: The Sampler accepts same-labelled tuples. This may cause undesirable results
for the BiasAnalyzer. Need to decide to keep duplicates or enforce unique labelling.
'''

import pandas as pd
import pytest

from common.sampling.Sampler import Sampler

from common.sampling._sampling import (
    generate_latin_hypercube_samples,
    get_random_samples
)


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        'feature_a': range(120),
        'feature_b': range(120, 240),
        'target': range(240, 360),
    })

def test_constructor():
    strategies = [
        ('bootstrap', get_random_samples, {
            'n_samples': 101,
            'random_state': 42,
            'with_replacement': True
        }),
        ('lhs', generate_latin_hypercube_samples, {
            'n_samples': 101,
            'random_state': 42
        })
    ]
    sampler = Sampler(strategies)

    assert sampler.strategies == strategies

    empty_sampler = Sampler()

    assert empty_sampler.strategies == []

def test_add_strategy():
    sampler = Sampler()
    sampler = (
        sampler
        .add_strategy('bootstrap', get_random_samples, n_samples=101, random_state=42, with_replacement=True)
        .add_strategy('lhs', generate_latin_hypercube_samples, n_samples=101, random_state=42)
    )
    assert sampler.strategies == [
        ('bootstrap', get_random_samples, {
            'n_samples': 101,
            'random_state': 42,
            'with_replacement': True
        }),
        ('lhs', generate_latin_hypercube_samples, {
            'n_samples': 101,
            'random_state': 42
        })
    ]
    sampler.add_strategy('bootstrap', get_random_samples, n_samples=101, random_state=42, with_replacement=True)
    assert sampler.strategies == [
        ('bootstrap', get_random_samples, {
            'n_samples': 101,
            'random_state': 42,
            'with_replacement': True
        }),
        ('lhs', generate_latin_hypercube_samples, {
            'n_samples': 101,
            'random_state': 42
        }),
        ('bootstrap', get_random_samples, {
            'n_samples': 101,
            'random_state': 42,
            'with_replacement': True
        })
    ] # Sampler can accept strategies with same labels

def test_remove_strategy():
    strategies = [
        ('bootstrap', get_random_samples, {
            'n_samples': 101,
            'random_state': 42,
            'with_replacement': True
        }),
        ('lhs', generate_latin_hypercube_samples, {
            'n_samples': 101,
            'random_state': 42
        }),
        ('bootstrap', get_random_samples, {
            'n_samples': 101,
            'random_state': 42,
            'with_replacement': True
        })
    ]
    sampler = Sampler(strategies)

    sampler.remove_strategy('other_sampling_func')
    assert sampler.strategies == strategies

    sampler.remove_strategy('bootstrap') # removes all strategies w/ same label
    assert sampler.strategies == [
        ('lhs', generate_latin_hypercube_samples, {
            'n_samples': 101,
            'random_state': 42
        })
    ]

def test_generate(sample_df):
    strategies = [
        ('bootstrap', get_random_samples, {
            'n_samples': 101,
            'random_state': 42,
            'with_replacement': True
        }),
        ('lhs', generate_latin_hypercube_samples, {
            'n_samples': 101,
            'random_state': 42
        })
    ]
    sampler = Sampler(strategies)

    results = sampler.generate(sample_df)

    bootstrap_df = get_random_samples(sample_df, n_samples=101, random_state=42, with_replacement=True)
    lhs_df = generate_latin_hypercube_samples(sample_df, n_samples=101, random_state=42)

    pd.testing.assert_frame_equal(results['bootstrap'], bootstrap_df)
    pd.testing.assert_frame_equal(results['lhs'], lhs_df)

def test_generate_injected_kwargs(sample_df):
    strategies = [
        ('bootstrap', get_random_samples, {
            'n_samples': 101,
            'with_replacement': True
        }),
        ('lhs', generate_latin_hypercube_samples, {
            'n_samples': 101
        })
    ]
    sampler = Sampler(strategies)

    results = sampler.generate(sample_df, {'random_state': 42})  # applies kwarg to all sampling funcs

    bootstrap_df = get_random_samples(sample_df, n_samples=101, random_state=42, with_replacement=True)
    lhs_df = generate_latin_hypercube_samples(sample_df, n_samples=101, random_state=42)

    pd.testing.assert_frame_equal(results['bootstrap'], bootstrap_df)
    pd.testing.assert_frame_equal(results['lhs'], lhs_df)
