from types import MappingProxyType

import pytest

from bias_variance.generators.fnn_architecture import (
    ArchitectureName,
    FnnArchitectureGeneratorConfigBuilder,
)
from bias_variance.generators.noise import NoiseGeneratorConfigBuilder
from bias_variance.generators.sampling import (
    SamplingGeneratorConfigBuilder,
    SamplingStrategyName,
)


def test_fnn_builder_defaults_to_all_five_architectures() -> None:
    config = FnnArchitectureGeneratorConfigBuilder().build()

    assert config.variation_labels == (
        'wide',
        'narrow',
        'taper',
        'reverse_taper',
        'combined_taper',
    )


def test_fnn_builder_selects_a_subset_and_fills_missing_settings() -> None:
    builder = FnnArchitectureGeneratorConfigBuilder()
    config = builder.apply_settings({
        'wide': {'layer_range': (2, 4)},
        'taper': {'max_size': 512},
    }).build()

    assert config.variation_labels == ('wide', 'taper')
    assert config.range_architectures is not None
    assert config.taper_architectures is not None
    assert config.range_architectures[ArchitectureName.WIDE].layer_range == (
        2,
        4,
    )
    assert config.range_architectures[ArchitectureName.WIDE].size_range == (
        64,
        256,
    )
    assert config.taper_architectures[ArchitectureName.TAPER].max_size == 512
    assert (
        config.taper_architectures[
            ArchitectureName.TAPER
        ].taper_rate_range
        == (0.25, 0.5)
    )
    assert isinstance(config.range_architectures, MappingProxyType)


@pytest.mark.parametrize('settings', ({}, [], {'unknown': {}}))
def test_fnn_builder_rejects_invalid_settings(settings) -> None:
    error = TypeError if isinstance(settings, list) else ValueError
    with pytest.raises(error):
        FnnArchitectureGeneratorConfigBuilder().apply_settings(settings)


def test_noise_builder_supports_defaults_and_dictionary_overrides() -> None:
    default = NoiseGeneratorConfigBuilder().build()
    custom = (
        NoiseGeneratorConfigBuilder()
        .apply_settings({'standard_deviations': (0.15, 0.25)})
        .build()
    )

    assert default.standard_deviations == (0.1, 0.2, 0.3, 0.4, 0.5)
    assert custom.standard_deviations == (0.15, 0.25)


def test_noise_builder_rejects_unknown_dictionary_settings() -> None:
    with pytest.raises(ValueError, match='Unknown noise settings'):
        NoiseGeneratorConfigBuilder().apply_settings({'unknown': True})


def test_sampling_builder_defaults_to_all_implied_strategies() -> None:
    config = SamplingGeneratorConfigBuilder().build()

    assert config.variation_labels == ('bootstrap', 'stratified', 'lhs')
    for name, strategy in config.sampling_strategies.items():
        assert strategy.function is name.function
        assert 'random_state' not in strategy.kwargs


def test_sampling_builder_selects_subset_and_merges_kwargs() -> None:
    config = SamplingGeneratorConfigBuilder().apply_settings({
        'bootstrap': {'sample_fraction': 0.5},
    }).build()

    strategy = config.sampling_strategies[SamplingStrategyName.BOOTSTRAP]
    assert config.variation_labels == ('bootstrap',)
    assert strategy.kwargs == {
        'sample_fraction': 0.5,
        'with_replacement': True,
    }


def test_sampling_builder_allows_n_samples_instead_of_default_fraction() -> None:
    config = SamplingGeneratorConfigBuilder().apply_settings({
        'bootstrap': {'n_samples': 3},
    }).build()

    kwargs = config.sampling_strategies[
        SamplingStrategyName.BOOTSTRAP
    ].kwargs
    assert kwargs == {'n_samples': 3, 'with_replacement': True}


@pytest.mark.parametrize(
    'settings',
    ({}, [], {'unknown': {}}, {'bootstrap': {'random_state': 1}}),
)
def test_sampling_builder_rejects_invalid_settings(settings) -> None:
    error = TypeError if isinstance(settings, list) else ValueError
    with pytest.raises(error):
        SamplingGeneratorConfigBuilder().apply_settings(settings)
