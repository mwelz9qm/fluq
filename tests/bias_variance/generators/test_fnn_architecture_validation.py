import itertools
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from bias_variance.generators.fnn_architecture import (
    ArchitectureName,
    FnnArchitectureGenerator,
    FnnArchitectureGeneratorConfig,
    FnnArchitectureVariation,
    FnnRandomArchitectureConfig as RandomArchitectureConfig,
    FnnTaperArchitectureConfig as TaperArchitectureConfig,
)
from bias_variance.models.fnn import FnnArchitecture


def FnnRandomArchitectureConfig(**overrides):
    settings = {
        'layer_range': (1, 4),
        'size_range': (64, 256),
    }
    settings.update(overrides)
    return RandomArchitectureConfig(**settings)


def FnnTaperArchitectureConfig(**overrides):
    settings = {
        'layer_range': (1, 16),
        'start_size_range': (1, 9),
        'taper_rate_range': (0.25, 0.5),
        'max_size': 256,
    }
    settings.update(overrides)
    return TaperArchitectureConfig(**settings)


@pytest.mark.parametrize('name', ('layer_range', 'size_range'))
def test_random_ranges_must_be_tuples(name):
    with pytest.raises(TypeError, match=rf'{name} must be a tuple'):
        FnnRandomArchitectureConfig(**{name: [1, 2]})


@pytest.mark.parametrize(
    ('value', 'error', 'message'),
    [
        ((1,), ValueError, 'exactly two bounds'),
        ((1, True), TypeError, 'bounds must be integers'),
        ((0, 2), ValueError, 'lower bound must be at least 1'),
        ((2, 2), ValueError, 'lower bound must be less'),
        ((3, 2), ValueError, 'lower bound must be less'),
    ],
)
def test_random_ranges_validate_shape_types_and_values(
    value,
    error,
    message,
):
    with pytest.raises(error, match=message):
        FnnRandomArchitectureConfig(layer_range=value)


@pytest.mark.parametrize(
    ('value', 'error', 'message'),
    [
        ([0.1, 0.2], TypeError, 'must be a tuple'),
        ((0.1,), ValueError, 'exactly two bounds'),
        ((0.1, True), TypeError, 'bounds must be numeric'),
        ((0.0, 0.2), ValueError, '0 < lower < upper < 1'),
        ((0.2, 0.2), ValueError, '0 < lower < upper < 1'),
        ((0.3, 0.2), ValueError, '0 < lower < upper < 1'),
        ((0.2, 1.0), ValueError, '0 < lower < upper < 1'),
    ],
)
def test_taper_rate_range_validates_shape_types_and_values(
    value,
    error,
    message,
):
    with pytest.raises(error, match=message):
        FnnTaperArchitectureConfig(taper_rate_range=value)


def test_taper_integer_ranges_use_random_range_validation():
    with pytest.raises(ValueError, match='start_size_range lower bound'):
        FnnTaperArchitectureConfig(start_size_range=(0, 2))


@pytest.mark.parametrize('value', (True, 1.5, '10'))
def test_max_size_must_be_an_integer(value):
    with pytest.raises(TypeError, match='max_size must be an integer'):
        FnnTaperArchitectureConfig(max_size=value)


def test_max_size_must_be_positive():
    with pytest.raises(ValueError, match='max_size must be at least 1'):
        FnnTaperArchitectureConfig(max_size=0)


def test_max_size_need_not_cover_layer_range_upper_bound():
    config = FnnTaperArchitectureConfig(
        layer_range=(1, 8),
        start_size_range=(1, 7),
        max_size=7,
    )

    assert config.max_size == 7


def test_max_size_must_cover_start_size_range_upper_bound():
    with pytest.raises(ValueError, match='upper bound of start_size_range'):
        FnnTaperArchitectureConfig(
            start_size_range=(4, 10),
            max_size=7,
        )


def test_architecture_collections_must_be_mappings():
    with pytest.raises(TypeError, match='range_architectures must be a mapping'):
        FnnArchitectureGeneratorConfig(range_architectures=[])


def test_architecture_mapping_keys_must_be_architecture_names():
    with pytest.raises(TypeError, match='keys must be ArchitectureName'):
        FnnArchitectureGeneratorConfig(
            range_architectures={'wide': FnnRandomArchitectureConfig()}
        )


def test_architecture_names_must_be_in_the_correct_mapping():
    with pytest.raises(ValueError, match='not supported'):
        FnnArchitectureGeneratorConfig(
            range_architectures={
                ArchitectureName.TAPER: FnnRandomArchitectureConfig()
            }
        )


def test_architecture_mapping_values_must_have_the_correct_config_type():
    with pytest.raises(TypeError, match='must be a FnnRandomArchitectureConfig'):
        FnnArchitectureGeneratorConfig(
            range_architectures={
                ArchitectureName.WIDE: FnnTaperArchitectureConfig()
            }
        )


def test_wide_and_narrow_layer_ranges_must_be_disjoint_and_ordered():
    with pytest.raises(ValueError, match='layer_range values must be disjoint'):
        FnnArchitectureGeneratorConfig(
            range_architectures={
                ArchitectureName.WIDE: FnnRandomArchitectureConfig(
                    layer_range=(1, 17),
                ),
                ArchitectureName.NARROW: FnnRandomArchitectureConfig(
                    layer_range=(16, 64),
                ),
            }
        )


def test_wide_and_narrow_size_ranges_must_be_disjoint_and_ordered():
    with pytest.raises(ValueError, match='size_range values must be disjoint'):
        FnnArchitectureGeneratorConfig(
            range_architectures={
                ArchitectureName.WIDE: FnnRandomArchitectureConfig(
                    layer_range=(1, 4),
                    size_range=(63, 256),
                ),
                ArchitectureName.NARROW: FnnRandomArchitectureConfig(
                    layer_range=(4, 16),
                    size_range=(2, 64),
                ),
            }
        )


def test_default_random_config_satisfies_cross_architecture_rules():
    config = FnnArchitectureGeneratorConfig(
        range_architectures={
            ArchitectureName.WIDE: FnnRandomArchitectureConfig(),
            ArchitectureName.NARROW: FnnRandomArchitectureConfig(
                layer_range=(4, 16),
                size_range=(2, 64),
            ),
        }
    )

    assert config.range_architectures[ArchitectureName.WIDE].layer_range == (
        1,
        4,
    )
    assert config.range_architectures[ArchitectureName.NARROW].size_range == (
        2,
        64,
    )


def test_default_generator_uses_every_predefined_configuration():
    generator = FnnArchitectureGenerator()

    assert set(generator.settings.range_architectures) == {
        ArchitectureName.WIDE,
        ArchitectureName.NARROW,
    }
    assert set(generator.settings.taper_architectures) == {
        ArchitectureName.TAPER,
        ArchitectureName.REVERSE_TAPER,
        ArchitectureName.COMBINED_TAPER,
    }


def test_architecture_config_mappings_and_values_are_immutable():
    source = {
        ArchitectureName.WIDE: FnnRandomArchitectureConfig(),
    }
    config = FnnArchitectureGeneratorConfig(range_architectures=source)
    source.clear()

    assert ArchitectureName.WIDE in config.range_architectures
    with pytest.raises(TypeError):
        config.range_architectures[ArchitectureName.NARROW] = (
            FnnRandomArchitectureConfig()
        )
    with pytest.raises(FrozenInstanceError):
        config.range_architectures[
            ArchitectureName.WIDE
        ].layer_range = (2, 3)


def test_unknown_taper_type_is_rejected():
    config = FnnTaperArchitectureConfig(
        layer_range=(2, 3),
        start_size_range=(4, 5),
        taper_rate_range=(0.2, 0.3),
        max_size=5,
    )

    with pytest.raises(ValueError, match='Unknown taper_type'):
        FnnArchitectureGenerator()._generate_taper_sizes(
            config,
            ArchitectureName.WIDE,
            np.random.default_rng(42),
        )


def test_reverse_taper_uses_a_positive_decay_multiplier():
    config = FnnTaperArchitectureConfig(
        layer_range=(8, 9),
        start_size_range=(20, 21),
        taper_rate_range=(0.2, 0.4),
        max_size=21,
    )
    generator = FnnArchitectureGenerator(
        FnnArchitectureGeneratorConfig(
            range_architectures={},
            taper_architectures={
                ArchitectureName.REVERSE_TAPER: config,
            }
        )
    )

    architecture = next(generator.generate(random_state=42)).generated

    assert architecture.hidden_layers[0] == 20
    assert all(
        left >= right >= 1
        for left, right in zip(
            architecture.hidden_layers,
            architecture.hidden_layers[1:],
        )
    )


def test_combined_taper_grows_and_then_shrinks():
    config = FnnTaperArchitectureConfig(
        layer_range=(7, 8),
        start_size_range=(4, 5),
        taper_rate_range=(0.2, 0.3),
        max_size=100,
    )
    generator = FnnArchitectureGenerator(
        FnnArchitectureGeneratorConfig(
            range_architectures={},
            taper_architectures={
                ArchitectureName.COMBINED_TAPER: config,
            }
        )
    )

    hidden_layers = next(
        generator.generate(random_state=42)
    ).generated.hidden_layers
    midpoint = 4

    assert all(
        left <= right
        for left, right in itertools.pairwise(hidden_layers[:midpoint])
    )
    assert all(
        left >= right
        for left, right in zip(
            hidden_layers[midpoint - 1:],
            hidden_layers[midpoint:],
        )
    )
    assert hidden_layers[0] < hidden_layers[midpoint - 1]
    assert hidden_layers[-1] < hidden_layers[midpoint - 1]


@pytest.mark.parametrize(
    ('arguments', 'error', 'message'),
    [
        (
            {'label': 1, 'variation_seed': 1, 'generated': FnnArchitecture(())},
            TypeError,
            'label must be a string',
        ),
        (
            {'label': '', 'variation_seed': 1, 'generated': FnnArchitecture(())},
            ValueError,
            'label cannot be empty string or whitespace',
        ),
        (
            {'label': 'wide', 'variation_seed': True, 'generated': FnnArchitecture(())},
            TypeError,
            'variation_seed must be an integer',
        ),
        (
            {'label': 'wide', 'variation_seed': 1, 'generated': object()},
            TypeError,
            'generated must be an FnnArchitecture',
        ),
    ],
)
def test_architecture_variation_validates_inherited_fields(
    arguments,
    error,
    message,
):
    with pytest.raises(error, match=message):
        FnnArchitectureVariation(**arguments)
