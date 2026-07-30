import pytest

from bias_variance.generators.fnn_architecture import (
    ArchitectureName,
    FnnArchitectureConfig,
    FnnArchitectureVariation,
    FnnRandomArchitectureConfig,
    FnnTaperArchitectureConfig,
)
from bias_variance.models.fnn import FnnArchitecture


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


@pytest.mark.parametrize(
    ('ranges', 'message'),
    [
        ({'layer_range': (1, 8)}, 'upper bound of layer_range'),
        ({'start_size_range': (4, 10)}, 'upper bound of start_size_range'),
    ],
)
def test_max_size_must_cover_integer_range_upper_bounds(ranges, message):
    with pytest.raises(ValueError, match=message):
        FnnTaperArchitectureConfig(max_size=7, **ranges)


def test_architecture_collections_must_be_mappings():
    with pytest.raises(TypeError, match='range_architectures must be a mapping'):
        FnnArchitectureConfig(range_architectures=[])


def test_architecture_mapping_keys_must_be_architecture_names():
    with pytest.raises(TypeError, match='keys must be ArchitectureName'):
        FnnArchitectureConfig(
            range_architectures={'wide': FnnRandomArchitectureConfig()}
        )


def test_architecture_names_must_be_in_the_correct_mapping():
    with pytest.raises(ValueError, match='not supported'):
        FnnArchitectureConfig(
            range_architectures={
                ArchitectureName.TAPER: FnnRandomArchitectureConfig()
            }
        )


def test_architecture_mapping_values_must_have_the_correct_config_type():
    with pytest.raises(TypeError, match='must be a FnnRandomArchitectureConfig'):
        FnnArchitectureConfig(
            range_architectures={
                ArchitectureName.WIDE: FnnTaperArchitectureConfig()
            }
        )


def test_wide_and_narrow_layer_ranges_must_be_disjoint_and_ordered():
    with pytest.raises(ValueError, match='layer_range values must be disjoint'):
        FnnArchitectureConfig(
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
        FnnArchitectureConfig(
            range_architectures={
                ArchitectureName.WIDE: FnnRandomArchitectureConfig(
                    size_range=(63, 256),
                ),
                ArchitectureName.NARROW: FnnRandomArchitectureConfig(
                    size_range=(2, 64),
                ),
            }
        )


def test_default_random_config_satisfies_cross_architecture_rules():
    config = FnnArchitectureConfig(
        range_architectures={
            ArchitectureName.WIDE: FnnRandomArchitectureConfig(),
            ArchitectureName.NARROW: FnnRandomArchitectureConfig(),
        }
    )

    assert config.range_architectures[ArchitectureName.WIDE].layer_range == (
        1,
        16,
    )
    assert config.range_architectures[ArchitectureName.NARROW].size_range == (
        2,
        64,
    )


@pytest.mark.parametrize(
    ('arguments', 'error', 'message'),
    [
        (
            {'label': 1, 'random_state': None, 'generated': FnnArchitecture(())},
            TypeError,
            'label must be a string',
        ),
        (
            {'label': '', 'random_state': None, 'generated': FnnArchitecture(())},
            ValueError,
            'label must not be empty',
        ),
        (
            {'label': 'wide', 'random_state': True, 'generated': FnnArchitecture(())},
            TypeError,
            'random_state must be an integer',
        ),
        (
            {'label': 'wide', 'random_state': None, 'generated': object()},
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
