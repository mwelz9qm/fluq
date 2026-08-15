import itertools

import pytest

from bias_variance.generators.fnn_architecture import (
    ArchitectureName,
    FnnArchitectureGenerator,
    FnnArchitectureGeneratorConfig,
    FnnRandomArchitectureConfig,
    FnnTaperArchitectureConfig,
)
from bias_variance.models.fnn import FnnArchitecture

ARCHITECTURE_NAMES = {
    "wide",
    "narrow",
    "taper",
    "reverse_taper",
    "combined_taper",
}


def _generate_by_label(
    generator: FnnArchitectureGenerator,
    *,
    random_state: int | None = 42,
) -> dict[str, FnnArchitecture]:
    return {
        variation.label: variation.generated
        for variation in generator.generate(random_state=random_state)
    }


def test_default_generator_returns_every_supported_architecture():
    architectures = _generate_by_label(
        FnnArchitectureGenerator(),
        random_state=42,
    )

    assert set(architectures) == ARCHITECTURE_NAMES
    assert all(
        isinstance(architecture, FnnArchitecture)
        for architecture in architectures.values()
    )
    assert all(
        isinstance(architecture.hidden_layers, tuple)
        for architecture in architectures.values()
    )
    assert all(
        isinstance(size, int)
        for architecture in architectures.values()
        for size in architecture.hidden_layers
    )


def test_generation_is_reproducible_for_same_seed():
    generator = FnnArchitectureGenerator()

    assert _generate_by_label(
        generator,
        random_state=42,
    ) == _generate_by_label(
        generator,
        random_state=42,
    )


@pytest.mark.parametrize(
    ("label", "generator", "layer_range", "max_allowed"),
    [
        (
            "wide",
            FnnArchitectureGenerator(
                FnnArchitectureGeneratorConfig(
                    range_architectures={
                        ArchitectureName.WIDE: FnnRandomArchitectureConfig(
                            layer_range=(2, 3),
                            size_range=(64, 80),
                        ),
                    },
                    taper_architectures={},
                )
            ),
            (2, 3),
            79,
        ),
        (
            "narrow",
            FnnArchitectureGenerator(
                FnnArchitectureGeneratorConfig(
                    range_architectures={
                        ArchitectureName.NARROW: FnnRandomArchitectureConfig(
                            layer_range=(3, 4),
                            size_range=(2, 10),
                        ),
                    },
                    taper_architectures={},
                )
            ),
            (3, 4),
            9,
        ),
        (
            "taper",
            FnnArchitectureGenerator(
                FnnArchitectureGeneratorConfig(
                    range_architectures={},
                    taper_architectures={
                        ArchitectureName.TAPER: FnnTaperArchitectureConfig(
                            layer_range=(4, 5),
                            start_size_range=(4, 8),
                            taper_rate_range=(0.1, 0.2),
                            max_size=20,
                        ),
                    },
                )
            ),
            (4, 5),
            20,
        ),
        (
            "reverse_taper",
            FnnArchitectureGenerator(
                FnnArchitectureGeneratorConfig(
                    range_architectures={},
                    taper_architectures={
                        ArchitectureName.REVERSE_TAPER: FnnTaperArchitectureConfig(
                            layer_range=(4, 5),
                            start_size_range=(16, 20),
                            taper_rate_range=(0.1, 0.2),
                            max_size=20,
                        ),
                    },
                )
            ),
            (4, 5),
            20,
        ),
        (
            "combined_taper",
            FnnArchitectureGenerator(
                FnnArchitectureGeneratorConfig(
                    range_architectures={},
                    taper_architectures={
                        ArchitectureName.COMBINED_TAPER: FnnTaperArchitectureConfig(
                            layer_range=(5, 6),
                            start_size_range=(4, 8),
                            taper_rate_range=(0.1, 0.2),
                            max_size=20,
                        ),
                    },
                )
            ),
            (5, 6),
            20,
        ),
    ],
)
def test_custom_architecture_respects_layer_and_neuron_bounds(
    label,
    generator,
    layer_range,
    max_allowed,
):
    architecture = _generate_by_label(
        generator,
        random_state=42,
    )[label]
    hidden_layers = architecture.hidden_layers

    assert layer_range[0] <= len(hidden_layers) < layer_range[1]
    assert all(1 <= size <= max_allowed for size in hidden_layers)


def test_taper_is_non_decreasing():
    generator = FnnArchitectureGenerator(
        FnnArchitectureGeneratorConfig(
            range_architectures={},
            taper_architectures={
                ArchitectureName.TAPER: FnnTaperArchitectureConfig(
                    layer_range=(8, 9),
                    start_size_range=(4, 8),
                    taper_rate_range=(0.1, 0.2),
                    max_size=100,
                ),
            },
        )
    )

    architecture = _generate_by_label(
        generator,
        random_state=42,
    )["taper"]
    hidden_layers = architecture.hidden_layers

    assert all(
        left <= right
        for left, right in itertools.pairwise(hidden_layers)
    )


def test_reverse_taper_is_non_increasing_and_never_zero():
    generator = FnnArchitectureGenerator(
        FnnArchitectureGeneratorConfig(
            range_architectures={},
            taper_architectures={
                ArchitectureName.REVERSE_TAPER: FnnTaperArchitectureConfig(
                    layer_range=(20, 21),
                    start_size_range=(8, 12),
                    taper_rate_range=(0.4, 0.5),
                    max_size=20,
                ),
            },
        )
    )

    architecture = _generate_by_label(
        generator,
        random_state=42,
    )["reverse_taper"]
    hidden_layers = architecture.hidden_layers

    assert all(
        left >= right
        for left, right in itertools.pairwise(hidden_layers)
    )
    assert min(hidden_layers) == 1


def test_empty_settings_generate_no_architectures():
    generator = FnnArchitectureGenerator(
        FnnArchitectureGeneratorConfig(
            range_architectures={},
            taper_architectures={},
        )
    )

    assert _generate_by_label(generator, random_state=42) == {}


def test_unsupported_architecture_is_rejected():
    with pytest.raises(
        TypeError,
        match="range_architectures keys must be ArchitectureName members",
    ):
        FnnArchitectureGeneratorConfig(
            range_architectures={"unknown": FnnRandomArchitectureConfig()},
            taper_architectures={},
        )