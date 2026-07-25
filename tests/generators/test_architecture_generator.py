import pytest

from bias_variance.generators.fnn_architecture import (
    FnnArchitectureGenerator,
)
from bias_variance.models.fnn.FnnArchitecture import FnnArchitecture

ARCHITECTURE_NAMES = {
    "wide",
    "narrow",
    "taper",
    "reverse_taper",
    "combined_taper",
}


def test_default_generator_returns_every_supported_architecture():
    architectures = FnnArchitectureGenerator().generate(random_state=42)

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

    assert generator.generate(random_state=42) == generator.generate(
        random_state=42
    )


@pytest.mark.parametrize(
    ("label", "settings"),
    [
        ("wide", {"layers": (2, 3), "neurons": (64, 80)}),
        ("narrow", {"layers": (3, 4), "neurons": (2, 10)}),
        (
            "taper",
            {
                "layers": (4, 5),
                "init_neurons": (4, 8),
                "taper_rate": (0.1, 0.2),
                "max_neurons": 20,
            },
        ),
        (
            "reverse_taper",
            {
                "layers": (4, 5),
                "init_neurons": (16, 20),
                "taper_rate": (0.1, 0.2),
                "max_neurons": 20,
            },
        ),
        (
            "combined_taper",
            {
                "layers": (5, 6),
                "init_neurons": (4, 8),
                "taper_rate": (0.1, 0.2),
                "max_neurons": 20,
            },
        ),
    ],
)
def test_custom_architecture_respects_layer_and_neuron_bounds(label, settings):
    architecture = FnnArchitectureGenerator({label: settings}).generate(
        random_state=42
    )[label]
    hidden_layers = architecture.hidden_layers
    max_allowed = (
        settings["max_neurons"]
        if "max_neurons" in settings
        else settings["neurons"][1] - 1
    )

    assert settings["layers"][0] <= len(hidden_layers) < settings["layers"][1]
    assert all(1 <= size <= max_allowed for size in hidden_layers)


def test_taper_is_non_decreasing():
    architecture = FnnArchitectureGenerator(
        {
            "taper": {
                "layers": (8, 9),
                "init_neurons": (4, 8),
                "taper_rate": (0.1, 0.2),
                "max_neurons": 100,
            }
        }
    ).generate(random_state=42)["taper"]
    hidden_layers = architecture.hidden_layers

    assert all(
        left <= right
        for left, right in zip(hidden_layers, hidden_layers[1:])
    )


def test_reverse_taper_is_non_increasing_and_never_zero():
    architecture = FnnArchitectureGenerator(
        {
            "reverse_taper": {
                "layers": (20, 21),
                "init_neurons": (8, 12),
                "taper_rate": (0.4, 0.5),
                "max_neurons": 20,
            }
        }
    ).generate(random_state=42)["reverse_taper"]
    hidden_layers = architecture.hidden_layers

    assert all(
        left >= right
        for left, right in zip(hidden_layers, hidden_layers[1:])
    )
    assert min(hidden_layers) == 1


def test_empty_settings_generate_no_architectures():
    assert FnnArchitectureGenerator({}).generate(random_state=42) == {}


def test_unsupported_architecture_is_rejected():
    with pytest.raises(ValueError, match="Unsupported architecture"):
        FnnArchitectureGenerator({"unknown": {}})
