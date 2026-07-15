from .Generator import Generator
from collections.abc import Mapping
import numpy as np


class ArchitectureGenerator(Generator[tuple[int, ...]]):
    def __init__(
        self,
        settings: Mapping[str, Mapping[str, object]] | None = None,
    ) -> None:
        self.settings = self._normalize_settings(settings)

    def _normalize_settings(self, settings: dict | None = None):
        supported = {
                'wide',
                'narrow',
                'taper',
                'reverse_taper',
                'combined_taper'
            }
        default_settings = {
                'wide': {
                    'layers': (1, 16),
                    'neurons': (64, 256),
                },
                'narrow': {
                    'layers': (16, 64),
                    'neurons': (2, 64),
                },
                'taper': {
                    'layers': (16, 64),
                    'init_neurons': (1, 9),
                    'taper_rate': (0.25, 0.5),
                    'max_neurons': 256,
                },
                'reverse_taper': {
                    'layers': (16, 64),
                    'init_neurons': (128, 256),
                    'taper_rate': (0.25, 0.5),
                    'max_neurons': 256,
                },
                'combined_taper': {
                    'layers': (16, 64),
                    'init_neurons': (1, 9),
                    'taper_rate': (0.25, 0.5),
                    'max_neurons': 256,
                },
            }
        selected_settings = default_settings if settings is None else settings
        normalized_settings = {}
        for architecture_name, overrides in selected_settings.items():
            if architecture_name not in supported: raise ValueError(
                f'Unsupported architecture: {architecture_name}'
            )
            normalized_settings[architecture_name] = (
                default_settings[architecture_name]
                | (overrides or {})
            )
        return normalized_settings

    def _generate_random_sizes(
        self,
        n_layers,
        low_neurons,
        high_neurons,
        rng
    ) -> tuple[int, ...]:
        return tuple(rng.integers(low_neurons, high_neurons, size=n_layers))

    def _generate_taper_sizes(
        self,
        n_layers,
        init_neurons,
        low_size_rate,
        high_size_rate,
        max_neurons,
        rng
    ) -> tuple[int, ...]:
        low_neurons, high_neurons = init_neurons
        first_layer_size = rng.integers(low_neurons, high_neurons)
        size_rate = rng.uniform(low_size_rate, high_size_rate)
        sizes = []
        for i in np.arange(n_layers):
            size = round(first_layer_size * (size_rate ** i))
            sizes.append(min(max(size, 1), max_neurons))
        return tuple(sizes)
    
    def _generate_combined_taper_sizes(
        self,
        n_layers,
        init_neurons,
        taper_rate,
        max_neurons,
        rng
    ) -> tuple[int, ...]:
        low_neurons, high_neurons = init_neurons
        first_layer_size = rng.integers(low_neurons, high_neurons)
        rate = rng.uniform(*taper_rate)
        sizes = []
        midpoint = max(1, int(np.ceil(n_layers / 2)))
        for i in np.arange(n_layers):
            if i < midpoint:
                size = round(first_layer_size * ((1 + rate) ** i))
            else:
                peak = first_layer_size * ((1 + rate) ** (midpoint  - 1))
                size = round(peak * ((1 - rate) ** (i - midpoint + 1)))
            sizes.append(min(max(size, 1), max_neurons))
        return tuple(sizes)
    
    def _generate_architecture(
        self,
        label: str,
        settings: Mapping[str, object],
        rng: np.random.Generator
    ) -> tuple[int, ...]:
        low_layers, high_layers = settings['layers']
        n_layers = rng.integers(low_layers, high_layers, dtype=int)

        match label:
            case 'wide' | 'narrow':
                low_neurons, high_neurons = settings['neurons']
                return self._generate_random_sizes(
                    n_layers,
                    low_neurons,
                    high_neurons,
                    rng
                )
            
            case 'taper':
                low_rate, high_rate = settings['taper_rate']
                return self._generate_taper_sizes(
                    n_layers,
                    settings['init_neurons'],
                    1 + low_rate,
                    1 + high_rate,
                    settings['max_neurons'],
                    rng
                )
            
            case 'reverse_taper':
                low_rate, high_rate = settings['taper_rate']
                return self._generate_taper_sizes(
                    n_layers,
                    settings['init_neurons'],
                    1 - high_rate,
                    1 - low_rate,
                    settings['max_neurons'],
                    rng
                )
            
            case 'combined_taper':
                return self._generate_combined_taper_sizes(
                    n_layers,
                    settings['init_neurons'],
                    settings['taper_rate'],
                    settings['max_neurons'],
                    rng
                )
            
            case _:
                raise ValueError(f'Unsupported architecture: {label!r}')

    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> dict[str, tuple[int, ...]]:
        rng = np.random.default_rng(random_state)
        architectures = {}
        
        for label, settings in self.settings.items():
            sizes = self._generate_architecture(label, settings, rng)
            architectures[label] = tuple(int(size) for size in sizes)
        
        return architectures
    