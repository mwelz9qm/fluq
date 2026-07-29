from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np

from ..models.fnn import FnnArchitecture
from .base import Generator, Variation


class ArchitectureName(StrEnum):
    '''
    Supported architectures in config.

    Attributes:
        WIDE: Short and wide - many neurons but few layers.
        NARROW: Tall and narrow - many layers but few neurons.
        TAPER: Narrow to wide - starts with few neurons and grows to many.
        REVERSE_TAPER: Wide to narrow - starts with many neurons and shrinks to few.
        COMBINED_TAPER: Narrow to wide then returns to narrow - starts with few neurons, grows to many neurons, then returns back to few.
    '''

    WIDE = 'wide'
    NARROW = 'narrow'
    TAPER = 'taper'
    REVERSE_TAPER = 'reverse_taper'
    COMBINED_TAPER = 'combined_taper'


@dataclass(frozen=True, slots=True)
class FnnRandomArchitectureConfig:
    '''
    Represents the FNN architecture generator's ranges for randomly selecting sizes per layer.

    Attributes:
        layer_range: The range of layers.
        size_range: The range of neurons per layer.
    '''
    layer_range: tuple[int, int] | None = None
    size_range: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class FnnTaperArchitectureConfig:
    '''
    Represents the FNN architecture generator's ranges for randomly selecting sizes per layer.
    
    Attributes:
        layer_range: The range of layers.
        start_size_range: The range of starting layer sizes.
        taper_rate_range: The range of size increase or decrease rates spanning from the starting and ending layers.
        max_size: The maximum neurons for all layers.
    '''
    layer_range: tuple[int, int] | None = None
    start_size_range: tuple[int, int] | None = None
    taper_rate_range: tuple[float, float] | None = None
    max_size: int | None = None


DEFAULT_RANDOM_CONFIG = {
    ArchitectureName.WIDE: FnnRandomArchitectureConfig(
        layer_range=(1, 16),
        size_range=(64, 256),
    ),
    ArchitectureName.NARROW: FnnRandomArchitectureConfig(
        layer_range=(16, 64),
        size_range=(2, 64),
    ),
}

DEFAULT_TAPER_CONFIG = {
    ArchitectureName.TAPER: FnnTaperArchitectureConfig(
        layer_range=(16, 64),
        start_size_range=(1, 9),
        taper_rate_range=(0.25, 0.5),
        max_size=256,
    ),
    ArchitectureName.REVERSE_TAPER: FnnTaperArchitectureConfig(
        layer_range=(16, 64),
        start_size_range=(128, 256),
        taper_rate_range=(0.25, 0.5),
        max_size=256,
    ),
    ArchitectureName.COMBINED_TAPER: FnnTaperArchitectureConfig(
        layer_range=(16, 64),
        start_size_range=(1, 9),
        taper_rate_range=(0.25, 0.5),
        max_size=256,
    ),
}


@dataclass(frozen=True, slots=True)
class FnnArchitectureConfig:
    '''
    Represents the FNN architecture generator's full configurations.

    Attributes:
        range_architectures: A dictionary of selected random range architectures and their configurations.
        taper_architectures: A dictionary of selected taper range architectures and their configurations.
    '''
    range_architectures: Mapping[ArchitectureName, FnnRandomArchitectureConfig] = field(default_factory=dict)
    taper_architectures: Mapping[ArchitectureName, FnnTaperArchitectureConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        '''
        Handles configuration normalizations. Will provide defaults if user selects
        None for any attribute in FnnRandomArchitectureConfig or FnnTaperArchitectureConfig.
        Default configurations are selected if provided an empty dictionary for FnnArchitectureConfig
        attributes.
        '''
        normalized_range_config = {}
        for name, architecture in self.range_architectures.items():
            layer_range = architecture.layer_range or DEFAULT_RANDOM_CONFIG[name].layer_range
            size_range = architecture.size_range or DEFAULT_RANDOM_CONFIG[name].size_range
            normalized_range_config[name] = FnnRandomArchitectureConfig(layer_range, size_range)

        if normalized_range_config is None:
            normalized_range_config = DEFAULT_RANDOM_CONFIG

        normalized_taper_config = {}
        for name, architecture in self.taper_architectures.items():
            layer_range = architecture.layer_range or DEFAULT_TAPER_CONFIG[name].layer_range
            start_size_range = architecture.start_size_range or DEFAULT_TAPER_CONFIG[name].start_size_range
            taper_rate_range = architecture.taper_rate_range or DEFAULT_TAPER_CONFIG[name].taper_rate_range
            max_size = architecture.max_size or DEFAULT_TAPER_CONFIG[name].max_size
            normalized_taper_config[name] = FnnTaperArchitectureConfig(layer_range, start_size_range, taper_rate_range, max_size)

        if normalized_taper_config is None:
            normalized_taper_config = DEFAULT_TAPER_CONFIG

        object.__setattr__(self, 'range_architectures', normalized_range_config)
        object.__setattr__(self, 'taper_architectures', normalized_taper_config)


@dataclass(frozen=True, slots=True)
class FnnArchitectureVariation(Variation[FnnArchitecture]):
    '''
    Represents the generated variation returned by FnnArchitectureGenerator.
    
    Attributes:
        architecture: The generated architecture.
    '''
    @property
    def architecture(self) -> FnnArchitecture:
        """The generated architecture (kept as a compatibility alias)."""
        return self.generated


class FnnArchitectureGenerator(Generator[FnnArchitecture]):
    '''
    Generates FNN architectures that describe a model's hidden layer structure.

    Attributes
    --------------
    settings: FnnArchitectureConfig | None, default = None
        The generator's settings for configuring architecture ranges, rates, and maximums.
    '''
    def __init__(
        self,
        settings: FnnArchitectureConfig | None = None,
    ) -> None:
        self.settings = settings or FnnArchitectureConfig()

    def _generate_random_sizes(
        self,
        config: FnnRandomArchitectureConfig,
        rng: np.random.Generator,
    ) -> FnnArchitecture:
        '''
        Generates hidden layer sizes based on ranges for the number of layers and neurons.

        Parameters
        -----------
        config: FnnRandomArchitectureConfig
            The wide or narrow architecture's ranges.
        rng: np.random.Generator
            The random seed for reproducibility.
        
        Returns
        --------
        FnnArchitecture
            An FNN architecture.
        '''
        low_layers, high_layers = config.layer_range
        n_layers = rng.integers(low_layers, high_layers, dtype=int)
        low_size, high_size = config.size_range

        return FnnArchitecture(
            hidden_layers=tuple(
                int(size)
                for size in rng.integers(
                    low_size,
                    high_size,
                    size=n_layers
                )
            )
        )

    def _generate_taper_sizes(
        self,
        config: FnnTaperArchitectureConfig,
        taper_type: ArchitectureName,
        rng: np.random.Generator,
    ) -> FnnArchitecture:
        '''
        Generates hidden layer sizes based on ranges for the number of layers, number of neurons
        for the first layer, and selection of taper rates, and maximum neurons for all layers.

        The method uses the taper_type to build the correct taper architecture. The rng object is
        used to set the random seed when selecting the number of layers (n_layers), the starting
        layer size (start_size), and size increase or decrease rate (size_rate).
        
        Parameters
        -----------
        config: FnnRandomArchitectureConfig
            The taper, reverse taper, or combined taper architecture's ranges and maximum.
        taper_type: ArchitectureName
            The name of the architecture type i.e., 'taper', 'reverse_taper', or 'combined_taper'.
        rng: np.random.Generator
            The random seed for reproducibility.
        
        Returns
        --------
        FnnArchitecture
            An FNN architecture.
        '''
        low_layers, high_layers = config.layer_range
        n_layers = rng.integers(low_layers, high_layers, dtype=int)

        low_size, high_size = config.start_size_range
        start_size = rng.integers(low_size, high_size, dtype=int)

        low_taper_rate, high_taper_rate = config.taper_rate_range
        if taper_type == ArchitectureName.TAPER:
            size_rate = rng.uniform(low_taper_rate + 1, high_taper_rate + 1)

        elif taper_type == ArchitectureName.REVERSE_TAPER:
            size_rate = rng.uniform(high_taper_rate - 1, low_taper_rate - 1)

        else:
            size_rate = rng.uniform(low_taper_rate, high_taper_rate)
        

        sizes = []

        if taper_type == ArchitectureName.TAPER or ArchitectureName.REVERSE_TAPER:
            for i in np.arange(n_layers):
                size = round(start_size * (size_rate ** i))
                sizes.append(min(max(size, 1), config.max_size))

        else:
            midpoint = max(1, int(np.ceil(n_layers / 2)))
            for i in np.arange(n_layers):
                if i < midpoint:
                    size = round(start_size * ((1 + size_rate) ** i))
                else:
                    peak = start_size * ((1 + size_rate) ** (midpoint  - 1))
                    size = round(peak * ((1 - size_rate) ** (i - midpoint + 1)))
                sizes.append(min(max(size, 1), config.max_size))

        return FnnArchitecture(
            hidden_layers=tuple(sizes)
        )

    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> list[Variation[FnnArchitecture]]:
        '''
        Generates an FNN architecture.

        This method iterates through two main groups of architecture types: Random and Taper. Each group generates
        an architecture with either random layers and sizes per layer, or random layers, starting layer size and
        size increase or decrease rate per layer with a set maximum layer size.  

        Parameters
        ------------
        random_state: int | None, default = None
            An integer to set a random seed if provided.
        
        Returns
        ----------
        dict[str, FnnArchitecture]
            A dictionary of FNN architectures with string keys for identification.
        '''
        rng = np.random.default_rng(random_state)
        variations = []

        for name, config in self.settings.range_architectures.items():
            variation = FnnArchitectureVariation(
                label=name.value,
                random_state=random_state,
                generated=self._generate_random_sizes(config, rng)
            )
            variations.append(variation)

        for name, config in self.settings.taper_architectures.items():
            variation = FnnArchitectureVariation(
                label=name.value,
                random_state=random_state,
                generated=self._generate_taper_sizes(config, name, rng)
            )
            variations.append(variation)
        
        return variations
