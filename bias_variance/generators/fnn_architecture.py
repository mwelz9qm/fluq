from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Real
from types import MappingProxyType

import numpy as np

from ..models.fnn import FnnArchitecture
from .base import Generator, GeneratorConfig, Variation


class ArchitectureName(StrEnum):
    '''
    Architectures supported in the configuration.

    Attributes
    ------------
        WIDE
            Short and wide: many neurons but few layers.
        NARROW
            Tall and narrow: many layers but few neurons.
        TAPER
            Narrow to wide: starts with few neurons and grows to many.
        REVERSE_TAPER
            Wide to narrow: starts with many neurons and shrinks to few.
        COMBINED_TAPER
            Narrow to wide and then back to narrow: starts with few neurons,
            grows to many, and then returns to few.
    '''

    WIDE = 'wide'
    NARROW = 'narrow'
    TAPER = 'taper'
    REVERSE_TAPER = 'reverse_taper'
    COMBINED_TAPER = 'combined_taper'


@dataclass(frozen=True, slots=True)
class FnnRandomArchitectureConfig:
    '''
    Represents the ranges used by the FNN architecture generator to randomly
    select layer sizes.

    All ranges are half-open: the lower bound is inclusive, and the upper bound
    is exclusive.

    Each range must be a two-item tuple of integers with
    ``1 <= lower < upper``. Boolean bounds are not accepted. A range may be
    ``None`` so that ``FnnArchitectureGeneratorConfig`` can supply its default.

    Attributes
    ------------
        layer_range: tuple[int, int] | None, default = None
            The range of layers.
        size_range: tuple[int, int] | None, default = None
            The range of neurons per layer.
    '''
    layer_range: tuple[int, int] | None = None
    size_range: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        '''
        Validate the types, lengths, bounds, and ordering of both ranges.
        '''
        self._validate_integer_range('layer_range', self.layer_range)
        self._validate_integer_range('size_range', self.size_range)

    @staticmethod
    def _validate_integer_range(
        name: str,
        value: tuple[int, int] | None,
    ) -> None:
        '''
        Validate an optional two-item integer range.

        The lower bound must be at least 1 and strictly less than the upper
        bound. Boolean bounds are rejected even though ``bool`` is a subclass
        of ``int``.
        '''
        if value is None:
            return

        if not isinstance(value, tuple):
            raise TypeError(f'{name} must be a tuple or None.')

        if len(value) != 2:
            raise ValueError(f'{name} must contain exactly two bounds.')

        if any(
            not isinstance(bound, int) or isinstance(bound, bool)
            for bound in value
        ):
            raise TypeError(f'{name} bounds must be integers.')

        lower, upper = value
        if lower < 1:
            raise ValueError(f'{name} lower bound must be at least 1.')

        if lower >= upper:
            raise ValueError(
                f'{name} lower bound must be less than its upper bound.'
            )


@dataclass(frozen=True, slots=True)
class FnnTaperArchitectureConfig:
    '''
    Represents the ranges used by the FNN architecture generator to randomly
    select layer sizes.

    All ranges are half-open: the lower bound is inclusive, and the upper bound
    is exclusive. The reverse-taper rate range is the exception: its lower bound
    is exclusive, and its upper bound is inclusive. ``max_size`` is not a range
    bound; it is an inclusive limit.

    ``layer_range`` and ``start_size_range`` follow the integer-range validation
    rules in ``FnnRandomArchitectureConfig``. ``taper_rate_range`` must contain
    exactly two numeric, non-boolean bounds satisfying
    ``0 < lower < upper < 1``. ``max_size`` must be a positive, non-boolean
    integer at least as large as the upper bound of ``start_size_range``, when
    that range is provided. Any attribute may be ``None`` so that
    ``FnnArchitectureGeneratorConfig`` can supply its default.
    
    Attributes
    -----------
        layer_range: tuple[int, int] | None, default = None
            The range of layers.
        start_size_range: tuple[int, int] | None, default = None
            The range of starting layer sizes.
        taper_rate_range: tuple[float, float] | None, default = None
            The range of rates at which layer sizes increase or decrease between
            the starting and ending layers.
        max_size: int | None, default = None
            The maximum number of neurons in any layer.
    '''
    layer_range: tuple[int, int] | None = None
    start_size_range: tuple[int, int] | None = None
    taper_rate_range: tuple[float, float] | None = None
    max_size: int | None = None

    def __post_init__(self) -> None:
        '''
        Validate all taper configuration types, bounds, and relationships.
        '''
        FnnRandomArchitectureConfig._validate_integer_range(
            'layer_range',
            self.layer_range,
        )
        FnnRandomArchitectureConfig._validate_integer_range(
            'start_size_range',
            self.start_size_range,
        )
        self._validate_taper_rate_range()
        self._validate_max_size()

    def _validate_taper_rate_range(self) -> None:
        '''
        Validate that the optional taper-rate bounds lie strictly between 0 and 1.
        '''
        value = self.taper_rate_range
        if value is None:
            return

        if not isinstance(value, tuple):
            raise TypeError('taper_rate_range must be a tuple or None.')

        if len(value) != 2:
            raise ValueError(
                'taper_rate_range must contain exactly two bounds.'
            )

        if any(
            not isinstance(bound, Real) or isinstance(bound, bool)
            for bound in value
        ):
            raise TypeError('taper_rate_range bounds must be numeric.')

        lower, upper = value
        if not 0 < lower < upper < 1:
            raise ValueError(
                'taper_rate_range bounds must satisfy 0 < lower < upper < 1.'
            )

    def _validate_max_size(self) -> None:
        '''
        Validate the type, minimum, and start-size coverage of ``max_size``.
        '''
        if self.max_size is None:
            return

        if not isinstance(self.max_size, int) or isinstance(self.max_size, bool):
            raise TypeError('max_size must be an integer or None.')

        if self.max_size < 1:
            raise ValueError('max_size must be at least 1.')

        if (
            self.start_size_range is not None
            and self.max_size < self.start_size_range[1]
        ):
            raise ValueError(
                'max_size must be at least the upper bound of '
                'start_size_range.'
            )


DEFAULT_RANDOM_CONFIG = MappingProxyType({
    ArchitectureName.WIDE: FnnRandomArchitectureConfig(
        layer_range=(1, 16),
        size_range=(64, 256),
    ),
    ArchitectureName.NARROW: FnnRandomArchitectureConfig(
        layer_range=(16, 64),
        size_range=(2, 64),
    ),
})

DEFAULT_TAPER_CONFIG = MappingProxyType({
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
})


@dataclass(frozen=True, slots=True)
class FnnArchitectureGeneratorConfig(GeneratorConfig):
    '''
    Represents the complete configuration for the FNN architecture generator.

    All ranges in the contained configurations are half-open: the lower bound is
    inclusive, and the upper bound is exclusive. The reverse-taper rate range
    instead excludes its lower bound and includes its upper bound.

    Both architecture collections must be mappings with ``ArchitectureName``
    keys and configuration values of the appropriate type. Random architectures
    support only ``WIDE`` and ``NARROW``; taper architectures support only
    ``TAPER``, ``REVERSE_TAPER``, and ``COMBINED_TAPER``.

    When both ``WIDE`` and ``NARROW`` are configured, their corresponding ranges
    must be disjoint. The WIDE layer range must be below the NARROW layer range,
    while the NARROW size range must be below the WIDE size range. Because these
    ranges are half-open, adjacent ranges are valid.

    The mappings are copied and exposed as read-only mapping proxies after
    normalization. Their values are frozen dataclasses containing only immutable
    tuples and scalar values, so neither the mappings nor their contents can be
    changed through this configuration.

    Attributes
    -------------
        range_architectures: Mapping[ArchitectureName, FnnRandomArchitectureConfig], default = dict()
            A read-only mapping of selected random-range architectures and their
            configurations after initialization.
        taper_architectures: Mapping[ArchitectureName, FnnTaperArchitectureConfig], default = dict()
            A read-only mapping of selected taper architectures and their
            configurations after initialization.
    '''
    range_architectures: Mapping[
        ArchitectureName,
        FnnRandomArchitectureConfig,
    ] = field(default_factory=dict)
    taper_architectures: Mapping[
        ArchitectureName,
        FnnTaperArchitectureConfig,
    ] = field(default_factory=dict)

    def __post_init__(self) -> None:
        '''
        Normalize the configuration.

        All normalized ranges remain half-open: the lower bound is inclusive,
        and the upper bound is exclusive. The reverse-taper rate range instead
        excludes its lower bound and includes its upper bound.

        Defaults are provided when the user selects ``None`` for any attribute
        in ``FnnRandomArchitectureConfig`` or ``FnnTaperArchitectureConfig``.
        An empty architecture mapping remains empty.

        Both fields must be mappings whose keys are supported
        ``ArchitectureName`` members and whose values use the corresponding
        configuration type. When WIDE and NARROW are both present, their layer
        and size ranges are validated for the required ordering and separation.
        The normalized mappings are copied into read-only proxies.
        '''
        self._validate_architecture_mapping(
            'range_architectures',
            self.range_architectures,
            frozenset((ArchitectureName.WIDE, ArchitectureName.NARROW)),
            FnnRandomArchitectureConfig,
        )
        self._validate_architecture_mapping(
            'taper_architectures',
            self.taper_architectures,
            frozenset((
                ArchitectureName.TAPER,
                ArchitectureName.REVERSE_TAPER,
                ArchitectureName.COMBINED_TAPER,
            )),
            FnnTaperArchitectureConfig,
        )

        normalized_range_config = {}
        for name, architecture in self.range_architectures.items():
            layer_range = architecture.layer_range or DEFAULT_RANDOM_CONFIG[name].layer_range
            size_range = architecture.size_range or DEFAULT_RANDOM_CONFIG[name].size_range
            normalized_range_config[name] = FnnRandomArchitectureConfig(layer_range, size_range)

        normalized_taper_config = {}
        for name, architecture in self.taper_architectures.items():
            layer_range = architecture.layer_range or DEFAULT_TAPER_CONFIG[name].layer_range
            start_size_range = architecture.start_size_range or DEFAULT_TAPER_CONFIG[name].start_size_range
            taper_rate_range = architecture.taper_rate_range or DEFAULT_TAPER_CONFIG[name].taper_rate_range
            max_size = architecture.max_size or DEFAULT_TAPER_CONFIG[name].max_size
            normalized_taper_config[name] = FnnTaperArchitectureConfig(layer_range, start_size_range, taper_rate_range, max_size)

        self._validate_random_architecture_relationships(
            normalized_range_config
        )

        object.__setattr__(
            self,
            'range_architectures',
            MappingProxyType(normalized_range_config),
        )
        object.__setattr__(
            self,
            'taper_architectures',
            MappingProxyType(normalized_taper_config),
        )

    @staticmethod
    def _validate_architecture_mapping(
        name: str,
        value: Mapping,
        supported_names: frozenset[ArchitectureName],
        config_type: type,
    ) -> None:
        '''
        Validate a configuration mapping's type, keys, names, and value types.
        '''
        if not isinstance(value, Mapping):
            raise TypeError(f'{name} must be a mapping.')

        for architecture_name, config in value.items():
            if not isinstance(architecture_name, ArchitectureName):
                raise TypeError(
                    f'{name} keys must be ArchitectureName members.'
                )

            if architecture_name not in supported_names:
                raise ValueError(
                    f'{architecture_name.value!r} is not supported in {name}.'
                )

            if not isinstance(config, config_type):
                raise TypeError(
                    f'{name}[{architecture_name.value!r}] must be a '
                    f'{config_type.__name__}.'
                )

    @staticmethod
    def _validate_random_architecture_relationships(
        configurations: Mapping[
            ArchitectureName,
            FnnRandomArchitectureConfig,
        ],
    ) -> None:
        '''
        Validate the ordering and separation of WIDE and NARROW ranges.

        WIDE must have the lower layer range, and NARROW must have the lower
        neuron-size range. Adjacent half-open ranges are considered disjoint.
        '''
        if not {
            ArchitectureName.WIDE,
            ArchitectureName.NARROW,
        }.issubset(configurations):
            return

        wide = configurations[ArchitectureName.WIDE]
        narrow = configurations[ArchitectureName.NARROW]

        if wide.layer_range[1] > narrow.layer_range[0]:
            raise ValueError(
                'WIDE and NARROW layer_range values must be disjoint, with '
                'WIDE below NARROW.'
            )

        if narrow.size_range[1] > wide.size_range[0]:
            raise ValueError(
                'WIDE and NARROW size_range values must be disjoint, with '
                'NARROW below WIDE.'
            )

    @property
    def variation_labels(self) -> tuple[str]:
        labels = [
            architecture.value
            for architecture
            in self.range_architectures
        ]
        labels.extend([
            architecture.value
            for architecture
            in self.taper_architectures
        ])

        return tuple(labels)


@dataclass(frozen=True, slots=True)
class FnnArchitectureVariation(Variation[FnnArchitecture]):
    '''
    Represents a generated variation returned by ``FnnArchitectureGenerator``.

    ``label`` must be a nonempty string, ``random_state`` must be a non-boolean
    integer or ``None``, and ``generated`` must be an ``FnnArchitecture``.
    
    Attributes
    ------------
        architecture: FnnArchitecture
            The generated architecture.
    '''
    def __post_init__(self) -> None:
        '''
        Validate the inherited label, random state, and generated architecture.
        '''
        if not isinstance(self.label, str):
            raise TypeError('label must be a string.')

        if not self.label:
            raise ValueError('label must not be empty.')

        if (
            not isinstance(self.random_state, int)
            or isinstance(self.random_state, bool)
        ) and self.random_state is not None:
            raise TypeError('random_state must be an integer or None.')

        if not isinstance(self.generated, FnnArchitecture):
            raise TypeError('generated must be an FnnArchitecture.')

    @property
    def architecture(self) -> FnnArchitecture:
        """The generated architecture (kept as a compatibility alias)."""
        return self.generated


class FnnArchitectureGenerator(Generator[FnnArchitecture]):
    '''
    Generate FNN architectures that describe a model's hidden-layer structure.

    All configured ranges are half-open: the lower bound is inclusive, and the
    upper bound is exclusive. The reverse-taper rate range instead excludes its
    lower bound and includes its upper bound.

    Attributes
    --------------
    settings: FnnArchitectureGeneratorConfig | None, default = None
        The generator's settings for configuring architecture ranges, rates, and maximums.

    Examples
    --------
    Generate every architecture using the predefined configurations:

    >>> settings = FnnArchitectureGeneratorConfig(
    ...     range_architectures=DEFAULT_RANDOM_CONFIG,
    ...     taper_architectures=DEFAULT_TAPER_CONFIG,
    ... )
    >>> generator = FnnArchitectureGenerator(settings)
    >>> variations = generator.generate(random_state=42)

    Generate only a customized wide architecture:

    >>> settings = FnnArchitectureGeneratorConfig(
    ...     range_architectures={
    ...         ArchitectureName.WIDE: FnnRandomArchitectureConfig(
    ...             layer_range=(2, 5),
    ...             size_range=(64, 129),
    ...         ),
    ...     },
    ... )
    >>> generator = FnnArchitectureGenerator(settings)
    >>> variation = generator.generate(random_state=42)[0]
    >>> variation.label
    'wide'
    '''
    def __init__(
        self,
        settings: FnnArchitectureGeneratorConfig | None = None,
    ) -> None:
        self.settings = settings or FnnArchitectureGeneratorConfig(
            DEFAULT_RANDOM_CONFIG,
            DEFAULT_TAPER_CONFIG
        )

    def _generate_random_sizes(
        self,
        config: FnnRandomArchitectureConfig,
        rng: np.random.Generator,
    ) -> FnnArchitecture:
        '''
        Generate hidden-layer sizes from ranges for the numbers of layers and
        neurons.

        Both ``layer_range`` and ``size_range`` are half-open: the lower bound
        is inclusive, and the upper bound is exclusive.

        Parameters
        -----------
        config: FnnRandomArchitectureConfig
            The ranges for a wide or narrow architecture.
        rng: np.random.Generator
            The random number generator used for reproducibility.
        
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
        Generate hidden-layer sizes from ranges for the number of layers, the
        number of neurons in the first layer, and the taper rate, subject to a
        maximum number of neurons per layer.

        ``layer_range``, ``start_size_range``, and ``taper_rate_range`` are
        half-open: the lower bound is inclusive, and the upper bound is
        exclusive. For a reverse taper, ``taper_rate_range`` instead excludes
        its lower bound and includes its upper bound: ``(lower, upper]``.
        ``max_size`` is an inclusive limit.

        The method uses ``taper_type`` to build the appropriate taper
        architecture. The ``rng`` object is used to select the number of layers
        (``n_layers``), starting layer size (``start_size``), and rate of size
        increase or decrease (``size_rate``).

        For ``TAPER``, a multiplier ``q`` is sampled from
        ``[1 + lower_rate, 1 + upper_rate)``. Before rounding and clamping, the
        size of zero-indexed layer ``i`` is ``start_size * q**i``. Because
        ``q > 1``, layer sizes grow exponentially. Each result is rounded to the
        nearest integer and limited to ``[1, max_size]``.

        For ``REVERSE_TAPER``, a positive multiplier ``q`` is sampled from
        ``[1 - upper_rate, 1 - lower_rate)``. Equivalently, the effective decay
        rate ``1 - q`` lies in ``(lower_rate, upper_rate]``. Before rounding and
        clamping, the size of zero-indexed layer ``i`` is
        ``start_size * q**i``. Because ``0 < q < 1``, layer sizes decay
        exponentially and remain positive. Each result is rounded to the nearest
        integer and limited to ``[1, max_size]``.

        For ``COMBINED_TAPER``, a rate ``r`` is sampled from
        ``[lower_rate, upper_rate)``. The first half grows exponentially as
        ``start_size * (1 + r)**i``. After the midpoint, sizes decay from the
        peak as ``peak * (1 - r)**(i - midpoint + 1)``. Results are rounded and
        limited to ``[1, max_size]``.
        
        Parameters
        -----------
        config: FnnTaperArchitectureConfig
            The ranges and maximum for a taper, reverse-taper, or combined-taper
            architecture.
        taper_type: ArchitectureName
            The architecture type, such as ``'taper'``, ``'reverse_taper'``, or
            ``'combined_taper'``.
        rng: np.random.Generator
            The random number generator used for reproducibility.
        
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
        match taper_type:
            case ArchitectureName.TAPER:
                size_rate = rng.uniform(low_taper_rate + 1, high_taper_rate + 1)

            case ArchitectureName.REVERSE_TAPER:
                size_rate = rng.uniform(1 - high_taper_rate, 1 - low_taper_rate)

            case ArchitectureName.COMBINED_TAPER:
                size_rate = rng.uniform(low_taper_rate, high_taper_rate)

            case _:
                raise ValueError(
                    f'Unknown taper_type: {taper_type!r}.'
                )

        sizes = []

        if taper_type in (
            ArchitectureName.TAPER,
            ArchitectureName.REVERSE_TAPER,
        ):
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
        Generate FNN architectures.

        This method iterates through two groups of architecture types: random and
        taper. Each group generates an architecture with either a random number
        of layers and neurons per layer, or a random number of layers, a starting
        layer size, and a rate of size increase or decrease, subject to a maximum
        layer size.

        All ranges used during generation are half-open: the lower bound is
        inclusive, and the upper bound is exclusive. The reverse-taper rate
        range instead excludes its lower bound and includes its upper bound:
        ``(lower, upper]``.

        Parameters
        ------------
        random_state: int | None, default = None
            An integer used to set the random seed, if provided.
        
        Returns
        ----------
        list[Variation[FnnArchitecture]]
            A list of FNN architectures variations.
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
