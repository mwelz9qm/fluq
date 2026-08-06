from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GeneratorConfig(ABC):
    '''Configures generator settings.'''

    @abstractmethod
    @property
    def variation_labels(self) -> tuple[str]:
        '''Returns all variation labels.'''
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Variation[Generated]:
    '''
    Represents the base variation returned by any Generator.

    Attributes:
        label: The identifier of the generated variation.
        random_state: The random seed used if provided.
        generated: The value produced by the generator.
    '''
    label: str
    random_state: int | None
    generated: Generated


class Generator[Generated](ABC):
    '''Produces labeled variations for one study iteration.'''

    @abstractmethod
    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> list[Variation[Generated]]:
        '''Generates one variation per configured label.'''
        raise NotImplementedError
