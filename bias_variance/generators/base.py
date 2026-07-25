from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Variation(ABC):
    '''
    Represents the base variation returned by any Generator.

    Attributes:
        label: The identifier of the generated variation.
        random_state: The random seed used if provided.
    '''
    label: str
    random_state: int | None


class Generator(ABC):
    '''Produces labeled variations for one study iteration.'''

    @abstractmethod
    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> list[Variation]:
        '''Generates one variation per configured label.'''
        raise NotImplementedError
