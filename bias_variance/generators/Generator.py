from abc import ABC,  abstractmethod
from collections.abc import Mapping
from typing import Generic, TypeVar

GeneratedT = TypeVar('GeneratedT')


class Generator(ABC, Generic[GeneratedT]):
    '''Produces labeled variations for one study iteration.'''

    @abstractmethod
    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> Mapping[str, GeneratedT]:
        '''Generates one variation per configured label.'''
        raise NotImplementedError
