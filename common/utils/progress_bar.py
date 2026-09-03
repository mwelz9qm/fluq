"""Console progress reporting for multi-study model runs."""

import sys
from typing import TextIO


class ProgressBar:
    """Display study and model progress on a single console line.

    A study must be started with :meth:`start_study` before models can be
    recorded with :meth:`advance`. The model count is reset for every study.
    """

    def __init__(
        self,
        total_studies: int,
        *,
        width: int = 30,
        stream: TextIO | None = None,
    ) -> None:
        if (
            not isinstance(total_studies, int)
            or isinstance(total_studies, bool)
        ):
            raise TypeError("total_studies must be an integer.")
        if total_studies <= 0:
            raise ValueError("total_studies must be greater than zero.")
        if not isinstance(width, int) or isinstance(width, bool):
            raise TypeError("width must be an integer.")
        if width <= 0:
            raise ValueError("width must be greater than zero.")

        self.total_studies = total_studies
        self.width = width
        self.stream = sys.stdout if stream is None else stream
        self.current_study = 0
        self.total_models = 0
        self.completed_models = 0

    def start_study(self, study_number: int, total_models: int) -> None:
        """Start reporting a study and reset its completed-model count."""
        if not isinstance(study_number, int) or isinstance(study_number, bool):
            raise TypeError("study_number must be an integer.")
        if not 1 <= study_number <= self.total_studies:
            raise ValueError(
                "study_number must be between one and total_studies."
            )
        if not isinstance(total_models, int) or isinstance(total_models, bool):
            raise TypeError("total_models must be an integer.")
        if total_models <= 0:
            raise ValueError("total_models must be greater than zero.")

        self.current_study = study_number
        self.total_models = total_models
        self.completed_models = 0
        self._render()

    def advance(self) -> None:
        """Record one fully trained, tested, and evaluated model."""
        if self.current_study == 0:
            raise RuntimeError("Call start_study() before advance().")
        if self.completed_models >= self.total_models:
            raise RuntimeError("All models in the current study are complete.")

        self.completed_models += 1
        self._render()

    def _render(self) -> None:
        fraction = self.completed_models / self.total_models
        filled = int(self.width * fraction)
        bar = "#" * filled + "-" * (self.width - filled)
        ending = "\n" if self.completed_models == self.total_models else "\r"
        message = (
            f"Study {self.current_study}/{self.total_studies} | "
            f"Models trained/tested/evaluated [{bar}] "
            f"{self.completed_models}/{self.total_models} "
            f"({fraction:6.1%})"
        )
        print(message, end=ending, file=self.stream, flush=True)
