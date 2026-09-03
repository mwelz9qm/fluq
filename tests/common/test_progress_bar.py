from io import StringIO

import pytest

from common.utils import ProgressBar


def test_progress_bar_reports_study_and_model_progress() -> None:
    output = StringIO()
    progress = ProgressBar(total_studies=2, width=10, stream=output)

    progress.start_study(study_number=1, total_models=2)
    progress.advance()
    progress.advance()
    progress.start_study(study_number=2, total_models=1)
    progress.advance()

    rendered = output.getvalue()
    assert "Study 1/2 | Models trained/tested/evaluated [----------] 0/2" in rendered
    assert "Study 1/2 | Models trained/tested/evaluated [#####-----] 1/2" in rendered
    assert "Study 1/2 | Models trained/tested/evaluated [##########] 2/2" in rendered
    assert "Study 2/2 | Models trained/tested/evaluated [##########] 1/1" in rendered
    assert rendered.endswith("(100.0%)\n")


def test_progress_bar_requires_a_started_incomplete_study() -> None:
    progress = ProgressBar(total_studies=1, stream=StringIO())

    with pytest.raises(RuntimeError, match="start_study"):
        progress.advance()

    progress.start_study(study_number=1, total_models=1)
    progress.advance()

    with pytest.raises(RuntimeError, match="complete"):
        progress.advance()
