import pandas as pd
import pytest

from bias_variance.generators.sampling import (
    SamplingGenerator,
    SamplingStrategy,
)
from common.sampling._sampling import get_random_samples


@pytest.fixture
def dataset() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": range(20),
            "target": range(100, 120),
        }
    )


def test_generate_returns_labeled_reproducible_samples(dataset):
    generator = SamplingGenerator(
        dataset,
        [
            SamplingStrategy(
                label="bootstrap",
                function=get_random_samples,
                kwargs={"n_samples": 20, "with_replacement": True},
            ),
            SamplingStrategy(
                label="subset",
                function=get_random_samples,
                kwargs={"n_samples": 8},
            ),
        ],
    )

    first = generator.generate(random_state=42)
    second = generator.generate(random_state=42)

    assert set(first) == {"bootstrap", "subset"}
    assert len(first["bootstrap"]) == 20
    assert len(first["subset"]) == 8
    pd.testing.assert_frame_equal(first["bootstrap"], second["bootstrap"])
    pd.testing.assert_frame_equal(first["subset"], second["subset"])


def test_constructor_copies_source_dataset(dataset):
    generator = SamplingGenerator(
        dataset,
        [
            SamplingStrategy(
                "all",
                get_random_samples,
                {"sample_fraction": 1.0},
            )
        ],
    )
    original = dataset.copy()
    dataset.loc[:, "feature"] = -1

    generated = generator.generate(random_state=1)["all"].sort_index()

    pd.testing.assert_frame_equal(generated, original)


def test_each_strategy_receives_an_independent_dataset_copy(dataset):
    def mutate(frame: pd.DataFrame, *, random_state=None) -> pd.DataFrame:
        frame.loc[:, "feature"] = -1
        return frame

    def observe(frame: pd.DataFrame, *, random_state=None) -> pd.DataFrame:
        return frame

    generator = SamplingGenerator(
        dataset,
        [
            SamplingStrategy("mutate", mutate),
            SamplingStrategy("observe", observe),
        ],
    )

    generated = generator.generate(random_state=42)

    assert (generated["mutate"]["feature"] == -1).all()
    pd.testing.assert_frame_equal(generated["observe"], dataset)


def test_duplicate_strategy_label_is_rejected(dataset):
    strategy = SamplingStrategy(
        "sample",
        get_random_samples,
        {"n_samples": 4},
    )
    generator = SamplingGenerator(dataset, [strategy])

    with pytest.raises(ValueError, match="Duplicate sampling strategy label"):
        generator.add_strategy(strategy)


def test_strategy_random_state_is_rejected(dataset):
    strategy = SamplingStrategy(
        "sample",
        get_random_samples,
        {"n_samples": 4, "random_state": 42},
    )

    with pytest.raises(ValueError, match="Configure random_state on BiasAnalyzer"):
        SamplingGenerator(dataset, [strategy])


def test_remove_strategy_is_chainable_and_idempotent(dataset):
    generator = SamplingGenerator(
        dataset,
        [
            SamplingStrategy(
                "sample",
                get_random_samples,
                {"n_samples": 4},
            )
        ],
    )

    assert generator.remove_strategy("sample") is generator
    assert generator.remove_strategy("missing") is generator
    assert generator.generate(random_state=42) == {}
