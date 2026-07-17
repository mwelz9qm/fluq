from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from bias_variance.BiasAnalyzer import BiasAnalyzer
from bias_variance.generators.ArchitectureGenerator import ArchitectureGenerator
from bias_variance.generators.Generator import Generator
from bias_variance.generators.SamplingGenerator import SamplingGenerator


@pytest.fixture
def test_input_and_output_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmarks = Path(__file__).resolve().parents[1] / "benchmarks"
    df = pd.concat(
        [
            pd.read_csv(benchmarks / "chf_train_synth.csv"),
            pd.read_csv(benchmarks / "chf_test_synth.csv"),
        ]
    )
    inputs_df = df.iloc[:, :6]
    outputs_df = df.iloc[:, 6:]
    return inputs_df, outputs_df


@pytest.fixture
def analyzer(test_input_and_output_dfs) -> BiasAnalyzer:
    inputs_df, outputs_df = test_input_and_output_dfs
    return BiasAnalyzer(inputs_df, outputs_df, random_state=40)


@pytest.fixture
def small_regression_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small deterministic dataset for framework-level behavioral tests."""
    x = np.linspace(-1.0, 1.0, 30, dtype=np.float32)
    inputs = pd.DataFrame(
        {
            "x1": x,
            "x2": x**2,
            "x3": np.sin(x),
        }
    )
    outputs = pd.DataFrame({"y": 1.5 * x - 0.25})
    return inputs, outputs


def short_model_settings() -> dict[str, object]:
    return {
        "hidden_layers": [6, 4],
        "activation": "relu",
        "optimizer": "adam",
        "loss": "mse",
        "metrics": ["rmse", "r2", "mse", "mae"],
        "epochs": 2,
        "batch_size": 5,
        "verbose": 0,
    }


class RecordingGenerator(Generator[object]):
    def __init__(self, variations: dict[str, object]) -> None:
        self.variations = variations
        self.random_states: list[int | None] = []

    def generate(
        self,
        *,
        random_state: int | None = None,
    ) -> dict[str, object]:
        self.random_states.append(random_state)
        return self.variations


def test_constructor(test_input_and_output_dfs):
    inputs_df, outputs_df = test_input_and_output_dfs
    analyzer = BiasAnalyzer(inputs_df, outputs_df)

    pd.testing.assert_frame_equal(analyzer.inputs_df, inputs_df)
    pd.testing.assert_frame_equal(analyzer.outputs_df, outputs_df)
    assert analyzer.model_settings == {
        "hidden_layers": [32, 32],
        "activation": "relu",
        "optimizer": "adam",
        "loss": "mse",
        "metrics": ["rmse", "r2", "mse", "mae"],
        "epochs": 100,
        "batch_size": 10,
        "verbose": 0,
    }
    assert analyzer.test_size == 0.2
    assert analyzer.random_state is None
    assert analyzer._model is None
    assert analyzer._init_weights is None
    assert analyzer._results_df is None


def test_build_model_preserves_dense_regression_architecture(
    small_regression_data,
):
    inputs, outputs = small_regression_data
    analyzer = BiasAnalyzer(
        inputs,
        outputs,
        model_settings=short_model_settings(),
    )

    model = analyzer._build_model([6, 4])
    dense_layers = [layer for layer in model.layers if hasattr(layer, "units")]

    assert model.input_shape == (None, 3)
    assert model.output_shape == (None, 1)
    assert [layer.units for layer in dense_layers] == [6, 4, 1]
    assert [layer.activation.__name__ for layer in dense_layers] == [
        "relu",
        "relu",
        "linear",
    ]


def test_short_training_preserves_prediction_and_metric_contract(
    small_regression_data,
):
    inputs, outputs = small_regression_data
    analyzer = BiasAnalyzer(
        inputs,
        outputs,
        model_settings=short_model_settings(),
        test_size=0.2,
        random_state=17,
    )

    scores, predictions, actuals = analyzer._get_test_result_and_data()

    assert set(scores) == {
        "loss",
        "rmse",
        "r2",
        "mse",
        "mae",
        "variance",
        "mean",
        "conf_interval_lower",
        "conf_interval_upper",
    }
    assert predictions.shape == actuals.shape == (6, 1)
    assert all(np.isscalar(value) for value in scores.values())
    assert all(np.isfinite(value) for value in scores.values())


def test_get_test_result_restores_initial_weights_before_fitting(
    small_regression_data,
    monkeypatch,
):
    inputs, outputs = small_regression_data
    settings = short_model_settings()
    settings["epochs"] = 1
    analyzer = BiasAnalyzer(inputs, outputs, model_settings=settings, random_state=5)
    analyzer._init_model()
    initial_weights = [weight.copy() for weight in analyzer._init_weights]
    analyzer._model.set_weights(
        [np.full_like(weight, 42.0) for weight in analyzer._model.get_weights()]
    )
    weights_at_fit = []

    def record_fit(*args, **kwargs):
        weights_at_fit.extend(
            weight.copy() for weight in analyzer._model.get_weights()
        )

    monkeypatch.setattr(analyzer._model, "fit", record_fit)
    monkeypatch.setattr(
        analyzer._model,
        "predict",
        lambda X: np.arange(len(X), dtype=np.float32).reshape(-1, 1),
    )
    monkeypatch.setattr(
        analyzer._model,
        "evaluate",
        lambda *args, **kwargs: {"loss": 0.0},
    )

    analyzer._get_test_result_and_data()

    assert len(weights_at_fit) == len(initial_weights)
    for restored, initial in zip(weights_at_fit, initial_weights):
        np.testing.assert_array_equal(restored, initial)


def test_random_state_preserves_test_split_across_analyzers(
    small_regression_data,
):
    inputs, outputs = small_regression_data
    settings = short_model_settings()
    settings["epochs"] = 0
    first = BiasAnalyzer(inputs, outputs, model_settings=settings, random_state=23)
    second = BiasAnalyzer(inputs, outputs, model_settings=settings, random_state=23)

    _, _, first_actuals = first._get_test_result_and_data()
    _, _, second_actuals = second._get_test_result_and_data()

    pd.testing.assert_frame_equal(first_actuals, second_actuals)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "The shared Keras R2Score is initialized for one output and cannot be "
        "reused by a two-output model."
    ),
)
def test_short_training_supports_multi_output_shape(small_regression_data):
    inputs, single_output = small_regression_data
    outputs = single_output.assign(y_squared=single_output["y"] ** 2)
    analyzer = BiasAnalyzer(
        inputs,
        outputs,
        model_settings=short_model_settings(),
        test_size=0.2,
        random_state=11,
    )

    scores, predictions, actuals = analyzer._get_test_result_and_data()

    assert analyzer._model.output_shape == (None, 2)
    assert predictions.shape == actuals.shape == (6, 2)
    assert set(("loss", "rmse", "r2", "mse", "mae")).issubset(scores)


def test_build_generator_returns_configured_architecture_generator(analyzer):
    settings = {
        "wide": {
            "layers": (2, 3),
            "neurons": (8, 16),
        }
    }

    generator = analyzer._build_generator("model", settings)

    assert isinstance(generator, ArchitectureGenerator)
    assert generator.settings == settings
    assert set(generator.generate(random_state=42)) == {"wide"}


@pytest.mark.parametrize(
    "strategies",
    [
        ["bootstrap"],
        ["stratified"],
        ["lhs"],
        ["bootstrap", "stratified", "lhs"],
    ],
)
def test_build_generator_returns_configured_sampling_generator(
    analyzer,
    strategies,
):
    generator = analyzer._build_generator(
        "sampling",
        {"strategies": strategies},
    )

    assert isinstance(generator, SamplingGenerator)
    assert set(generator._strategies) == set(strategies)
    generated = generator.generate(random_state=42)
    assert set(generated) == set(strategies)
    assert all(len(dataset) == len(analyzer.inputs_df) for dataset in generated.values())


def test_build_generator_rejects_unsupported_study(analyzer):
    with pytest.raises(ValueError, match="Unsupported study"):
        analyzer._build_generator("unknown", {})


def test_get_results_generates_model_variations_for_each_seed(
    analyzer,
    monkeypatch,
):
    generator = RecordingGenerator(
        {
            "wide": (16, 12),
            "narrow": (6, 4, 2),
        }
    )
    analyzer._run_id = "run_test"
    training_calls = []
    saved_calls = []

    def fake_train(*, split=None, hidden_layers=None):
        training_calls.append((split, hidden_layers))
        return {"loss": float(len(hidden_layers))}, np.array([[1.0]]), pd.DataFrame({"y": [1.0]})

    def fake_save(predictions, actuals, *, study, label, iteration):
        saved_calls.append((study, label, iteration))

    monkeypatch.setattr(analyzer, "_get_test_result_and_data", fake_train)
    monkeypatch.setattr(analyzer, "_save_predictions_and_actuals", fake_save)

    results = analyzer._get_results(2, generator, "model")

    assert generator.random_states == [40, 41]
    assert training_calls == [
        (None, [16, 12]),
        (None, [6, 4, 2]),
        (None, [16, 12]),
        (None, [6, 4, 2]),
    ]
    assert saved_calls == [
        ("model", "wide", 0),
        ("model", "narrow", 0),
        ("model", "wide", 1),
        ("model", "narrow", 1),
    ]
    assert results["run_id"].tolist() == ["run_test"] * 4
    assert results["iteration"].tolist() == [0, 0, 1, 1]
    assert results["study"].tolist() == ["model"] * 4
    assert results["variable"].tolist() == ["wide", "narrow"] * 2
    assert results["loss"].tolist() == [2.0, 3.0, 2.0, 3.0]


def test_get_results_splits_sampling_variation_and_can_skip_saving(
    analyzer,
    monkeypatch,
):
    sampled_df = pd.concat(
        [analyzer.inputs_df, analyzer.outputs_df],
        axis=1,
    )
    generator = RecordingGenerator({"bootstrap": sampled_df})
    received = []

    def fake_train(*, split=None, hidden_layers=None):
        received.append((split, hidden_layers))
        return {"loss": 1.5}, np.array([[1.0]]), split[3]

    def fail_if_saved(*args, **kwargs):
        pytest.fail("Predictions should not be saved")

    monkeypatch.setattr(analyzer, "_get_test_result_and_data", fake_train)
    monkeypatch.setattr(analyzer, "_save_predictions_and_actuals", fail_if_saved)

    results = analyzer._get_results(
        1,
        generator,
        "sampling",
        save_predictions=False,
    )

    assert generator.random_states == [40]
    split, hidden_layers = received[0]
    X_train, X_test, y_train, y_test = split
    assert hidden_layers is None
    assert list(X_train.columns) == list(analyzer.inputs_df.columns)
    assert list(X_test.columns) == list(analyzer.inputs_df.columns)
    assert list(y_train.columns) == list(analyzer.outputs_df.columns)
    assert list(y_test.columns) == list(analyzer.outputs_df.columns)
    assert len(X_train) + len(X_test) == len(sampled_df)
    assert results.loc[0, "study"] == "sampling"
    assert results.loc[0, "variable"] == "bootstrap"


def test_get_results_passes_none_seed_when_random_state_is_unset(
    test_input_and_output_dfs,
    monkeypatch,
):
    inputs_df, outputs_df = test_input_and_output_dfs
    analyzer = BiasAnalyzer(inputs_df, outputs_df)
    generator = RecordingGenerator({"wide": (8,)})

    monkeypatch.setattr(
        analyzer,
        "_get_test_result_and_data",
        lambda **kwargs: ({"loss": 1.0}, np.array([[1.0]]), outputs_df.iloc[:1]),
    )

    analyzer._get_results(2, generator, "model", save_predictions=False)

    assert generator.random_states == [None, None]


def test_save_predictions_and_actuals_requires_run_id(analyzer):
    with pytest.raises(ValueError, match="_run_id is None"):
        analyzer._save_predictions_and_actuals(
            np.array([[1.0]]),
            pd.DataFrame({"y": [1.0]}),
            study="model",
            label="wide",
            iteration=0,
        )


def test_save_predictions_and_actuals_writes_expected_hdf5_group(
    analyzer,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    analyzer._run_id = "run_test"
    predictions = np.array([[1.0], [2.0]])
    actuals = pd.DataFrame({"y": [1.5, 2.5]})

    analyzer._save_predictions_and_actuals(
        predictions,
        actuals,
        study="sampling",
        label="bootstrap",
        iteration=3,
    )

    with h5py.File(tmp_path / "iterations" / "run_test.h5", "r") as hf:
        group = hf["sampling/bootstrap/iteration_3"]
        np.testing.assert_array_equal(group["predictions"][:], predictions)
        np.testing.assert_array_equal(group["actuals"][:], actuals.to_numpy())


@pytest.mark.parametrize("n_iter", [1.5, "2", True, None])
def test_run_bias_studies_rejects_non_integer_iterations(analyzer, n_iter):
    with pytest.raises(TypeError, match="n_iter must be an integer"):
        analyzer.run_bias_studies(
            {"n_iter": n_iter, "studies": {"model": {}}},
            save_results=False,
            save_predictions=False,
        )


@pytest.mark.parametrize("n_iter", [0, -1])
def test_run_bias_studies_rejects_non_positive_iterations(analyzer, n_iter):
    with pytest.raises(ValueError, match="n_iter must be greater than 0"):
        analyzer.run_bias_studies(
            {"n_iter": n_iter, "studies": {"model": {}}},
            save_results=False,
            save_predictions=False,
        )


def test_run_bias_studies_requires_at_least_one_study(analyzer):
    with pytest.raises(ValueError, match="At least one study"):
        analyzer.run_bias_studies(
            {"n_iter": 1, "studies": {}},
            save_results=False,
            save_predictions=False,
        )


def test_run_bias_studies_rejects_unknown_study(analyzer):
    with pytest.raises(ValueError, match="Unsupported studies"):
        analyzer.run_bias_studies(
            {"n_iter": 1, "studies": {"unknown": {}}},
            save_results=False,
            save_predictions=False,
        )


def test_run_bias_studies_rejects_unknown_sampling_strategy(analyzer):
    with pytest.raises(ValueError, match="Unsupported sampling strategies"):
        analyzer.run_bias_studies(
            {
                "n_iter": 1,
                "studies": {"sampling": {"strategies": ["unknown"]}},
            },
            save_results=False,
            save_predictions=False,
        )


def test_run_bias_studies_orchestrates_generators_and_accumulates_results(
    analyzer,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    built = []
    result_calls = []

    def fake_build(study, settings):
        generator = RecordingGenerator({})
        built.append((study, settings, generator))
        return generator

    def fake_results(n_iter, generator, study, *, save_predictions=True):
        result_calls.append((n_iter, generator, study, save_predictions))
        return pd.DataFrame(
            {
                "run_id": [analyzer._run_id],
                "iteration": [0],
                "study": [study],
                "variable": [f"{study}_variation"],
                "loss": [1.0],
            }
        )

    monkeypatch.setattr(analyzer, "_init_model", lambda: None)
    monkeypatch.setattr(analyzer, "_build_generator", fake_build)
    monkeypatch.setattr(analyzer, "_get_results", fake_results)

    settings = {
        "n_iter": 2,
        "studies": {
            "model": {"wide": None},
            "sampling": {"strategies": ["bootstrap"]},
        },
    }

    returned = analyzer.run_bias_studies(
        settings,
        save_results=False,
        save_predictions=False,
    )

    assert returned is analyzer
    assert analyzer._run_id.startswith("run_")
    assert [item[:2] for item in built] == [
        ("model", {"wide": None}),
        ("sampling", {"strategies": ["bootstrap"]}),
    ]
    assert [(n, study, save) for n, _, study, save in result_calls] == [
        (2, "model", False),
        (2, "sampling", False),
    ]
    assert analyzer._results_df["study"].tolist() == ["model", "sampling"]
    assert not (tmp_path / BiasAnalyzer.RESULTS_FILENAME).exists()


def test_run_bias_studies_saves_accumulated_results(
    analyzer,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(analyzer, "_init_model", lambda: None)
    monkeypatch.setattr(
        analyzer,
        "_build_generator",
        lambda study, settings: RecordingGenerator({}),
    )
    monkeypatch.setattr(
        analyzer,
        "_get_results",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "run_id": [analyzer._run_id],
                "iteration": [0],
                "study": ["model"],
                "variable": ["wide"],
                "loss": [1.0],
            }
        ),
    )

    analyzer.run_bias_studies(
        {"n_iter": 1, "studies": {"model": {"wide": None}}},
        save_results=True,
        save_predictions=False,
    )

    saved = pd.read_csv(tmp_path / BiasAnalyzer.RESULTS_FILENAME)
    assert saved["study"].tolist() == ["model"]
    assert saved["variable"].tolist() == ["wide"]
