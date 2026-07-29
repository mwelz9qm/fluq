import json

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split

from bias_variance.BiasAnalyzer import BiasAnalyzer
from bias_variance.generators.fnn_architecture import (
    FnnArchitectureGenerator,
)
from bias_variance.generators.base import Generator
from bias_variance.generators.sampling import SamplingGenerator
from bias_variance.models.TrainingConfig import TrainingConfig
from bias_variance.models.fnn import FnnArchitecture, FnnBuilder, FnnConfig
from bias_variance.generators.noise import NoiseGenerator, NoiseVariation


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.DataFrame]:
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


@pytest.fixture
def analyzer(regression_data) -> BiasAnalyzer:
    inputs, outputs = regression_data
    return BiasAnalyzer(
        inputs,
        outputs,
        fnn_builder=FnnBuilder(
            FnnConfig(
                input_size=inputs.shape[1],
                output_size=outputs.shape[1],
            )
        ),
        baseline_architecture=FnnArchitecture((6, 4)),
        training_config=TrainingConfig(
            epochs=1,
            batch_size=5,
            device="cpu",
        ),
    )

@pytest.fixture
def base_training_dataset(
    analyzer,
) -> pd.DataFrame:
    X_train, _, y_train, _ = train_test_split(
        analyzer.inputs_df,
        analyzer.outputs_df,
        test_size=0.2,
        random_state=42,
    )

    return pd.concat(
        [X_train, y_train],
        axis=1,
    )

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


def empty_results() -> pd.DataFrame:
    return pd.DataFrame(columns=BiasAnalyzer.RESULT_COLUMNS)


def empty_runs_metadata() -> pd.DataFrame:
    return pd.DataFrame(columns=BiasAnalyzer.RUN_METADATA_COLUMNS)


def test_constructor_stores_pytorch_configuration(analyzer, regression_data):
    inputs, outputs = regression_data

    pd.testing.assert_frame_equal(analyzer.inputs_df, inputs)
    pd.testing.assert_frame_equal(analyzer.outputs_df, outputs)
    assert analyzer.fnn_builder.config.input_size == 3
    assert analyzer.fnn_builder.config.output_size == 1
    assert analyzer.baseline_architecture.hidden_layers == (6, 4)
    assert analyzer.training_config.device == "cpu"
    assert analyzer._results_df is None
    assert analyzer._runs_metadata_df is None
    assert analyzer._run_id is None
    assert not hasattr(analyzer, "random_state")
    assert not hasattr(analyzer, "test_size")


@pytest.mark.parametrize(
    ("mutation", "error", "message"),
    [
        (lambda X, y: (X.iloc[:0], y.iloc[:0]), ValueError, "must not be empty"),
        (
            lambda X, y: (X, y.iloc[:-1]),
            ValueError,
            "same number of rows",
        ),
        (
            lambda X, y: (X, y.rename(columns={"y": "x1"})),
            ValueError,
            "distinct column names",
        ),
        (
            lambda X, y: (X.assign(text="bad"), y),
            TypeError,
            "only numeric columns",
        ),
        (
            lambda X, y: (X.assign(x1=np.nan), y),
            ValueError,
            "only finite values",
        ),
    ],
)
def test_constructor_rejects_invalid_dataframes(
    regression_data,
    mutation,
    error,
    message,
):
    inputs, outputs = mutation(*regression_data)
    builder = FnnBuilder(FnnConfig(inputs.shape[1], outputs.shape[1]))

    with pytest.raises(error, match=message):
        BiasAnalyzer(
            inputs,
            outputs,
            fnn_builder=builder,
            baseline_architecture=FnnArchitecture((4,)),
            training_config=TrainingConfig(device="cpu"),
        )


def test_constructor_rejects_builder_dimension_mismatch(regression_data):
    inputs, outputs = regression_data
    builder = FnnBuilder(FnnConfig(input_size=2, output_size=1))

    with pytest.raises(ValueError, match="input_size"):
        BiasAnalyzer(
            inputs,
            outputs,
            fnn_builder=builder,
            baseline_architecture=FnnArchitecture((4,)),
            training_config=TrainingConfig(device="cpu"),
        )


def test_build_model_uses_generated_architecture(analyzer):
    model = analyzer._build_model(FnnArchitecture((8, 5)))
    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]

    assert isinstance(model, nn.Sequential)
    assert [
        (layer.in_features, layer.out_features)
        for layer in linear_layers
    ] == [(3, 8), (8, 5), (5, 1)]
    assert sum(isinstance(layer, nn.ReLU) for layer in model) == 2


def test_build_model_rejects_non_architecture(analyzer):
    with pytest.raises(TypeError, match="FnnArchitecture"):
        analyzer._build_model((4, 2))


def test_tensor_conversion_copies_numeric_dataframe(analyzer):
    tensor = analyzer._to_tensor(analyzer.inputs_df)
    original_value = analyzer.inputs_df.iloc[0, 0]
    tensor[0, 0] = 99.0

    assert tensor.dtype == torch.float32
    assert tensor.shape == (30, 3)
    assert analyzer.inputs_df.iloc[0, 0] == original_value


def test_loss_and_optimizer_factories(analyzer):
    model = analyzer._build_model(analyzer.baseline_architecture)

    assert isinstance(analyzer._build_loss(), nn.MSELoss)
    optimizer = analyzer._build_optimizer(model)
    assert isinstance(optimizer, optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_short_training_preserves_prediction_and_metric_contract(analyzer):
    scores, predictions, actuals = analyzer._get_test_result_and_data(
        random_state=17,
        test_size=0.2,
    )

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


def test_training_is_reproducible_for_same_run_seed(analyzer):
    first_scores, first_predictions, first_actuals = (
        analyzer._get_test_result_and_data(
            random_state=23,
            test_size=0.2,
        )
    )
    second_scores, second_predictions, second_actuals = (
        analyzer._get_test_result_and_data(
            random_state=23,
            test_size=0.2,
        )
    )

    np.testing.assert_allclose(first_predictions, second_predictions)
    pd.testing.assert_frame_equal(first_actuals, second_actuals)
    assert first_scores == pytest.approx(second_scores)


def test_multi_output_training_is_supported(regression_data):
    inputs, output = regression_data
    outputs = output.assign(y_squared=output["y"] ** 2)
    analyzer = BiasAnalyzer(
        inputs,
        outputs,
        fnn_builder=FnnBuilder(FnnConfig(3, 2)),
        baseline_architecture=FnnArchitecture((5,)),
        training_config=TrainingConfig(
            epochs=1,
            batch_size=5,
            device="cpu",
        ),
    )

    scores, predictions, actuals = analyzer._get_test_result_and_data(
        random_state=11,
        test_size=0.2,
    )

    assert predictions.shape == actuals.shape == (6, 2)
    assert "r2" in scores


@pytest.mark.parametrize(
    ("random_state", "test_size", "error", "message"),
    [
        (True, 0.2, TypeError, "random_state"),
        (1.5, 0.2, TypeError, "random_state"),
        (1, True, TypeError, "test_size"),
        (1, 0.0, ValueError, "between 0 and 1"),
        (1, 1.0, ValueError, "between 0 and 1"),
    ],
)
def test_training_rejects_invalid_run_split_settings(
    analyzer,
    random_state,
    test_size,
    error,
    message,
):
    with pytest.raises(error, match=message):
        analyzer._get_test_result_and_data(
            random_state=random_state,
            test_size=test_size,
        )


def test_calculate_scores_rejects_shape_mismatch(analyzer):
    with pytest.raises(ValueError, match="matching shapes"):
        analyzer._calculate_scores(
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((3, 2), dtype=np.float32),
        )


def test_build_generator_returns_fnn_architecture_generator(analyzer):
    settings = {
        "wide": {
            "layers": (2, 3),
            "neurons": (8, 16),
        }
    }

    generator = analyzer._build_generator("model", settings, training_dataset=analyzer.inputs_df)
    generated = generator.generate(random_state=42)

    assert isinstance(generator, FnnArchitectureGenerator)
    assert set(generated) == {"wide"}
    assert isinstance(generated["wide"], FnnArchitecture)


def test_build_generator_returns_sampling_generator(
    analyzer,
    base_training_dataset,
):
    generator = analyzer._build_generator(
        "sampling",
        {"strategies": ["bootstrap", "lhs"]},
        training_dataset=base_training_dataset,
    )

    assert isinstance(generator, SamplingGenerator)
    assert set(generator.generate(random_state=42)) == {"bootstrap", "lhs"}

def test_build_generator_returns_noise_generator(
    analyzer,
    base_training_dataset,
):
    generator = analyzer._build_generator(
        "data",
        {"standard_deviations": [0.1, 0.2]},
        training_dataset=base_training_dataset,
    )

    assert isinstance(generator, NoiseGenerator)

    generated = generator.generate(random_state=42)
    assert set(generated) == {"std_0.1", "std_0.2"}
    assert all(
        isinstance(variation, NoiseVariation)
        for variation in generated.values()
    )

def test_build_generator_rejects_invalid_study_and_strategies(
    analyzer,
    base_training_dataset,
):
    with pytest.raises(ValueError, match="Unsupported study"):
        analyzer._build_generator("unknown", {})

    with pytest.raises(ValueError, match="Unsupported sampling strategies"):
        analyzer._build_generator(
            "sampling",
            {"strategies": ["unknown"]},
        )

    with pytest.raises(ValueError, match="must not contain duplicates"):
        analyzer._build_generator(
            "sampling",
            {"strategies": ["lhs", "lhs"]},
        )


def test_get_results_injects_seeds_and_records_architecture(
    analyzer,
    monkeypatch,
):
    generator = RecordingGenerator(
        {
            "wide": FnnArchitecture((16, 12)),
            "narrow": FnnArchitecture((6, 4, 2)),
        }
    )
    analyzer._run_id = "run_test"
    calls = []

    def fake_train(*, split, architecture, random_state, test_size):
        calls.append((split, architecture, random_state, test_size))
        actuals = split[3]
        return {"loss": 1.0}, np.zeros(actuals.shape), actuals

    monkeypatch.setattr(analyzer, "_get_test_result_and_data", fake_train)

    base_split = train_test_split(
        analyzer.inputs_df,
        analyzer.outputs_df,
        test_size=0.25,
        random_state=40,
    )

    results, prediction_groups = analyzer._get_results(
        2,
        generator,
        "model",
        base_split=base_split,
        random_state=40,
        test_size=0.25,
        save_predictions=False,
    )

    assert generator.random_states == [40, 41]
    assert [call[2] for call in calls] == [40, 41, 41, 42]
    assert all(call[3] == 0.25 for call in calls)
    assert all(call[0] is base_split for call in calls)
    assert calls[0][0] is calls[1][0]
    assert calls[2][0] is calls[3][0]
    assert results["run_id"].tolist() == ["run_test"] * 4
    assert results["model_seed"].tolist() == [40, 41, 41, 42]
    assert results["architecture"].tolist() == [
        "[16, 12]",
        "[6, 4, 2]",
        "[16, 12]",
        "[6, 4, 2]",
    ]
    assert set(prediction_groups) == {
        "wide",
        "narrow",
    }
    assert len(prediction_groups["wide"]) == 2
    assert len(prediction_groups["narrow"]) == 2

def test_get_results_uses_noise_generator_without_training(
    analyzer,
    regression_data,
    monkeypatch,
):
    inputs, outputs = regression_data
    base_split = train_test_split(
        inputs,
        outputs,
        test_size=0.2,
        random_state=42,
    )

    X_train_base, X_test_fixed, y_train_base, y_test_fixed = (
        analyzer._validate_split(base_split)
    )

    base_training_dataset = pd.concat(
        [X_train_base, y_train_base],
        axis=1,
    )

    generator = NoiseGenerator(
        base_training_dataset,
        standard_deviations=[0.1],
    )

    analyzer._run_id = "run_test"
    calls = []

    def fake_train(
        *,
        split,
        architecture,
        random_state,
        test_size,
    ):
        calls.append(
            {
                "split": split,
                "architecture": architecture,
                "random_state": random_state,
                "test_size": test_size,
            }
        )

        actuals = split[3]
        predictions = np.zeros(
            actuals.shape,
            dtype=np.float32,
        )

        return {"loss": 1.0}, predictions, actuals

    monkeypatch.setattr(
        analyzer,
        "_get_test_result_and_data",
        fake_train,
    )

    results, prediction_groups = analyzer._get_results(
        n_iter=1,
        generator=generator,
        study="data",
        base_split=base_split,
        random_state=42,
        test_size=0.2,
        save_predictions=False,
    )

    assert len(calls) == 1

    call = calls[0]

    # Data studies should use the baseline architecture.
    assert call["architecture"] is None

    # Verify seed and test-size propagation.
    assert call["random_state"] == 42
    assert call["test_size"] == 0.2

    # Verify the resulting metadata row.
    assert len(results) == 1
    assert results.iloc[0]["run_id"] == "run_test"
    assert results.iloc[0]["study"] == "data"
    assert results.iloc[0]["variable"] == "std_0.1"
    assert results.iloc[0]["model_seed"] == 42
    assert results.iloc[0]["loss"] == pytest.approx(1.0)
    assert results.iloc[0]["architecture"] == "[6, 4]"

    expected_variation = generator.generate(
        random_state=42
    )["std_0.1"]

    expected_data_train = expected_variation.dataset

    expected_split = (
        expected_data_train[inputs.columns],
        X_test_fixed,
        expected_data_train[outputs.columns],
        y_test_fixed,
    )

    actual_split = call["split"]

    assert len(actual_split) == 4

    for actual_frame, expected_frame in zip(
        actual_split,
        expected_split,
    ):
        pd.testing.assert_frame_equal(
            actual_frame,
            expected_frame,
        )

    assert set(prediction_groups) == {"std_0.1"}
    assert len(prediction_groups["std_0.1"]) == 1

def test_get_results_rejects_non_noise_data_variation(analyzer):
    analyzer._run_id = "run_test"

    generator = RecordingGenerator(
        {
            "std_0.1": pd.concat(
                [analyzer.inputs_df, analyzer.outputs_df],
                axis=1,
            )
        }
    )

    base_split = train_test_split(
        analyzer.inputs_df,
        analyzer.outputs_df,
        test_size=0.2,
        random_state=42,
    )

    with pytest.raises(
        TypeError,
        match='Data studies must generate NoiseVariation values.',
    ):
        analyzer._get_results(
            n_iter=1,
            generator=generator,
            study='data',
            base_split=base_split,
            random_state=42,
            test_size=0.2,
            save_predictions=False,
        )

def test_get_results_requires_run_id(analyzer):
    base_split = train_test_split(
        analyzer.inputs_df,
        analyzer.outputs_df,
        test_size=0.2,
        random_state=1,
    )

    with pytest.raises(ValueError, match="_run_id is None"):
        analyzer._get_results(
            1,
            RecordingGenerator({"wide": FnnArchitecture((4,))}),
            "model",
            base_split=base_split,
            random_state=1,
            test_size=0.2,
            save_predictions=False,
        )


def test_save_predictions_writes_expected_hdf5_group(
    analyzer,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    analyzer._run_id = "run_test"
    predictions = np.array([[1.0], [2.0]], dtype=np.float32)
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


def test_save_predictions_rejects_shape_mismatch(analyzer):
    analyzer._run_id = "run_test"

    with pytest.raises(ValueError, match="matching shapes"):
        analyzer._save_predictions_and_actuals(
            np.zeros((2, 1)),
            pd.DataFrame({"y": [1.0]}),
            study="model",
            label="wide",
            iteration=0,
        )


def test_run_metadata_build_record_lookup_and_join(analyzer):
    analyzer._run_id = "run_test"
    analyzer._runs_metadata_df = empty_runs_metadata()
    analyzer._results_df = pd.DataFrame(
        [
            {
                "run_id": "run_test",
                "iteration": 0,
                "study": "model",
                "variable": "wide",
            }
        ]
    ).reindex(columns=BiasAnalyzer.RESULT_COLUMNS)

    analyzer._record_run_metadata(
        n_iter=2,
        random_state=42,
        test_size=0.2,
    )

    metadata = analyzer.get_run_metadata("run_test")
    assert metadata["random_state"] == 42
    assert metadata["test_size"] == pytest.approx(0.2)
    assert metadata["n_iter"] == 2
    assert json.loads(metadata["baseline_architecture"]) == [6, 4]
    assert len(analyzer.get_results_with_metadata()) == 1


def test_record_run_metadata_rejects_duplicate_run_id(analyzer):
    analyzer._run_id = "run_test"
    analyzer._runs_metadata_df = pd.DataFrame(
        [{"run_id": "run_test"}]
    ).reindex(columns=BiasAnalyzer.RUN_METADATA_COLUMNS)

    with pytest.raises(ValueError, match="Duplicate run ID"):
        analyzer._record_run_metadata(
            n_iter=1,
            random_state=None,
            test_size=0.2,
        )


def test_get_run_metadata_rejects_unknown_id(analyzer):
    analyzer._runs_metadata_df = empty_runs_metadata()

    with pytest.raises(KeyError, match="Unknown run ID"):
        analyzer.get_run_metadata("missing")


@pytest.mark.parametrize("n_iter", [1.5, "2", True, None])
def test_run_bias_studies_rejects_non_integer_iterations(analyzer, n_iter):
    with pytest.raises(TypeError, match="n_iter must be an integer"):
        analyzer.run_bias_studies(
            {"n_iter": n_iter, "studies": {"model": {"wide": {}}}},
            save_results=False,
            save_predictions=False,
        )


@pytest.mark.parametrize("n_iter", [0, -1])
def test_run_bias_studies_rejects_non_positive_iterations(analyzer, n_iter):
    with pytest.raises(ValueError, match="n_iter must be greater than 0"):
        analyzer.run_bias_studies(
            {"n_iter": n_iter, "studies": {"model": {"wide": {}}}},
            save_results=False,
            save_predictions=False,
        )


def test_run_bias_studies_rejects_invalid_settings(analyzer):
    with pytest.raises(ValueError, match="At least one study"):
        analyzer.run_bias_studies(
            {"n_iter": 1, "studies": {}},
            save_results=False,
            save_predictions=False,
        )

    with pytest.raises(ValueError, match="Unsupported studies"):
        analyzer.run_bias_studies(
            {"n_iter": 1, "studies": {"unknown": {}}},
            save_results=False,
            save_predictions=False,
        )

    with pytest.raises(ValueError, match="Unsupported run settings"):
        analyzer.run_bias_studies(
            {"n_iter": 1, "studies": {"model": {}}, "extra": True},
            save_results=False,
            save_predictions=False,
        )


def test_run_bias_studies_records_one_metadata_row_and_persists_tables(
    analyzer,
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)

    analyzer.run_bias_studies(
        {
            "n_iter": 1,
            "random_state": 9,
            "test_size": 0.2,
            "studies": {
                "model": {
                    "wide": {
                        "layers": (1, 2),
                        "neurons": (4, 6),
                    }
                }
            },
        },
        save_results=True,
        save_predictions=False,
    )
    run_evaluations = analyzer._evaluations[analyzer._run_id]
    assert set(run_evaluations) == {"model"}
    assert set(run_evaluations["model"]) == {"wide"}
    assert set(run_evaluations["model"]["wide"]) == {
        "averaging",
        "pointwise",
    }

    assert len(analyzer._results_df) == 1
    assert len(analyzer._runs_metadata_df) == 1
    assert analyzer._results_df["run_id"].iloc[0] == analyzer._run_id
    assert analyzer._runs_metadata_df["run_id"].iloc[0] == analyzer._run_id
    assert analyzer._runs_metadata_df["random_state"].iloc[0] == 9
    assert (tmp_path / BiasAnalyzer.RESULTS_FILENAME).exists()
    assert (tmp_path / BiasAnalyzer.RUN_METADATA_FILENAME).exists()


def test_run_bias_studies_save_false_keeps_data_in_memory(analyzer, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    analyzer.run_bias_studies(
        {
            "n_iter": 1,
            "random_state": 4,
            "test_size": 0.2,
            "studies": {
                "model": {
                    "wide": {
                        "layers": (1, 2),
                        "neurons": (4, 6),
                    }
                }
            },
        },
        save_results=False,
        save_predictions=False,
    )
    run_evaluations = analyzer._evaluations[analyzer._run_id]
    assert set(run_evaluations) == {"model"}
    assert set(run_evaluations["model"]) == {"wide"}
    assert set(run_evaluations["model"]["wide"]) == {
        "averaging",
        "pointwise",
    }

    assert len(analyzer._results_df) == 1
    assert len(analyzer._runs_metadata_df) == 1
    assert not (tmp_path / BiasAnalyzer.RESULTS_FILENAME).exists()
    assert not (tmp_path / BiasAnalyzer.RUN_METADATA_FILENAME).exists()


def test_decompose_variance_rejects_invalid_inputs(analyzer):
    with pytest.raises(TypeError, match="confidence must be numeric"):
        analyzer.decompose_variance(confidence="0.95")

    with pytest.raises(ValueError, match="between 0 and 1"):
        analyzer.decompose_variance(confidence=1.0)

    with pytest.raises(ValueError, match="Unsupported study views"):
        analyzer.decompose_variance(view=["unknown"])


def test_decompose_variance_uses_numeric_result_fields(analyzer):
    analyzer._results_df = pd.DataFrame(
        {
            "run_id": ["run_test", "run_test"],
            "iteration": [0, 1],
            "study": ["model", "model"],
            "variable": ["wide", "wide"],
            "architecture": ["[4]", "[6]"],
            "loss": [1.0, 3.0],
            "mse": [1.0, 3.0],
        }
    ).reindex(columns=BiasAnalyzer.RESULT_COLUMNS)

    decomposition = analyzer.decompose_variance(view=["model"])
    wide = decomposition["model"]["wide"]["wide"]

    assert wide["averages"]["loss"] == pytest.approx(2.0)
    assert "architecture" not in wide["averages"]


def test_plot_disagreement_map_rejects_invalid_selections(analyzer):
    with pytest.raises(ValueError, match="Unsupported study views"):
        analyzer.plot_disagreement_map(view=["unknown"])

    with pytest.raises(ValueError, match="Unsupported plot types"):
        analyzer.plot_disagreement_map(plot_type=["unknown"])
