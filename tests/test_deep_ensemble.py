import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyMAISE as mai
from pyMAISE import Int
from pyMAISE.utils.hyperparameters import Choice
from pyMAISE.preprocessing import train_test_split
import xarray as xr
from deep_ensembles import DeepEnsembleHyperModel


def test_deep_ensemble():
    def test(ensemble):
        assert ensemble.num_models == 3

        print("————— All tests passed! —————")

    # Generate pseudo data
    def linear_data(seed=42):
        rng = np.random.default_rng(seed)
        x = rng.random((30, 4))
        y = np.sum(x, axis=1, keepdims=True)
        noise = rng.normal(0, 0.3, size=y.shape)
        y += noise
        return xr.DataArray(x), xr.DataArray(y)

    # Define feed forward neural network settings
    nn_settings = {
        "models": ["DeepEnsembleHyperModel"],   # Can use BaseNN instead
        "DeepEnsembleHyperModel": {
            "structural_params": {
                "Dense_input": {
                    "units": Int(min_value=25, max_value=250),
                    "activation": "relu",
                    "kernel_initializer": "normal",
                    "sublayer": "Dropout",
                    "Dropout": {
                        "rate": 0.5,
                    },
                },
                "Dense_hidden": {
                    "num_layers": Choice([2, 3]),
                    "units": Int(min_value=25, max_value=250),
                    "activation": "relu",
                    "kernel_initializer": "normal",
                },
                "Dense_output": {
                    "units": 1,
                    "activation": "linear",
                    "kernel_initializer": "normal",
                },
            },
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": 0.0001,
            },
            "compile_params": {
                "loss": "mean_absolute_error",
                "metrics": ["mean_absolute_error"],
            },
            "fitting_params": {
                "batch_size": 16,
                "epochs": 50,
                "validation_split": 0.15,
            },
        }
    }

    ### Regression ###
    # Init model
    mai.settings.init(problem_type="regression")
    x, y = linear_data()
    xtrain, xtest, ytrain, ytest = train_test_split(data=[x, y])

    # Tune
    tuner = mai.Tuner(xtrain, ytrain, model_settings=nn_settings)
    results = tuner.nn_random_search(
        objective="mean_absolute_error",
        max_trials=2,
        directory="tuning_dir",
        project_name="test_deep_ensemble",
        overwrite=True,
    )

    # Build Ensemble and inject results
    de_hypermodel = DeepEnsembleHyperModel.inject_for_postprocessing(
        tuner_results=results,
        model_name="DeepEnsembleHyperModel",
        parameters=nn_settings["DeepEnsembleHyperModel"],
        input_shape=(xtrain.shape[1],),
        num_models=3,
    )

    # Run Tests
    test(de_hypermodel)



if __name__ == "__main__":
    test_deep_ensemble()

