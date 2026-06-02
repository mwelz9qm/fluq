import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.utils.hyperparameters import Choice
from pyMAISE.datasets import load_MITR
from pyMAISE.preprocessing import train_test_split, scale_data

from deep_ensembles._deep_ensemble import DeepEnsembleHyperModel


#################################
### FULLY AI GENERATED SCRIPT ###
#################################
# ^^ Based on deep_ensemble_proto_demo.py


def main():
    # Setup pyMAISE global settings
    mai.settings.init("regression", verbosity=1, )#random_state=42)

    # Load and scale MIT reactor data
    data, inputs, outputs = load_MITR()
    xtrain, xtest, ytrain, ytest = train_test_split(data=[inputs, outputs], test_size=0.3)
    xtrain, xtest, xscaler = scale_data(xtrain, xtest, scaler=MinMaxScaler())
    ytrain, ytest, yscaler = scale_data(ytrain, ytest, scaler=MinMaxScaler())
    print("xtrain shape:", xtrain.shape)
    print("ytrain shape:", ytrain.shape)

    # Define standard model settings for the base model
    model_settings = {
        "models": ["BaseNN"],
        "BaseNN": {
            "structural_params": {
                "Dense_1": {
                    "units": Choice([16, 32]),
                    "activation": "relu",
                    "num_layers": 1,
                },
                "Dense_2": {
                    "units": ytrain.shape[-1],
                    "activation": "linear",
                    "num_layers": 1,
                }
            },
            "optimizer": "Adam",
            "Adam": {
                "learning_rate": 0.001
            },
            "compile_params": {
                "loss": "mse",
                "metrics": ["mae"]
            },
            "fitting_params": {
                "epochs": 10,
                "batch_size": 16,
                "validation_split": 0.2
            }
        }
    }

    # 1. Run pyMAISE Tuner for the BaseNN
    print("Starting pyMAISE Tuner Search...")
    tuner = mai.Tuner(xtrain, ytrain, model_settings=model_settings)
    
    # Inject DeepEnsembleHyperModel to tune the base architecture and build the ensemble later
    num_models = 3
    tuner._models["BaseNN"] = DeepEnsembleHyperModel(
        parameters=model_settings["BaseNN"],
        input_shape=(xtrain.shape[1],),
        name="BaseNN",
        num_models=num_models,
        tune_ensemble=False,
    )
    
    # We use a small number of trials for fast testing
    search_data = tuner.nn_random_search(
        objective="mean_absolute_error",
        max_trials=2,
        directory="tuning_dir",
        project_name="mai_tuner_de_test",
        overwrite=True
    )

    # 2. Extract best hyperparameters and the underlying DeepEnsembleHyperModel
    top_configs, de_hypermodel = search_data["BaseNN"]
    
    # top_configs["params"] is a pandas Series containing the keras_tuner.HyperParameters object
    best_hps = top_configs["params"].iloc[0]

    # 3. Build the ensemble using the DeepEnsembleHyperModel method
    print("\nBuilding Deep Ensemble using best model hyperparameters...")
    ensemble = de_hypermodel.build_ensemble(best_hps)

    # 4. Fit the Deep Ensemble
    print("\nFitting Deep Ensemble...")
    ensemble.fit(
        x=xtrain.values,
        y=ytrain.values,
        **model_settings["BaseNN"]["fitting_params"]
    )
    
    # 5. Predictions and Uncertainty
    print("\nRunning predictions with the best ensemble...")
    predictions, mean, epistemic, aleatoric = ensemble.predict_with_uncertainty(xtest.values)
    
    print("\n==================================")
    print(f"Ensemble Size: {num_models}")
    print(f"Expected Predictions Shape: ({num_models}, {xtest.shape[0]}, 1)")
    print("Actual Predictions Shape:", predictions.shape)
    
    print("Test passed successfully!")


if __name__ == "__main__":
    main()
