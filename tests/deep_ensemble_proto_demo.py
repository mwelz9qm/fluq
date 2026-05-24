import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import pyMAISE as mai
from pyMAISE.utils.hyperparameters import Choice
from pyMAISE.datasets import load_MITR
from pyMAISE.preprocessing import train_test_split, scale_data

from deep_ensembles._deep_ensemble import DeepEnsembleHyperModel


def main():
    # same pyMAISE setup as normal
    mai.settings.init("regression", verbosity=1)

    # load MIT reactor data
    data, inputs, outputs = load_MITR()
    xtrain, xtest, ytrain, ytest = train_test_split(data=[inputs, outputs], test_size=0.3)
    xtrain, xtest, xscaler = scale_data(xtrain, xtest, scaler=MinMaxScaler())
    ytrain, ytest, yscaler = scale_data(ytrain, ytest, scaler=MinMaxScaler())

    # set hyperparameters
    parameters = {
        "structural_params": {
            "Dense_1": {
                "units": Choice([16, 32]),
                "activation": "relu",
                "num_layers": 1,
            },
            "Dense_2": {
                "units": 1,
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
            "epochs": 20,
            "batch_size": 16,
            "validation_split": 0.2
        }
    }

    # Setup Deep Ensemble
    print("Initializing DeepEnsembleHyperModel...")
    de_hypermodel = DeepEnsembleHyperModel(
        parameters=parameters,
        input_shape=(xtrain.shape[1],),
        name="test_deep_ensemble",
        num_models=3,
    )

    # Set up Keras Tuner with our DE HyperModel
    print("Setting up Keras Tuner RandomSearch...")
    tuner = kt.RandomSearch(
        hypermodel=de_hypermodel,
        objective="val_loss",
        max_trials=2,
        directory="tuning_dir",
        project_name="de_test",
        overwrite=True
    )

    # Run the tuner search
    print("Starting Keras Tuner Search...")
    tuner.search(x=xtrain.values, y=ytrain.values)

    # Retrieve the best model (which returns the wrapper class)
    print("Retrieving best model...")
    best_model = tuner.get_best_models(num_models=1)[0]
    
    # Make predictions using the ensemble
    print("Running predictions with the best ensemble...")
    predictions = best_model.predict(xtest.values)
    
    print("\n==================================")
    print(f"Ensemble Size: {de_hypermodel.num_models}")
    print(f"Expected Predictions Shape: ({de_hypermodel.num_models}, {xtest.shape[0]}, 1)")
    print("Actual Predictions Shape:", predictions.shape)
    # print("Predictions:\n", predictions)
    plot(predictions)


def plot(predictions):
    plt.figure(figsize=(12, 6))

    for i, p in enumerate(predictions):
        plt.plot(p, label=f"Model {i + 1}", alpha=0.7)
    plt.plot(np.mean(predictions, axis=0), label="Mean", color="black", alpha=0.7)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
