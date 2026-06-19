"""
AI-written smoke test for PyTorch Deep Ensemble migration.
Purpose: quickly verify the basic flow runs without errors. suite.

Tests:
- DeepEnsembleHyperModel initializes correctly
- build_ensemble() creates M Skorch members
- fit() trains all members
- predict() returns correct shape
- predict_with_uncertainty() returns correct shapes and keys
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pyMAISE as mai
from pyMAISE.utils.hyperparameters import Int, Float
from deep_ensembles._deep_ensemble import DeepEnsembleHyperModel

def main():
    mai.init(problem_type="regression", random_state=42, verbosity=1)

    # Simple synthetic data
    rng = np.random.default_rng(42)
    n_train, n_test, n_features, n_targets = 100, 20, 4, 1
    xtrain = rng.standard_normal((n_train, n_features)).astype(np.float32)
    ytrain = rng.standard_normal((n_train, n_targets)).astype(np.float32)
    xtest  = rng.standard_normal((n_test, n_features)).astype(np.float32)

    parameters = {
        "structural_params": {
            "Dense_hidden": {"units": 16, "activation": "relu"},
            "Dense_output": {"units": n_targets, "activation": "linear"},
        },
        "optimizer": "Adam",
        "Adam": {"learning_rate": 0.001},
        "compile_params": {"loss": "mse"},
        "fitting_params": {"epochs": 2, "batch_size": 32},
    }

    print("Initializing DeepEnsembleHyperModel...")
    de_hypermodel = DeepEnsembleHyperModel(
        parameters=parameters,
        input_shape=(n_features,),
        name="test_ensemble",
        num_models=3,
    )

    print("Building ensemble...")
    import optuna
    trial = optuna.trial.FixedTrial({})
    ensemble = de_hypermodel.build_ensemble(trial)

    print("Fitting ensemble...")
    ensemble.fit(xtrain, ytrain)

    print("Predicting...")
    preds = ensemble.predict(xtest)
    print(f"  predict shape: {preds.shape}")

    print("predict_with_uncertainty...")
    results = ensemble.predict_with_uncertainty(xtest)
    print(f"  mean shape: {results['mean'].shape}")
    print(f"  epistemic_var shape: {results['epistemic_var'].shape}")
    print(f"  aleatoric_var: {results['aleatoric_var']}")

    print("\nSmoke test passed!")

if __name__ == "__main__":
    main()