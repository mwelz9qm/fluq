import numpy as np
import pyMAISE as mai
from pyMAISE import Int
from pyMAISE.preprocessing import train_test_split

from deep_ensembles import DeepEnsembleHyperModel


def test_deep_ensemble():
    def test(ensemble: DeepEnsembleHyperModel):
        assert ensemble.num_models == 3

    # Generate pseudo data
    def linear_data(seed=42):
        rng = np.random.default_rng(seed)
        x = rng.random((10, 4))
        y = np.sum(x, axis=1, keepdims=True)
        noise = rng.normal(0, 0.3, size=y.shape)
        y += noise
        return x, y

    # Define feed forward neural network settings
    nn_settings = {
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
                "num_layers": [2, 3],
                "units": Int(min_value=25, max_value=250),
                "activation": "relu",
                "kernel_initializer": "normal",
            },
            "Dense_output": {
                "units": 22,
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

    ### Regression ###
    # Init model
    mai.settings.init(problem_type="regression")
    x, y = linear_data()
    xtrain, xtest, ytrain, ytest = train_test_split(x, y)
    base_model = DeepEnsembleHyperModel(
        parameters=nn_settings,
        input_shape=(xtrain.shape[1], ),
        name="test_deep_ensemble",
        num_models=3,
    )

    # TODO we need to figure out how things are being tuned cuy calling ._models and replacing with DeepEnsemble is not a good practice!!
    # Tune
    tuner = mai.Tuner(xtrain, ytrain, model_settings=nn_settings)
    results = tuner.nn_random_search(
        objective="mean_absolute_error",
        max_trials=2,
        directory="tuning_dir",
        project_name="test_deep_ensemble",
        overwrite=True,
    )
    top_configs, de_hypermodel = results["BaseNN"]
    best_hps = top_configs["params"].iloc[0]

    keras_model = base_model.build(hp=best_hps)



