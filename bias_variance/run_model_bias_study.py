import numpy as np
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Architecture Generator Helper Function
def _build_architecture(
        self,
        architecture_type,
        architecture_id
        ):
    
    model = Sequential()

    # Placeholder for architecture generator. For now, we will generate 5 different architectures
    #  based on the same base architecture, but with varying layer widths and depths. 
    # Layer widths, depths, and sizes are arbitrary and can be changed in future iterations of this function.
    # The architecture_id parameter can be used to seed the random generation of architectures 
    # in future iterations of this function.

    if architecture_type == 'wide':
        model.add(Input(shape=(self.n_features,)))
        model.add(Dense(
            256,
            activation='relu',
        ))
        model.add(Dense(256, activation='relu'))

    elif architecture_type == 'narrow':
        model.add(Input(shape=(self.n_features,)))
        model.add(Dense(
            32,
            activation='relu',
        ))
        model.add(Dense(32, activation='relu'))

    elif architecture_type == 'taper':
        model.add(Input(shape=(self.n_features,)))
        model.add(Dense(
            256,
            activation='relu',
        ))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))

    elif architecture_type == 'reverse_taper':
        model.add(Input(shape=(self.n_features,)))
        model.add(Dense(
            32,
            activation='relu',
        ))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))

    elif architecture_type == 'combined_taper':
        model.add(Input(shape=(self.n_features,)))
        model.add(Dense(
            256,
            activation='relu',
        ))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse', metrics=self.metrics)

    return model

# Metric Helper Function
def _evaluate_predictions(
        self,
        y_true,
        y_pred
        ):
    
    mse = mean_squared_error(y_true, y_pred)

    return {
        'rmse': np.sqrt(mse),
        'mse': mse,
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
def run_model_bias_study(
        self, 
        architecture_types:list[str]=['wide','narrow','taper','reverse_taper','combined_taper'], 
        n_architectures=10,
        metrics:list[str]=['rmse','r2','mse','mae']
        ) -> None:
    
    for architecture_type in architecture_types:
        for architecture_idx in range(n_architectures):
            model = self._build_architecture(architecture_type, architecture_idx)

            history = model.fit(
                self.X_train, 
                self.y_train,
                epochs=100, 
                batch_size=32,
                verbose=0
            )

            y_pred = model.predict(self.X_test).flatten()

            scores = self._evaluate_predictions(self.y_test, y_pred)

            results_row = {
                "study": "model",
                "architecture_type": architecture_type,
                "architecture_id": architecture_idx,
            }

            for metric in metrics:
                results_row[metric] = scores[metric]

            self.results = pd.DataFrame(self.results)

            if self.save_predictions:
                predictions_row = {
                    "study": "model",
                    "architecture_type": architecture_type,
                    "architecture_id": architecture_idx,
                    "y_true": self.y_test.tolist(),
                    "y_pred": y_pred.tolist()
                }

            self.predictions = pd.DataFrame(self.predictions)
