import unittest
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from pyMAISE.datasets import load_chf
from bias_variance.BiasAnalyzer import BiasAnalyzer

class TestBiasAnalyzer(unittest.TestCase):

    def test_run_sampling_bias_study(self):
        # load chf data and create input and output dataframes
        train_data, X_train, y_train, test_data, X_test, y_test = load_chf()
        inputs_df = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_test)], ignore_index=True)
        outputs_df = pd.concat([pd.DataFrame(y_train), pd.DataFrame(y_test)], ignore_index=True)

        # rename columns since column names are not provided
        inputs_df.columns = [f'feature_{i}' for i in np.arange(inputs_df.shape[1])]
        outputs_df.columns = ['target']

        # use keras functional api to build base model (architecture = wide)
        inputs = tf.keras.Input(shape=(inputs_df.shape[1],))
        x = layers.Dense(64, activation='relu')(inputs)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='relu', name='predictions')(x)
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='functional_model')

        # create BiasAnalyzer object
        analyzer = BiasAnalyzer(
            None,
            model,
            inputs_df,
            outputs_df,
            pd.DataFrame(),
            pd.DataFrame(),
            test_size=0.2,
            random_state=42,
            is_predictions_saved=False
        )

        # run sampling study
        analyzer.run_sampling_bias_study(strategies=['bootstrap', 'lhs']) # TODO: resolve issue w/ 'stratified' sampling option
        
        # show results
        print(analyzer.results_df)

if __name__ == '__main__':
    unittest.main()