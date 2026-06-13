import unittest
import pandas as pd
import numpy as np
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

        # create BiasAnalyzer object
        analyzer = BiasAnalyzer(
            inputs_df,
            outputs_df
        )

        # run sampling study
        analyzer.run_sampling_bias_study(strategies=['bootstrap'], n_samples=1000, n_iter_per_strategy=100)
        
        # show results
        print(analyzer._results_df)

if __name__ == '__main__':
    unittest.main()