import unittest
import pandas as pd
from pyMAISE.datasets import load_chf
from common.sampling._sampling import get_random_samples, get_stratified_random_samples, generate_latin_hypercube_samples

class TestSampling(unittest.TestCase):
    def test_get_random_samples(self):
        print('testing get_random_samples()')
        train_data, X_train, y_train, test_data, X_test, y_test = load_chf()
        df = pd.DataFrame(train_data)
        print('df shape:', df.shape)
        print('----------------------------')
        print(get_random_samples(df, 100, 42, True))

    def test_get_stratified_random_samples(self):
        print('testing get_stratified_random_samples()')
        train_data, X_train, y_train, test_data, X_test, y_test = load_chf()
        df = pd.DataFrame(train_data)
        col_index = 0
        df[f'quantile_rank_col_{col_index}'] = pd.qcut(df.iloc[:,col_index], q=4, labels=False)

        print('df shape:', df.shape)
        print('----------------------------')
        print(get_stratified_random_samples(df, 101, f'quantile_rank_col_{col_index}', 42, True))
        # should the user have option to remove stratified col after sampling? (i.e. is_stratified_col_removed:bool=True)

    def test_generate_latin_hypercube_samples(self):
        pass

if __name__ == '__main__':
    unittest.main()