import pytest
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bias_variance.BiasAnalyzer import BiasAnalyzer
from common.sampling._sampling import (
    get_random_samples
)
from common.sampling.Sampler import Sampler

@pytest.fixture
def test_input_and_output_dfs() -> tuple[pd.DataFrame, pd.DataFrame]:
    benchmarks = Path(__file__).resolve().parents[1] / 'benchmarks'
    df = pd.concat([
        pd.read_csv(benchmarks / 'chf_train_synth.csv'),
        pd.read_csv(benchmarks / 'chf_test_synth.csv')
    ])
    inputs_df = df.iloc[:,:6]
    outputs_df = df.iloc[:,7:]
    return (inputs_df, outputs_df)

def test_constructor(test_input_and_output_dfs):
    inputs_df, outputs_df = test_input_and_output_dfs
    analyzer = BiasAnalyzer(inputs_df, outputs_df)
    pd.testing.assert_frame_equal(analyzer.inputs_df, inputs_df)
    pd.testing.assert_frame_equal(analyzer.outputs_df, outputs_df)
    assert analyzer.model_settings == {
        'hidden_layers': [32, 32],
        'activation': 'relu',
        'optimizer': 'adam',
        'loss': 'mse',
        'metrics': ['rmse', 'r2', 'mse', 'mae'],
        'epochs': 100,
        'batch_size': 10,
        'verbose': 0
    }
    assert analyzer.test_size == 0.2
    assert analyzer.random_state is None
    assert analyzer._model is None
    assert analyzer._init_weights is None
    assert analyzer._results_df is None