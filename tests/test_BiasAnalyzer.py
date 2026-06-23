import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from bias_variance.BiasAnalyzer import BiasAnalyzer

def test_bias_analyzer_init():
    BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
    train_df = pd.read_csv(BENCHMARKS / 'chf_train_synth.csv')
    test_df = pd.read_csv(BENCHMARKS / 'chf_test_synth.csv')
    df = pd.concat([train_df, test_df])
    inputs_df = df.iloc[:,:6]
    outputs_df = df.iloc[:,6:]

    analyer = BiasAnalyzer(inputs_df, outputs_df)