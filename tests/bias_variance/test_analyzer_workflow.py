from pathlib import Path

import pandas as pd

from bias_variance.analyzer import (
    BiasAnalyzer,
    DatasetSplit,
    RunBaseline,
    RunConfig,
    Study,
    StudyName,
)
from bias_variance.generators.fnn_architecture import (
    ArchitectureName,
    FnnArchitectureGeneratorConfig,
    FnnRandomArchitectureConfig,
)
from bias_variance.models.evaluation import EvaluationMethod, MetricName
from bias_variance.models.fnn import FnnArchitecture
from bias_variance.models.training import TrainingConfig


def test_workflows_share_file_database_and_return_multioutput_results(
    tmp_path: Path,
) -> None:
    inputs = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    outputs = pd.DataFrame(
        {
            'y': [0.0, 2.0, 4.0, 6.0],
            'z': [1.0, 2.0, 3.0, 4.0],
        }
    )
    split = DatasetSplit(
        x_train=inputs.iloc[:2],
        x_test=inputs.iloc[2:],
        y_train=outputs.iloc[:2],
        y_test=outputs.iloc[2:],
    )
    baseline = RunBaseline(
        inputs=inputs,
        outputs=outputs,
        split=split,
        architecture=FnnArchitecture((2,)),
    )
    generator_config = FnnArchitectureGeneratorConfig(
        range_architectures={
            ArchitectureName.WIDE: FnnRandomArchitectureConfig(
                layer_range=(1, 2),
                size_range=(2, 3),
            )
        }
    )
    run_config = RunConfig(
        baseline=baseline,
        studies=(
            Study(
                study_name=StudyName.MODEL,
                evaluation_method=EvaluationMethod.POINTWISE,
                generator_config=generator_config,
            ),
        ),
        n_iter=2,
        test_metrics=frozenset((MetricName.MSE,)),
        random_state=7,
    )
    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3')

    analyzer.run_studies(
        run_config,
        TrainingConfig(epochs=0, device='cpu'),
    )
    results = analyzer.decompose_bias_and_variance()

    assert tuple(results.columns) == (
        'study_name',
        'group_name',
        'evaluation_method',
        'bias',
        'variance',
    )
    assert results.loc[0, 'study_name'] == 'model'
    assert results.loc[0, 'group_name'] == 'wide'
    assert results.loc[0, 'evaluation_method'] == 'pointwise'
    assert len(results.loc[0, 'bias']) == 2
    assert len(results.loc[0, 'variance']) == 2
