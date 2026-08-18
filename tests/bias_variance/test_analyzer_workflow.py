import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from bias_variance.analyzer import (
    BiasAnalyzer,
)
from bias_variance.config import RunConfigBuilder, StudyBias
from bias_variance.generators.fnn_architecture import (
    ArchitectureName,
    FnnArchitectureGenerator,
    FnnArchitectureGeneratorConfig,
    FnnRandomArchitectureConfig,
)
from bias_variance.generators.noise import NoiseGenerator, NoiseGeneratorConfig
from bias_variance.generators.sampling import (
    SamplingGenerator,
    SamplingGeneratorConfig,
)
from bias_variance.models.evaluation import EvaluationMethod, MetricName
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
    generator_config = FnnArchitectureGeneratorConfig(
        range_architectures={
            ArchitectureName.WIDE: FnnRandomArchitectureConfig(
                layer_range=(1, 2),
                size_range=(2, 3),
            )
        },
        taper_architectures={},
    )
    analyzer = BiasAnalyzer(tmp_path / 'results.sqlite3')

    analyzer.run_studies(
        inputs,
        outputs,
        run_settings={
            'variation_generator_configs': (generator_config,),
            'evaluation_methods': (EvaluationMethod.POINTWISE,),
            'n_iter': 2,
            'test_size': 0.5,
            'test_metrics': (MetricName.MSE,),
            'random_state': 7,
            'base_architecture': (2,),
        },
        training_config=TrainingConfig(epochs=0, device='cpu'),
    )
    results = analyzer.decompose_bias_and_variance()

    with sqlite3.connect(analyzer.db_path) as connection:
        first_evaluation_count = connection.execute(
            'SELECT COUNT(*) FROM evaluations'
        ).fetchone()[0]

    repeated_results = analyzer.decompose_bias_and_variance()
    run_history = analyzer.get_run_history()

    with sqlite3.connect(analyzer.db_path) as connection:
        repeated_evaluation_count = connection.execute(
            'SELECT COUNT(*) FROM evaluations'
        ).fetchone()[0]

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
    pd.testing.assert_frame_equal(repeated_results, results)
    assert repeated_evaluation_count == first_evaluation_count
    assert len(run_history) == 1
    assert tuple(run_history.columns) == (
        'run_id',
        'created_at',
        'n_iter',
        'test_size',
        'test_metrics',
        'optimizer',
        'learning_rate',
        'loss',
        'epochs',
        'batch_size',
        'device',
        'input_columns',
        'output_columns',
        'base_architecture',
    )
    assert run_history.loc[0, 'n_iter'] == 2
    assert run_history.loc[0, 'test_size'] == 0.5
    assert run_history.loc[0, 'test_metrics'] == ('mse',)
    assert run_history.loc[0, 'input_columns'] == ('x',)
    assert run_history.loc[0, 'output_columns'] == ('y', 'z')


def test_get_run_history_returns_typed_empty_frame(tmp_path: Path) -> None:
    history = BiasAnalyzer(tmp_path / 'empty-results.sqlite3').get_run_history()

    assert history.empty
    assert tuple(history.columns) == (
        'run_id',
        'created_at',
        'n_iter',
        'test_size',
        'test_metrics',
        'optimizer',
        'learning_rate',
        'loss',
        'epochs',
        'batch_size',
        'device',
        'input_columns',
        'output_columns',
        'base_architecture',
    )


def test_run_studies_accepts_mapped_run_settings(tmp_path: Path) -> None:
    inputs = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    outputs = pd.DataFrame({'y': [0.0, 2.0, 4.0, 6.0]})
    analyzer = BiasAnalyzer(tmp_path / 'raw-results.sqlite3')

    analyzer.run_studies(
        inputs,
        outputs,
        {
            'variation_generator_configs': {
                StudyBias.MODEL.value: {
                    'range_architectures': {
                        ArchitectureName.WIDE: FnnRandomArchitectureConfig(
                            layer_range=(1, 2),
                            size_range=(2, 3),
                        ),
                    },
                    'taper_architectures': {},
                },
            },
            'n_iter': 1,
            'test_size': 0.5,
            'test_metrics': ['mse'],
            'evaluation_methods': ['pointwise'],
            'random_state': 7,
        },
        training_config=TrainingConfig(epochs=0, device='cpu'),
    )


def test_run_studies_rejects_unknown_run_setting(
    tmp_path: Path,
) -> None:
    inputs = pd.DataFrame({'x': [0.0, 1.0]})
    outputs = pd.DataFrame({'y': [0.0, 1.0]})

    with pytest.raises(ValueError, match='Unknown run settings'):
        BiasAnalyzer(tmp_path / 'mixed.sqlite3').run_studies(
            inputs,
            outputs,
            {'unknown': True},
        )


def test_builder_uses_defaults_for_undefined_run_settings() -> None:
    inputs = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    outputs = pd.DataFrame({'y': [0.0, 1.0, 2.0, 3.0]})

    config = (
        RunConfigBuilder()
        .set_X(inputs)
        .set_Y(outputs)
        .apply_run_settings(None)
        .build()
    )

    assert config.n_iter == 100
    assert config.test_size == 0.2
    assert config.evaluation_methods == (
        EvaluationMethod.AVERAGING,
        EvaluationMethod.POINTWISE,
    )
    assert {study.study_bias for study in config.studies} == set(StudyBias)


def test_generators_construct_their_own_default_configs() -> None:
    assert isinstance(
        FnnArchitectureGenerator().settings,
        FnnArchitectureGeneratorConfig,
    )
    assert isinstance(NoiseGenerator().settings, NoiseGeneratorConfig)
    assert isinstance(SamplingGenerator().settings, SamplingGeneratorConfig)
    assert FnnArchitectureGeneratorConfig().variation_labels == (
        'wide',
        'narrow',
        'taper',
        'reverse_taper',
        'combined_taper',
    )


def test_noise_generator_changes_values_without_mutating_base_data() -> None:
    dataset = pd.DataFrame({'x': [1.0, 2.0, 3.0, 4.0]})
    original = dataset.copy()
    generator = NoiseGenerator(NoiseGeneratorConfig((0.1,)))
    generator.base_dataset = dataset

    generated = next(generator.generate(random_state=7)).generated

    assert not generated.equals(original)
    pd.testing.assert_frame_equal(dataset, original)
