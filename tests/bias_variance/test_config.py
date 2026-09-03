import pandas as pd
import pytest

from bias_variance.config import RunConfigBuilder, StudyBias
from bias_variance.generators.fnn_architecture import FnnArchitectureGenerator
from bias_variance.generators.noise import NoiseGenerator, NoiseGeneratorConfig
from bias_variance.generators.sampling import SamplingGenerator
from bias_variance.models.evaluation import EvaluationMethod, MetricName


def _builder() -> RunConfigBuilder:
    return (
        RunConfigBuilder()
        .set_X(pd.DataFrame({'x': [0.0, 1.0]}))
        .set_Y(pd.DataFrame({'y': [0.0, 1.0]}))
    )


def test_build_adds_mse_for_averaging_evaluation() -> None:
    config = (
        _builder()
        .set_test_metrics((MetricName.R2,))
        .set_evaluation_methods((EvaluationMethod.AVERAGING,))
        .build()
    )

    assert config.test_metrics == frozenset({MetricName.MSE, MetricName.R2})


def test_build_does_not_add_mse_without_averaging_evaluation() -> None:
    config = (
        _builder()
        .set_evaluation_methods((EvaluationMethod.POINTWISE,))
        .set_test_metrics((MetricName.R2,))
        .build()
    )

    assert config.test_metrics == frozenset({MetricName.R2})


def test_builder_converts_fully_dictionary_defined_generators() -> None:
    config = _builder().apply_run_settings({
        'variation_generator_configs': {
            'model': {'wide': {'layer_range': (1, 2)}},
            'sampling': {'bootstrap': {'sample_fraction': 0.5}},
            'data': {'standard_deviations': (0.1,)},
        },
        'base_architecture': (8, 4),
        'n_iter': 3,
        'test_size': 0.5,
        'test_metrics': ('mae',),
        'evaluation_methods': ('pointwise',),
        'random_state': 12,
    }).build()

    assert config.baseline.architecture.hidden_layers == (8, 4)
    assert config.n_iter == 3
    assert config.test_size == 0.5
    assert config.test_metrics == frozenset({MetricName.MAE})
    assert config.evaluation_methods == (EvaluationMethod.POINTWISE,)
    assert config.random_state == 12
    assert tuple(type(study.variation_generator) for study in config.studies) == (
        FnnArchitectureGenerator,
        SamplingGenerator,
        NoiseGenerator,
    )
    assert tuple(study.study_bias for study in config.studies) == (
        StudyBias.MODEL,
        StudyBias.SAMPLING,
        StudyBias.DATA,
    )
    assert tuple(
        study.variation_generator.variation_labels
        for study in config.studies
    ) == (('wide',), ('bootstrap',), ('std_0.1',))


def test_explicit_split_overrides_the_generated_split() -> None:
    X = pd.DataFrame({'x': range(6)})
    Y = pd.DataFrame({'y': range(10, 16)})
    split = (X.iloc[:4], X.iloc[4:], Y.iloc[:4], Y.iloc[4:])

    config = (
        RunConfigBuilder()
        .set_X(X)
        .set_Y(Y)
        .set_split(*split)
        .set_random_state(42)
        .build()
    )

    for actual, expected in zip(config.baseline.split, split, strict=True):
        pd.testing.assert_frame_equal(actual, expected)


def test_explicit_split_requires_all_four_frames() -> None:
    frame = pd.DataFrame({'x': [1.0]})

    with pytest.raises(ValueError, match='must either all be provided'):
        RunConfigBuilder().set_split(X_train=frame)


def test_explicit_split_requires_matching_feature_columns() -> None:
    X = pd.DataFrame({'x': range(4)})
    Y = pd.DataFrame({'y': range(4)})

    with pytest.raises(ValueError, match='X_train columns must match'):
        (
            RunConfigBuilder()
            .set_X(X)
            .set_Y(Y)
            .set_split(
                pd.DataFrame({'other': [0, 1]}),
                X.iloc[2:],
                Y.iloc[:2],
                Y.iloc[2:],
            )
            .build()
        )


@pytest.mark.parametrize(
    ('settings', 'error'),
    (
        ([], TypeError),
        ({'n_iter': True}, TypeError),
        ({'n_iter': 0}, ValueError),
        ({'test_size': 0.0}, ValueError),
        ({'test_size': 1.0}, ValueError),
        ({'evaluation_methods': ()}, ValueError),
        ({'evaluation_methods': ('unknown',)}, ValueError),
        ({'test_metrics': ()}, ValueError),
        ({'test_metrics': ('unknown',)}, ValueError),
        ({'variation_generator_configs': {}}, ValueError),
    ),
)
def test_builder_rejects_invalid_run_settings(settings, error) -> None:
    with pytest.raises(error):
        _builder().apply_run_settings(settings).build()


def test_builder_requires_matching_unique_X_and_Y_indexes() -> None:
    X = pd.DataFrame({'x': [1.0, 2.0]}, index=('a', 'b'))
    mismatched_Y = pd.DataFrame({'y': [1.0, 2.0]}, index=('b', 'a'))
    duplicate_X = pd.DataFrame({'x': [1.0, 2.0]}, index=('a', 'a'))
    duplicate_Y = pd.DataFrame({'y': [1.0, 2.0]}, index=('a', 'a'))

    with pytest.raises(ValueError, match='indexes must match'):
        RunConfigBuilder().set_X(X).set_Y(mismatched_Y).build()
    with pytest.raises(ValueError, match='indexes must be unique'):
        RunConfigBuilder().set_X(duplicate_X).set_Y(duplicate_Y).build()


def test_explicit_split_indexes_must_be_disjoint() -> None:
    X = pd.DataFrame({'x': range(4)})
    Y = pd.DataFrame({'y': range(4)})

    with pytest.raises(ValueError, match='indexes must be disjoint'):
        (
            RunConfigBuilder()
            .set_X(X)
            .set_Y(Y)
            .set_split(X.iloc[:3], X.iloc[2:3], Y.iloc[:3], Y.iloc[2:3])
            .build()
        )


def test_explicit_split_must_contain_exact_source_indexes() -> None:
    X = pd.DataFrame({'x': range(4)})
    Y = pd.DataFrame({'y': range(4)})
    X_test = pd.DataFrame({'x': [2, 3]}, index=(2, 9))
    Y_test = pd.DataFrame({'y': [2, 3]}, index=(2, 9))

    with pytest.raises(ValueError, match='must partition X indexes'):
        (
            RunConfigBuilder()
            .set_X(X)
            .set_Y(Y)
            .set_split(X.iloc[:2], X_test, Y.iloc[:2], Y_test)
            .build()
        )


def test_explicit_split_rows_must_equal_source_rows() -> None:
    X = pd.DataFrame({'x': range(4)})
    Y = pd.DataFrame({'y': range(4)})
    X_train = X.iloc[:2].copy()
    X_train.loc[0, 'x'] = 99

    with pytest.raises(ValueError, match='X_train rows must come from X'):
        (
            RunConfigBuilder()
            .set_X(X)
            .set_Y(Y)
            .set_split(X_train, X.iloc[2:], Y.iloc[:2], Y.iloc[2:])
            .build()
        )


def test_duplicate_evaluation_methods_are_rejected() -> None:
    with pytest.raises(ValueError, match='Duplicate evaluation method'):
        _builder().set_evaluation_methods(('pointwise', 'pointwise'))


def test_duplicate_generator_configs_are_rejected() -> None:
    config = NoiseGeneratorConfig((0.1,))

    with pytest.raises(ValueError, match='Duplicate variation generator'):
        (
            _builder()
            .set_variation_generator_configs((config, config))
            .build()
        )


def test_distinct_configs_for_the_same_study_bias_are_allowed() -> None:
    config = (
        _builder()
        .set_variation_generator_configs((
            NoiseGeneratorConfig((0.1,)),
            NoiseGeneratorConfig((0.2,)),
        ))
        .build()
    )

    assert tuple(study.study_bias for study in config.studies) == (
        StudyBias.DATA,
        StudyBias.DATA,
    )
    assert tuple(
        study.variation_generator.variation_labels
        for study in config.studies
    ) == (('std_0.1',), ('std_0.2',))


def test_dictionary_settings_allow_multiple_configs_for_one_bias() -> None:
    config = _builder().apply_run_settings({
        'variation_generator_configs': {
            'data': (
                {'standard_deviations': (0.1,)},
                {'standard_deviations': (0.2,)},
            ),
        },
    }).build()

    assert tuple(study.study_bias for study in config.studies) == (
        StudyBias.DATA,
        StudyBias.DATA,
    )
    assert tuple(
        study.variation_generator.variation_labels
        for study in config.studies
    ) == (('std_0.1',), ('std_0.2',))


def test_dictionary_settings_reject_duplicate_same_bias_configs() -> None:
    with pytest.raises(ValueError, match='Duplicate variation generator'):
        _builder().apply_run_settings({
            'variation_generator_configs': {
                'data': (
                    {'standard_deviations': (0.1,)},
                    {'standard_deviations': (0.1,)},
                ),
            },
        }).build()
