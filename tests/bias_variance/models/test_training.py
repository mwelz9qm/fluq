import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from bias_variance.models.fnn import FnnArchitecture, FnnBuilder, FnnConfig
from bias_variance.models.training import TrainingConfig, Trainer


@pytest.mark.parametrize(
    ('settings', 'error'),
    (
        ({'optimizer': 'unknown'}, ValueError),
        ({'optimizer': 1}, TypeError),
        ({'learning_rate': True}, TypeError),
        ({'learning_rate': np.inf}, ValueError),
        ({'learning_rate': 0.0}, ValueError),
        ({'epochs': 1.5}, TypeError),
        ({'epochs': -1}, ValueError),
        ({'batch_size': True}, TypeError),
        ({'batch_size': 0}, ValueError),
        ({'loss': 'unknown'}, ValueError),
        ({'device': object()}, TypeError),
        ({'device': 'not-a-device'}, ValueError),
    ),
)
def test_training_config_rejects_invalid_settings(settings, error) -> None:
    with pytest.raises(error):
        TrainingConfig(**settings)


@pytest.mark.parametrize(
    ('settings', 'error'),
    (
        ({'input_size': True, 'output_size': 1}, TypeError),
        ({'input_size': 0, 'output_size': 1}, ValueError),
        ({'input_size': 1, 'output_size': 1.5}, TypeError),
        ({'input_size': 1, 'output_size': 0}, ValueError),
        ({'input_size': 1, 'output_size': 1, 'activation_factory': 1}, TypeError),
        ({'input_size': 1, 'output_size': 1, 'bias': 1}, TypeError),
    ),
)
def test_fnn_config_rejects_invalid_settings(settings, error) -> None:
    with pytest.raises(error):
        FnnConfig(**settings)


def test_fnn_builder_constructs_expected_layer_shapes() -> None:
    model = FnnBuilder(FnnConfig(3, 2)).build(FnnArchitecture((5, 4)))

    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [
        (3, 5),
        (5, 4),
        (4, 2),
    ]


def test_trainer_requires_a_model_builder() -> None:
    trainer = Trainer(TrainingConfig(epochs=0))

    with pytest.raises(RuntimeError, match='model builder'):
        trainer.train(
            FnnArchitecture((2,)),
            pd.DataFrame({'x': [1.0]}),
            pd.DataFrame({'y': [2.0]}),
            random_state=1,
        )


def test_cpu_training_is_reproducible_for_the_same_seed() -> None:
    X = pd.DataFrame({'x': [0.0, 1.0, 2.0, 3.0]})
    Y = pd.DataFrame({'y': [0.0, 2.0, 4.0, 6.0]})
    config = TrainingConfig(
        optimizer='sgd',
        learning_rate=0.01,
        epochs=2,
        batch_size=2,
        device='cpu',
    )

    def train():
        trainer = Trainer(config, FnnBuilder(FnnConfig(1, 1)))
        return trainer.train(
            FnnArchitecture((3,)),
            X,
            Y,
            random_state=25,
        )

    first = train()
    second = train()

    for first_parameter, second_parameter in zip(
        first.parameters(),
        second.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(first_parameter, second_parameter)
