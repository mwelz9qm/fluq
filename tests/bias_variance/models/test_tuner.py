import pytest

from bias_variance.models.training import TrainingConfig
from bias_variance.models.tuner import Tuner, TunerConfig


def test_tuner_uses_default_config_when_none_is_provided():
    tuner = Tuner()

    assert isinstance(tuner.config, TunerConfig)


def test_tuner_uses_provided_config():
    config = TunerConfig(n_trials=1)

    tuner = Tuner(config)

    assert tuner.config is config


def test_tuner_config_accepts_valid_values():
    config = TunerConfig(
        n_trials=1,
        direction="minimize",
        metric="rmse",
        optimizer_choices=("adam",),
        learning_rate_range=(1e-4, 1e-3),
        loss_choices=("mse",),
        epoch_choices=(1,),
        batch_size_choices=(2,),
    )

    assert config.n_trials == 1
    assert config.direction == "minimize"
    assert config.metric == "rmse"
    assert config.optimizer_choices == ("adam",)
    assert config.learning_rate_range == (1e-4, 1e-3)
    assert config.loss_choices == ("mse",)
    assert config.epoch_choices == (1,)
    assert config.batch_size_choices == (2,)


@pytest.mark.parametrize("value", (0, -1))
def test_tuner_config_rejects_non_positive_n_trials(value):
    with pytest.raises(ValueError, match="n_trials must be greater than 0"):
        TunerConfig(n_trials=value)


@pytest.mark.parametrize("value", (True, 1.5, "10"))
def test_tuner_config_rejects_non_integer_n_trials(value):
    with pytest.raises(TypeError, match="n_trials must be an integer"):
        TunerConfig(n_trials=value)


@pytest.mark.parametrize("value", ("wrong", "MINIMIZE"))
def test_tuner_config_rejects_invalid_direction_value(value):
    with pytest.raises(ValueError, match="direction must be either"):
        TunerConfig(direction=value)


@pytest.mark.parametrize("value", (1, None))
def test_tuner_config_rejects_non_string_direction(value):
    with pytest.raises(TypeError, match="direction must be a string"):
        TunerConfig(direction=value)


@pytest.mark.parametrize("value", ("accuracy", "loss"))
def test_tuner_config_rejects_invalid_metric_value(value):
    with pytest.raises(ValueError, match="metric must be one of"):
        TunerConfig(metric=value)


@pytest.mark.parametrize("value", (1, None))
def test_tuner_config_rejects_non_string_metric(value):
    with pytest.raises(TypeError, match="metric must be a string"):
        TunerConfig(metric=value)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (["adam"], TypeError, "optimizer_choices must be a tuple"),
        ((), ValueError, "optimizer_choices must not be empty"),
        ((1,), TypeError, "optimizer_choices must contain only strings"),
        (("rmsprop",), ValueError, "optimizer_choices contains unsupported"),
    ],
)
def test_tuner_config_rejects_invalid_optimizer_choices(
    value,
    error,
    message,
):
    with pytest.raises(error, match=message):
        TunerConfig(optimizer_choices=value)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        ([1e-4, 1e-3], TypeError, "learning_rate_range must be a tuple"),
        ((1e-4,), ValueError, "exactly two bounds"),
        ((1, 2), TypeError, "bounds must be floats"),
        ((0.0, 1e-3), ValueError, "lower bound must be greater than 0"),
        ((1e-3, 1e-3), ValueError, "lower bound must be less"),
        ((1e-2, 1e-3), ValueError, "lower bound must be less"),
    ],
)
def test_tuner_config_rejects_invalid_learning_rate_range(
    value,
    error,
    message,
):
    with pytest.raises(error, match=message):
        TunerConfig(learning_rate_range=value)


@pytest.mark.parametrize(
    ("value", "error", "message"),
    [
        (["mse"], TypeError, "loss_choices must be a tuple"),
        ((), ValueError, "loss_choices must not be empty"),
        ((1,), TypeError, "loss_choices must contain only strings"),
        (("huber",), ValueError, "loss_choices contains unsupported"),
    ],
)
def test_tuner_config_rejects_invalid_loss_choices(value, error, message):
    with pytest.raises(error, match=message):
        TunerConfig(loss_choices=value)


@pytest.mark.parametrize(
    ("name", "value", "error", "message"),
    [
        ("epoch_choices", [1], TypeError, "epoch_choices must be a tuple"),
        ("epoch_choices", (), ValueError, "epoch_choices must not be empty"),
        (
            "epoch_choices",
            (True,),
            TypeError,
            "epoch_choices must contain only integers",
        ),
        (
            "epoch_choices",
            (0,),
            ValueError,
            "epoch_choices must contain only positive integers",
        ),
        (
            "batch_size_choices",
            [2],
            TypeError,
            "batch_size_choices must be a tuple",
        ),
        (
            "batch_size_choices",
            (),
            ValueError,
            "batch_size_choices must not be empty",
        ),
        (
            "batch_size_choices",
            (True,),
            TypeError,
            "batch_size_choices must contain only integers",
        ),
        (
            "batch_size_choices",
            (0,),
            ValueError,
            "batch_size_choices must contain only positive integers",
        ),
    ],
)
def test_tuner_config_rejects_invalid_positive_integer_choices(
    name,
    value,
    error,
    message,
):
    with pytest.raises(error, match=message):
        TunerConfig(**{name: value})


def test_build_training_config_from_params_returns_training_config():
    tuner = Tuner()
    params = {
        "optimizer": "adam",
        "learning_rate": 1e-3,
        "loss": "mse",
        "epochs": 1,
        "batch_size": 2,
    }

    config = tuner._build_training_config_from_params(params)

    assert isinstance(config, TrainingConfig)
    assert config.optimizer == "adam"
    assert config.learning_rate == 1e-3
    assert config.loss == "mse"
    assert config.epochs == 1
    assert config.batch_size == 2