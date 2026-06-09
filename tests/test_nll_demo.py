"""
NLL mode demo/test for DeepEnsembleHyperModel.

Tests:
  1. gaussian_nll_loss — unit test on the loss function directly
  2. NLL model output shape — nnHyperModel adds the variance head correctly
  3. Full ensemble pipeline — build, train, predict_with_uncertainty returns aleatoric_var
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import keras_tuner as kt
import tensorflow as tf

import pyMAISE as mai
from deep_ensembles._deep_ensemble import DeepEnsembleHyperModel
from deep_ensembles._nn_hypermodel import gaussian_nll_loss, nnHyperModel


# ---------------------------------------------------------------------------
# 1. Unit test: gaussian_nll_loss
# ---------------------------------------------------------------------------
def test_gaussian_nll_loss():
    print("--- Test 1: gaussian_nll_loss ---")
    batch, n_targets = 16, 2

    y_true = tf.random.normal((batch, n_targets))
    # y_pred = [means | variances], variances must be positive
    y_pred = tf.concat([
        tf.random.normal((batch, n_targets)),
        tf.abs(tf.random.normal((batch, n_targets))) + 0.1,
    ], axis=-1)

    loss = gaussian_nll_loss(y_true, y_pred)

    assert loss.shape == (), f"Expected scalar loss, got shape {loss.shape}"
    assert np.isfinite(float(loss)), "NLL loss should be finite"

    # Sanity: higher variance prediction should penalise overconfident wrong answer
    y_true_fixed = tf.ones((batch, 1))
    pred_wrong_confident = tf.concat([tf.zeros((batch, 1)), 1e-6 * tf.ones((batch, 1))], axis=-1)
    pred_uncertain       = tf.concat([tf.zeros((batch, 1)), tf.ones((batch, 1))], axis=-1)
    loss_conf  = float(gaussian_nll_loss(y_true_fixed, pred_wrong_confident))
    loss_uncert = float(gaussian_nll_loss(y_true_fixed, pred_uncertain))
    assert loss_conf > loss_uncert, (
        "Overconfident wrong prediction should have higher NLL than uncertain one"
    )

    print(f"  loss value       : {float(loss):.4f}")
    print(f"  overconfident NLL: {loss_conf:.4f}  >  uncertain NLL: {loss_uncert:.4f}")
    print("  PASS\n")


# ---------------------------------------------------------------------------
# 2. Output shape: nnHyperModel should double output units for variance head
# ---------------------------------------------------------------------------
def test_nll_model_output_shape():
    print("--- Test 2: NLL model output shape ---")
    mai.settings.init("regression", verbosity=0, random_state=42)

    n_features, n_targets = 4, 2
    parameters = {
        "structural_params": {
            "Dense_1": {"units": 16, "activation": "relu",   "num_layers": 1},
            "Dense_2": {"units": n_targets, "activation": "linear", "num_layers": 1},
        },
        "optimizer": "Adam",
        "Adam": {"learning_rate": 0.001},
        "compile_params": {"loss_func": "nll"},
        "fitting_params": {"epochs": 1},
    }

    hypermodel = nnHyperModel(parameters=parameters, input_shape=(n_features,), name="shape_test")
    model = hypermodel.build(kt.HyperParameters())

    out_units = model.output_shape[-1]
    expected  = n_targets * 2
    assert out_units == expected, (
        f"Expected {expected} output units (mean + variance), got {out_units}"
    )
    # Loss should be gaussian_nll_loss
    assert model.loss is gaussian_nll_loss, "Model loss should be gaussian_nll_loss"

    print(f"  output units : {out_units}  (expected {expected})")
    print(f"  loss function: {model.loss.__name__}")
    print("  PASS\n")


# ---------------------------------------------------------------------------
# 3. Full pipeline: build → train → predict_with_uncertainty
# ---------------------------------------------------------------------------
def test_nll_ensemble_pipeline():
    print("--- Test 3: NLL ensemble pipeline (build → train → predict) ---")
    mai.settings.init("regression", verbosity=0, random_state=42)

    # Synthetic heteroscedastic regression data
    rng = np.random.default_rng(42)
    n_train, n_test, n_features, n_targets = 160, 40, 4, 1

    xtrain = rng.standard_normal((n_train, n_features)).astype(np.float32)
    xtest  = rng.standard_normal((n_test,  n_features)).astype(np.float32)
    # noise std grows with |x[:,0]| so aleatoric variance should be learnable
    noise  = (0.1 + 0.5 * np.abs(xtrain[:, 0])) * rng.standard_normal(n_train)
    w      = rng.standard_normal(n_features).astype(np.float32)
    ytrain = (xtrain @ w + noise).reshape(-1, n_targets).astype(np.float32)

    parameters = {
        "structural_params": {
            "Dense_1": {"units": 16, "activation": "relu",   "num_layers": 1},
            "Dense_2": {"units": n_targets, "activation": "linear", "num_layers": 1},
        },
        "optimizer": "Adam",
        "Adam": {"learning_rate": 0.01},
        "compile_params": {"loss_func": "nll"},
        "fitting_params": {"epochs": 10, "batch_size": 32},
    }

    de_hypermodel = DeepEnsembleHyperModel(
        parameters=parameters,
        input_shape=(n_features,),
        name="nll_ensemble_test",
        num_models=3,
    )

    ensemble = de_hypermodel.build_ensemble(kt.HyperParameters())

    # --- structural checks ---
    assert ensemble.heteroscedastic, "Wrapper must have heteroscedastic=True for NLL mode"
    print(f"  heteroscedastic flag: True  PASS")

    for i, m in enumerate(ensemble.ensemble_models):
        out = m.output_shape[-1]
        assert out == n_targets * 2, (
            f"Member {i}: expected {n_targets * 2} output units, got {out}"
        )
    print(f"  All {len(ensemble.ensemble_models)} members have output_shape[-1] == {n_targets * 2}  PASS")

    # --- training ---
    print("  Training 10 epochs (verbose=0)...")
    ensemble.fit(x=xtrain, y=ytrain, epochs=10, batch_size=32, verbose=0)

    # --- uncertainty decomposition ---
    results     = ensemble.predict_with_uncertainty(xtest, verbose=0)
    mean        = np.array(results["mean"])
    epis_var    = np.array(results["epistemic_var"])
    alea_var    = results["aleatoric_var"]

    assert alea_var is not None, "NLL mode must return non-None aleatoric_var"
    alea_var = np.array(alea_var)

    assert mean.shape     == (n_test, n_targets), f"mean shape wrong: {mean.shape}"
    assert epis_var.shape == (n_test, n_targets), f"epistemic_var shape wrong: {epis_var.shape}"
    assert alea_var.shape == (n_test, n_targets), f"aleatoric_var shape wrong: {alea_var.shape}"
    assert np.all(epis_var >= 0), "Epistemic variance must be >= 0"
    assert np.all(alea_var >= 0), "Aleatoric variance must be >= 0 (softplus activation)"

    print(f"  mean shape        : {mean.shape}")
    print(f"  epistemic_var     : shape={epis_var.shape}, mean={np.mean(epis_var):.6f}  PASS")
    print(f"  aleatoric_var     : shape={alea_var.shape}, mean={np.mean(alea_var):.6f}  PASS")
    print(f"  all variances >= 0  PASS")
    print("  PASS\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print("=" * 55)
    print("NLL Mode Tests")
    print("=" * 55)

    test_gaussian_nll_loss()
    test_nll_model_output_shape()
    test_nll_ensemble_pipeline()

    print("=" * 55)
    print("All NLL tests passed!")
    print("=" * 55)


if __name__ == "__main__":
    main()
