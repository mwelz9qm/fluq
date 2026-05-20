"""API skeleton for deep ensemble project

we will use a two class design, because pyMAISE wants a single Keras model to be returned from a HyperModels build() method. 
Since Deep Ensembles are a collection of models, we will use a two-class wrapper method proposed by Tyler:

DeepEnsemble: the outer HyperModel that builds all members and wraps them into a single object
DeepEnsembleSingle: the "single model" that pyMAISE sees, but secretly runs M models internally 
(pyMAISE thinks it will be talking to one Keras model, but it will actually be talking to M models)

Uncertainty decomposition:
total_var = epistemic_var + aleatoric_var

epistemic_var  = var(member_means) - shows disagreement between the models
aleatoric_var = mean(member_vars) - shows the inherent noise in the data, only if heteroscedastic=True, otherwise 0

Questions for the team:
1. I put heteroscedastic in __init__ because it changes the architecture and loss function, making it an architectural decision rather than a training decision. Does the team agree, or should it move to fit()?
2. Should visuals like plot_parity_with_uncertainty be included in DeepEnsembleSingle, or live in a separate module? Keeping them separate avoids a matplotlib dependency in the model class, but including them is closer to the sklearn style Matt suggested.
"""

from keras_tuner import HyperModel
import numpy as np
import tensorflow as tf
import pyMAISE.settings as settings

# class 1: Deep Ensemble
class DeepEnsemble(nnHyperModel):
    '''
    This is the class you instantiate when you want a
    Deep Ensemble. It extends nnHyperModel so it can fit into the existing pyMAISE training pipeline.

    when keras tuner calls the build() method it will Loop M times, 
    builds a fresh model for each member using super().build(hp),
    increments the random seed by 1 for each member to ensure different initializations,
    and then wraps the M models into a DeepEnsembleSingle object that pyMAISE can work with.

    PARAMETERS
    ----------
    parameters : dict, contains:
     "n_ensemble" : int  — number of ensemble members
     "base_seed"       : int  — starting random seed (member i gets seed = base_seed + i)
    "heteroscedastic" : bool — whether to model aleatoric uncertainty (if False, aleatoric_var is set to zero)

    EXAMPLE USAGE:
    >>> params = {"n_ensemble": 5, "base_seed": 42, "heteroscedastic": False}
    >>> ensemble = DeepEnsemble(parameters=params)
    >>> model = ensemble.build(hp)  returns a DeepEnsembleSingle
    '''
    def __init__(self, parameters: dict):
        super().__init__()
        self.parameters = parameters
 
    def build(self, hp) -> "DeepEnsembleSingle":
        """Called by keras-tuner during hyperparameter search or standard
        training. Each member is built with the same hyperparameters (hp)
        but a different random seed, allowing for different initializations.

        PARAMETERS
        ----------
        hp : HyperParameters object from keras-tuner, contains the hyperparameters to use for building each member model. 
             Each member will be built with the same hp values, but different random seeds.
        
        RETURNS
        -------
        DeepEnsembleSingle : a wrapper object that contains the M member models and 
        implements the Keras Model interface for pyMAISE compatibility.

        NOTES
        -----
        - The seed is incremented by 1 per member (base_seed, base_seed+1,
          base_seed+2, ...) to guarantee diversity while staying reproducible.
        - TODO: figure out the correct way to set TF and pyMAISE seeds per
          member. tf.random.set_seed()? numpy seed? Both?

        TODO (implementation)
        ---------------------
        - Read n_ensemble and base_seed from self.parameters
        - Loop n_ensemble times:
            - Set seed = base_seed + i
            - Set TF/numpy/pyMAISE seed (needs research)
            - Call super().build(hp) to get a fresh member model
            - Append to members list
        - Return DeepEnsembleSingle(members)
        """
        pass

# class 2: DeepEnsembleSingle
class DeepEnsembleSingle(tf.keras.Model):
    '''
    This is the class that pyMAISE will see as the "model" to train and evaluate, but it secretly contains M member models inside. 

    It implements the Keras Model interface by forwarding calls to the M member models and then aggregating their outputs.

    
    PARAMETERS
    ----------
    models : list of tf.keras.Model, The M independently initialized member models, built by
        DeepEnsemble.build(). Each has identical architecture but different random initialization.
    heteroscedastic : bool, default=False
        If True, each member outputs both mean and variance.
        If False, members output mean only and aleatoric_var will be zeros

   
    '''
    def __init__(self, models: list, heteroscedastic: bool = False):
        super().__init__()
        self.models = models
        self.heteroscedastic = heteroscedastic

    # Fitting method
    def fit(self, X: np.ndarray, y: np.ndarray, **fit_kwargs):
        """
        Train each ensemble member independently on the same data.
 
        Even though all members see the same (X, y), they produce different
        results because they were initialized with different random seeds.
        
        PARAMETERS
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Training features.
        y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
            Training targets.
        **fit_kwargs
            Forwarded to each member's fit() call.
            Common kwargs: epochs, batch_size, validation_split, callbacks.
 
        RETURNS
        -------
        histories : list
            List of Keras History objects, one per member. Useful for
            diagnosing whether members trained differently. 
        
        NOTES
        -----
        If heteroscedastic=True, each member should be compiled with NLL loss (configured in DeepEnsemble.build())

        TODO (implementation)
        ---------------------
        - Loop over self.models
        - Call model.fit(X, y, **fit_kwargs) for each
        - Collect and return history objects
        """
        pass

    # prediction method
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Returns the ensemble mean prediction.
 
        This is the standard Keras predict() override — returns a single
        array with no uncertainty. For uncertainty, use predict_uq().

        PARAMETERS
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features for prediction.
        
        RETURNS
        -------
        mean : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
            The mean prediction across all M members.


        TODO (implementation)
        ---------------------
        - Call predict_uq(X) and return only ["mean"]
        """
        pass

    def predict_uq(self, X: np.ndarray) -> dict:
        """
        Return ensemble mean plus full uncertainty decomposition.
 
        The core method of this whole project. Collects predictions from
        all M members, stacks them, and decomposes uncertainty into
        epistemic (between-model disagreement) and aleatoric (within-model
        noise, if heteroscedastic=True) components.
 
        PARAMETERS
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features.
        
        RETURNS
        -------
        results : dict
            {
                "mean": np.ndarray, shape (n_samples,) or (n_samples, n_targets)
                    The mean prediction across all M members.
                "epistemic_var": np.ndarray, shape (n_samples,) or (n_samples, n_targets)
                    The epistemic variance, calculated as the variance of member means.
                "aleatoric_var": np.ndarray, shape (n_samples,) or (n_samples, n_targets)
                    The aleatoric variance, calculated as the mean of member variances if heteroscedastic=True, otherwise zeros.
                "total_var": np.ndarray, shape (n_samples,) or (n_samples, n_targets)
                    The total variance, calculated as epistemic_var + aleatoric_var.
                "std": np.ndarray (n_samples, n_targets)
                    sqrt(total_var). Can be useful for CI calculations.
                "member_means": np.ndarray (n_models, n_samples, n_targets)
                    Raw per-member predictions. Useful for diagnosing member diversity.
                "member_vars": np.ndarray (n_models, n_samples, n_targets)
                    Raw per-member variances. All zeros if heteroscedastic=False.
            }
        TODO (implementation)
        ---------------------
        - Loop over self.models and collect member means and variances (if heteroscedastic)
        - Stack results: np.stack(mus, axis=0) to get shape (n_models, n_samples, n_targets)
        - Compute epistemic_var, aleatoric_var, total_var
        - Return results dict
        """
        pass

    #evaluation methods
    def score(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Compute evaluation metrics for the ensemble on the given data.
 
        This method can compute any relevant metrics, such as RMSE for regression
        or accuracy for classification, using the ensemble mean predictions. It
        can also compute uncertainty-aware metrics if desired.

        PARAMETERS
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features for evaluation.
        y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
            True targets for evaluation.
        
        RETURNS
        -------
        scores : dict
            {
                "R2": float - coefficient of determination
                "RMSE": float - root mean squared error
                "MAE": float - mean absolute error
                "95% CI": float - fraction of y_true within the 95% CI
                "mean_total_std"     : float
                "mean_epistemic_std" : float
                "mean_aleatoric_std" : float
            }

        TODO (implementation)
        ---------------------
        - Call predict_uq(X) to get mean and std
        - Compute R2, RMSE, MAE from mean vs y
        - Call ci_coverage(X, y) for the CI metric
        """
        pass

    def ci_coverage(self, X: np.ndarray, y: np.ndarray, z: float = 1.96) -> float:
        """
        Compute the fraction of true targets that fall within the ensemble's 95% confidence interval.

        This is a simple calibration metric to check if the predicted uncertainty is well-calibrated. 
        For a well-calibrated model, we would expect about 95% of the true targets to fall within the predicted mean ± 1.96 * std.

        PARAMETERS
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Input features for evaluation.
        y : np.ndarray, shape (n_samples,) or (n_samples, n_targets)
            True targets for evaluation.
        z : float, default=1.96
            The z-score corresponding to the desired confidence level (1.96 for 95% CI).

        RETURNS
        -------
        coverage : float
            The fraction of true targets that fall within the predicted confidence interval.
            Fraction of y_true within [mean - z*std, mean + z*std].
            A perfectly calibrated model returns ~0.95 for z=1.96.
            
        TODO (implementation)
        ---------------------
        - Call predict_uq(X) to get mean and std
        - Compute lower_bound = mean - z * std
        - Compute upper_bound = mean + z * std
        - Check how many y_true fall within [lower_bound, upper_bound] and return the fraction
        """
        pass

    #Print Methods
    def member_summary(self) -> None:
        """
        Print a summary of the each member individually, including their predictions and uncertainties.

        This method is useful for diagnosing the diversity of the ensemble. (different seeding for each member)
        It can print out the mean and variance predictions of each member, 
        allowing us to see how much they disagree (epistemic uncertainty) and how much inherent noise they predict (aleatoric uncertainty).

        TODO (implementation)
        ---------------------
        - Loop over self.models and print their individual predictions and variances
        - flag if all members are producing identical predictions (indicating a potential issue with seeding or model building)
        """
        pass
    def ensemble_summary(self) -> None:
        """
        Print a summary of the ensemble's overall predictions and uncertainty decomposition.

        This method can print out the ensemble mean prediction, epistemic variance, aleatoric variance, and total variance. 
        It can also print out the fraction of true targets that fall within the predicted confidence interval (calibration metric).

        Error if called before predict_uq() has been called at least once, since the summary relies on having predictions to summarize.

        TODO (implementation)
        ---------------------
        - Print ensemble size and variance component summaries
        - Warn if mean epistemic_var too low (members are identical)
        """
        pass



        
            