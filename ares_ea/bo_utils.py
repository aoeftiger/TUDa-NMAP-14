from typing import Optional, Union

import gpytorch
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.exceptions.errors import UnsupportedError
from botorch.models import ModelListGP, SingleTaskGP
from botorch.models.gpytorch import BatchedMultiOutputGPyTorchModel
from botorch.models.model import Model
from botorch.models.transforms.input import InputTransform
from botorch.utils import t_batch_mode_transform
from gpytorch.models import ExactGP
from gym.spaces.utils import unflatten
from torch import Tensor
from torch.nn import Module


def sample_gp_prior_plot(
    model: Union[ExactGP, SingleTaskGP],
    test_X,
    n_samples: int = 5,
    ax=None,
    y_lim: Optional[tuple] = None,
):
    with gpytorch.settings.prior_mode(True):
        model.eval()
        preds = model(test_X)

    # Plotting
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    prior_mean = preds.mean.detach().numpy()
    prior_std = preds.stddev.detach().numpy()
    ax.plot(test_X, prior_mean, label="GP mean")
    ax.fill_between(
        test_X,
        prior_mean - 2 * prior_std,
        prior_mean + 2 * prior_std,
        alpha=0.2,
        label=r"2$\sigma$ confidence bound",
    )
    for i in range(n_samples):
        y_sample = preds.sample()
        ax.plot(test_X, y_sample, ls=":")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Samples drawn from the prior Gaussian process")
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    return ax


def sample_gp_posterior_plot(
    model: Union[ExactGP, SingleTaskGP],
    test_X,
    n_samples: int = 5,
    ax=None,
    y_lim: Optional[tuple] = None,
    show_true_f: bool = False,
    true_f_x=None,
    true_f_y=None,
):
    with gpytorch.settings.prior_mode(False):
        model.eval()
        preds = model(test_X)

    # Plotting
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    prior_mean = preds.mean.detach().numpy()
    prior_std = preds.stddev.detach().numpy()
    ax.plot(test_X, prior_mean, label="GP mean", lw=4)
    ax.fill_between(
        test_X,
        prior_mean - 2 * prior_std,
        prior_mean + 2 * prior_std,
        alpha=0.2,
        label=r"2$\sigma$ confidence bound",
    )
    for i in range(n_samples):
        y_sample = preds.sample()
        ax.plot(test_X, y_sample, ls=":")
    # Add observed data
    ax.plot(
        model.train_inputs[0].flatten(),
        model.train_targets,
        color="black",
        ls="",
        marker="*",
        markersize=12,
        label="Data points",
    )
    # Add true objective
    if show_true_f:
        ax.plot(true_f_x,true_f_y,":", label="True objective", color='orange', lw=4)
    ax.set_xlabel("X feature")
    ax.set_ylabel("Y target")
    ax.set_title("Gaussian Process Posterior")
    if y_lim is not None:
        ax.set_ylim(y_lim)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    return ax


def plot_acq_with_gp(model, train_x, train_y, acq, test_X,show_true_f: bool = False, true_f_x=None, true_f_y=None,):
    test_acq_values = acq(test_X.reshape(-1, 1, 1)).detach().numpy()
    preds_mean = model(test_X).mean.detach().numpy()
    preds_sigma = model(test_X).stddev.detach().numpy()
    plt.plot(test_X, preds_mean, label="GP mean")
    plt.fill_between(
        test_X,
        preds_mean - 2 * preds_sigma,
        preds_mean + 2 * preds_sigma,
        alpha=0.3,
        label=r"$2\sigma$ confidence",
    )
    plt.plot(train_x, train_y, ls="", marker="*", markersize=12, color="black", label="Data points")
    # Add true objective
    if show_true_f:
        plt.plot(true_f_x, true_f_y, ":", color='orange', label="True objective", lw=4)
    plt.plot(test_X, test_acq_values, color="red", label="Acq")
    x_next = test_X[np.argmax(test_acq_values)]
    y_next = test_acq_values.max()
    plt.plot(x_next, y_next, ls="", marker="o", markersize=12, color="magenta", label="max(acq)")
    plt.xlabel("X feature")
    plt.ylabel("Y target")
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")


def get_new_bound(env, current_action, stepsize):
    bounds = np.array([env.action_space.low, env.action_space.high])
    bounds = stepsize * bounds + current_action
    bounds = np.clip(bounds, env.action_space.low, env.action_space.high)
    return bounds


def scale_action(env, observation, filter_action=None):
    """Scale the observed magnet settings to proper action values"""
    unflattened = (
        unflatten(env.unwrapped.observation_space, observation)
        if not isinstance(observation, dict)
        else observation
    )
    magnet_values = unflattened["magnets"]
    action_values = []
    if filter_action is None:
        filter_action = [0, 1, 2, 3, 4]

    for i, act in enumerate(filter_action):
        scaled_low = env.action_space.low[i]
        scaled_high = env.action_space.high[i]
        low = env.unwrapped.action_space.low[act]
        high = env.unwrapped.action_space.high[act]
        action = scaled_low + (scaled_high - scaled_low) * (
            (magnet_values[act] - low) / (high - low)
        )
        action_values.append(action)
    return action_values


# ---

"""
Taken from Ryan's implementation: https://github.com/ChristopherMayes/Xopt/blob/main/xopt/generators/bayesian/custom_botorch/proximal.py

A wrapper around AcquisitionFunctions to add proximal weighting of the
acquisition function.
"""


class ProximalAcquisitionFunction(AcquisitionFunction):
    """A wrapper around AcquisitionFunctions to add proximal weighting of the
    acquisition function. First a SoftPlus transform is applied to the
    acquisition function to ensure that it is positive. Then the acquisition function is
    weighted via a squared exponential centered at the last training point,
    with varying lengthscales corresponding to `proximal_weights`. Can only be used
    with acquisition functions based on single batch models.
    Small values of `proximal_weights` corresponds to strong biasing towards recently
    observed points, which smoothes optimization with a small potential decrese in
    convergence rate.
    Example:
        >>> model = SingleTaskGP(train_X, train_Y)
        >>> EI = ExpectedImprovement(model, best_f=0.0)
        >>> proximal_weights = torch.ones(d)
        >>> EI_proximal = ProximalAcquisitionFunction(EI, proximal_weights)
        >>> eip = EI_proximal(test_X)
    """

    def __init__(
        self,
        acq_function: AcquisitionFunction,
        proximal_weights: Tensor,
        transformed_weighting: bool = True,
        beta: float = 1.0,
    ) -> None:
        r"""Derived Acquisition Function weighted by proximity to recently
        observed point.
        Args:
            acq_function: The base acquisition function, operating on input tensors
                of feature dimension `d`.
            proximal_weights: A `d` dim tensor used to bias locality
                along each axis.
            transformed_weighting: If True, the proximal weights are applied in
                the transformed input space given by
                `acq_function.model.input_transform` (if available), otherwise
                proximal weights are applied in real input space.
            beta: Beta factor passed to softplus transform.
        """
        Module.__init__(self)

        self.acq_func = acq_function
        model = self.acq_func.model

        if hasattr(acq_function, "X_pending"):
            if acq_function.X_pending is not None:
                raise UnsupportedError(
                    "Proximal acquisition function requires `X_pending` to be None."
                )
            self.X_pending = acq_function.X_pending

        self.register_buffer("proximal_weights", proximal_weights)
        self.register_buffer(
            "transformed_weighting", torch.tensor(transformed_weighting)
        )
        self.register_buffer("beta", torch.tensor(beta))
        _validate_model(model, proximal_weights)

    @t_batch_mode_transform(expected_q=1, assert_output_shape=False)
    def forward(self, X: Tensor) -> Tensor:
        r"""Evaluate base acquisition function with proximal weighting.
        Args:
            X: Input tensor of feature dimension `d` .
        Returns:
            Base acquisition function evaluated on tensor `X` multiplied by proximal
            weighting.
        """
        model = self.acq_func.model

        train_inputs = model.train_inputs[0]

        # if the model is ModelListGP then get the first model
        if isinstance(model, ModelListGP):
            train_inputs = train_inputs[0]
            model = model.models[0]

        # if the model has more than one output get the first copy of training inputs
        if isinstance(model, BatchedMultiOutputGPyTorchModel) and model.num_outputs > 1:
            train_inputs = train_inputs[0]

        input_transform = _get_input_transform(model)

        last_X = train_inputs[-1].reshape(1, 1, -1)

        # if transformed_weighting, transform X to calculate diff
        # (proximal weighting in transformed space)
        # otherwise,un-transform the last observed point to real space
        # (proximal weighting in real space)
        if input_transform is not None:
            if self.transformed_weighting:
                # transformed space weighting
                diff = input_transform.transform(X) - last_X
            else:
                # real space weighting
                diff = X - input_transform.untransform(last_X)

        else:
            # no transformation
            diff = X - last_X

        M = torch.linalg.norm(diff / self.proximal_weights, dim=-1) ** 2
        proximal_acq_weight = torch.exp(-0.5 * M)
        return (
            torch.nn.functional.softplus(self.acq_func(X), beta=self.beta)
            * proximal_acq_weight.flatten()
        )


def _validate_model(model: Model, proximal_weights: Tensor) -> None:
    r"""Validate model
    Perform vaidation checks on model used in base acquisition function to make sure
    it is compatible with proximal weighting.
    Args:
        model: Model associated with base acquisition function to be validated.
        proximal_weights: A `d` dim tensor used to bias locality
                along each axis.
    """

    # check model for train_inputs and single batch
    if not hasattr(model, "train_inputs"):
        raise UnsupportedError("Acquisition function model must have `train_inputs`.")

    # get train inputs for each type of possible model
    if isinstance(model, ModelListGP):
        # ModelListGP models
        # check to make sure that the training inputs and input transformers for each
        # model match and are reversible
        train_inputs = model.train_inputs[0][0]
        input_transform = _get_input_transform(model.models[0])

        for i in range(len(model.train_inputs)):
            if not torch.equal(train_inputs, model.train_inputs[i][0]):
                raise UnsupportedError(
                    "Proximal acquisition function does not support unequal "
                    "training inputs"
                )

            if not input_transform == _get_input_transform(model.models[i]):
                raise UnsupportedError(
                    "Proximal acquisition function does not support non-identical "
                    "input transforms"
                )

    else:
        # any non-ModelListGP model
        train_inputs = model.train_inputs[0]

    # check to make sure that the model is single t-batch (q-batches are allowed)
    if model.batch_shape != torch.Size([]) and train_inputs.shape[1] != 1:
        raise UnsupportedError(
            "Proximal acquisition function requires a single batch model"
        )

    # check to make sure that weights match the training data shape
    if (
        len(proximal_weights.shape) != 1
        or proximal_weights.shape[0] != train_inputs.shape[-1]
    ):
        raise ValueError(
            "`proximal_weights` must be a one dimensional tensor with "
            "same feature dimension as model."
        )


def _get_input_transform(model: Model) -> Optional[InputTransform]:
    """get input transform if defined"""
    try:
        return model.input_transform
    except AttributeError:
        return None
