# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions and modules related to model definition.
"""

import torch
import methods
import numpy as np

_MODELS = {}


def register_model(cls=None, *, name=None):
  """A decorator for registering model classes."""

  def _register(cls):
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'Already registered model with name: {local_name}')
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register
  else:
    return _register(cls)


def get_model(name):
  return _MODELS[name]


def get_sigmas(config):
  """Get sigmas --- the set of noise levels for SMLD from config files.
  Args:
    config: A ConfigDict object parsed from the config file
  Returns:
    sigmas: a jax numpy arrary of noise levels
  """
  sigmas = np.exp(
    np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

  return sigmas


def create_model(config):
  """Create the model."""
  model_name = config.model.name
  model = get_model(model_name)(config)
  model = model.to(config.device)
  model = torch.nn.DataParallel(model)
  return model


def get_model_fn(model, train=False):
  """Create a function to give the output of the PFGM / score-based model.

  Args:
    model: The PFGM or score model.
    train: `True` for training and `False` for evaluation.

  Returns:
    A model function.
  """

  def model_fn(x, labels):
    """Compute the output of the PFGM / score-based model.

    Args:
      x: A mini-batch of input data.
      labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
        for different models.

    Returns:
      A tuple of (model output, new mutable states)
    """
    if not train:
      model.eval()
      return model(x, labels)
    else:
      model.train()
      return model(x, labels)

  return model_fn


def get_predict_fn(sde, model, train=False, continuous=True):
  """Wraps `predict_fn` so that the model output corresponds to a vector prediction

  Args:
    sde: An `methods.SDE` object that represents the forward SDE.
    model: A PFGM or score model.
    train: `True` for training and `False` for evaluation.
    continuous: If `True`, the score-based model is expected to directly take continuous time steps.

  Returns:
    A vector function.
  """
  model_fn = get_model_fn(model, train=train)

  if isinstance(sde, methods.VPSDE) or isinstance(sde, methods.subVPSDE):
    def predict_fn(x, t):
      # Scale neural network output by standard deviation and flip sign
      if continuous or isinstance(sde, methods.subVPSDE):
        # For VP-trained models, t=0 corresponds to the lowest noise level
        # The maximum value of time embedding is assumed to 999 for
        # continuously-trained models.
        labels = t * 999
        score = model_fn(x, labels)
        std = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VP-trained models, t=0 corresponds to the lowest noise level
        labels = t * (sde.N - 1)
        score = model_fn(x, labels)
        std = sde.sqrt_1m_alphas_cumprod.to(labels.device)[labels.long()]

      score = -score / std[:, None, None, None]
      return score

  elif isinstance(sde, methods.VESDE):
    def predict_fn(x, t):
      if continuous:
        # get sigmas by t
        labels = sde.marginal_prob(torch.zeros_like(x), t)[1]
      else:
        # For VE-trained models, t=0 corresponds to the highest noise level
        labels = sde.T - t
        labels *= sde.N - 1
        labels = torch.round(labels).long()

      score = model_fn(x, labels)
      return score

  elif isinstance(sde, methods.Poisson):
    # PFGM
    def predict_fn(x, z):
      # For PFGM, z is the augmented dimension
      normalized_poisson_field = model_fn(x, z)
      return normalized_poisson_field
  else:
    raise NotImplementedError(f"Method class {sde.__class__.__name__} not yet supported.")

  return predict_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))