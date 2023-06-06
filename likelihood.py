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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import torch
import numpy as np
from scipy import integrate
from models import utils as mutils
from models.utils import get_predict_fn

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn


def get_likelihood_fn(sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45', eps=1e-5):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `methods.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

  def drift_fn(model, x, t):
    """The drift function of the reverse-time SDE."""
    net_fn = mutils.get_predict_fn(sde, model, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(net_fn, probability_flow=True)
    return rsde.sde(x, t)[0]

  def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

  def likelihood_fn(model, data):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A score model.
      data: A PyTorch tensor.

    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    with torch.no_grad():
      shape = data.shape
      if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(data)
      elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

      def ode_func(t, x):
        sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))
        logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
        return np.concatenate([drift, logp_grad], axis=0)

      init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      zp = solution.y[:, -1]
      z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
      delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
      prior_logp = sde.prior_logp(z)
      bpd = -(prior_logp + delta_logp) / np.log(2)
      N = np.prod(shape[1:])
      bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      offset = 7. - inverse_scaler(-1.)
      bpd = bpd + offset
      return bpd, z, nfe

  return likelihood_fn


def to_flattened_numpy(x):
  """Flatten a torch tensor `x` and convert it to numpy."""
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
  return torch.from_numpy(x.reshape(shape))


def cat_z_scalar(x, z, img_size):
  """ concat x with a scalar z channel"""
  cat_z = torch.ones((len(x), 1, img_size, img_size)).cuda() * z
  return torch.cat((x, cat_z), dim=1)

def get_likelihood_fn_pfgm(sde, hutchinson_type='Rademacher',
                      rtol=1e-4, atol=1e-4, method='RK45', eps=1e-3):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `methods.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """
  data_dim = sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels
  constant = np.sqrt(data_dim)
  def drift_fn(model, x, z):

    if sde.config.sampling.vs:
      print(z)
    net_fn = get_predict_fn(sde, model, train=False, continuous=True)

    x_drift, z_drift = net_fn(x, torch.ones((len(x))).cuda() * z)
    x_drift = x_drift.view(len(x_drift), -1)

    z_exp = 5
    if z < z_exp and sde.config.training.gamma > 0:
      x_norm = x_drift.norm(p=2, dim=1) / constant
      v_norm = sde.config.training.gamma * x_norm / (1 - x_norm)
      v_norm = torch.sqrt(v_norm ** 2 + z ** 2)
      z_drift_ = - constant * torch.ones_like(z_drift) * z / (v_norm + sde.config.training.gamma)
      z_drift = z_drift_

    ### normalized to unit vector ###
    v = torch.cat([x_drift, z_drift[:, None]], dim=1)
    v_norm = v.norm(p=2, dim=1, keepdim=True)
    v = v / (v_norm + 1e-7)

    dt_dz = 1 / (v[:, -1] + 1e-5)
    dx_dt = v[:, :-1].view(x.shape[0], 3, x.shape[2], x.shape[3])

    drift = dx_dt * dt_dz.view(-1, *([1] * len(x.size()[1:])))
    return drift


  def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

  def likelihood_fn(model, data):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A pfgm model.
      data: A PyTorch tensor.

    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """

    shape = (len(data), sde.config.data.channels, sde.config.data.image_size, sde.config.data.image_size)

    with torch.no_grad():
      if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(data)
      elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

      def ode_func(z, x):
        sample = from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        drift = to_flattened_numpy(drift_fn(model, sample, z))
        logp_grad = to_flattened_numpy(div_fn(model, sample, z, epsilon))
        return np.concatenate([drift, logp_grad], axis=0)


      init = np.concatenate([to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      solution = integrate.solve_ivp(ode_func, (eps, sde.config.sampling.z_max), init, rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      zp = solution.y[:, -1]
      x = from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
      delta_logp = from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)

      N = np.prod(shape[1:])
      x_norm = x.view(len(x), -1).norm(p=2, dim=1)
      prior_logp = - torch.log(x_norm ** 2 + sde.config.sampling.z_max ** 2) * (N + 1) / 2. + np.log(2 * sde.config.sampling.z_max)
      prior_logp = (prior_logp).cuda()

      # https://mathworld.wolfram.com/Hypersphere.html for S_N(1)
      prior_log_theta = np.log(2) + N * 0.5 * np.log(np.pi)
      gamma_N = int(N * 0.5 - 1)
      for i in range(gamma_N, 1, -1):
        prior_log_theta -= np.log(int(i))
      prior_log_theta = - prior_log_theta

      print("delta:", delta_logp)
      print("prior logp:", prior_logp )
      print("prior theta:", prior_log_theta)
      bpd = -(prior_logp + delta_logp + prior_log_theta) / np.log(2)
      bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      offset = 7.
      bpd = bpd + offset
      return bpd, nfe

  return likelihood_fn