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

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils
from methods import VESDE, VPSDE
from models import utils_poisson

def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_loss_fn(sde, train, reduce_mean=True, continuous=True, eps=1e-5, method_name=None):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `methods.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `Truec` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    """Compute the loss function.

    Args:
      model: A PFGM or score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    if method_name == 'poisson':

      samples_full = batch
      # Get the mini-batch with size `training.small_batch_size`
      samples_batch = batch[: sde.config.training.small_batch_size]

      m = torch.rand((samples_batch.shape[0],), device=samples_batch.device) * sde.M
      # Perturb the (augmented) mini-batch data
      perturbed_samples_vec = utils_poisson.forward_pz(sde, sde.config, samples_batch, m)

      with torch.no_grad():
        real_samples_vec = torch.cat(
          (samples_full.reshape(len(samples_full), -1), torch.zeros((len(samples_full), 1)).to(samples_full.device)), dim=1)

        data_dim = sde.config.data.image_size * sde.config.data.image_size * sde.config.data.channels
        gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - real_samples_vec) ** 2,
                                dim=[-1]).sqrt()

        # For numerical stability, timing each row by its minimum value
        distance = torch.min(gt_distance, dim=1, keepdim=True)[0] / (gt_distance + 1e-7)
        distance = distance ** (data_dim + 1)
        distance = distance[:, :, None]
        # Normalize the coefficients (effectively multiply by c(\tilde{x}) in the paper)
        coeff = distance / (torch.sum(distance, dim=1, keepdim=True) + 1e-7)
        diff = - (perturbed_samples_vec.unsqueeze(1) - real_samples_vec)

        # Calculate empirical Poisson field (N+1 dimension in the augmented space)
        gt_direction = torch.sum(coeff * diff, dim=1)
        gt_direction = gt_direction.view(gt_direction.size(0), -1)

      gt_norm = gt_direction.norm(p=2, dim=1)
      # Normalizing the N+1-dimensional Poisson field
      gt_direction /= (gt_norm.view(-1, 1) + sde.config.training.gamma)
      gt_direction *= np.sqrt(data_dim)

      target = gt_direction
      net_fn = mutils.get_predict_fn(sde, model, train=train, continuous=continuous)

      perturbed_samples_x = perturbed_samples_vec[:, :-1].view_as(samples_batch)
      perturbed_samples_z = torch.clamp(perturbed_samples_vec[:, -1], 1e-10)
      net_x, net_z = net_fn(perturbed_samples_x, perturbed_samples_z)

      net_x = net_x.view(net_x.shape[0], -1)
      # Predicted N+1-dimensional Poisson field
      net = torch.cat([net_x, net_z[:, None]], dim=1)
      loss = ((net - target) ** 2)
      loss = reduce_op(loss.reshape(loss.shape[0], -1), dim=-1)
      loss = torch.mean(loss)

      return loss

    else:
      net_fn = mutils.get_predict_fn(sde, model, train=train, continuous=continuous)

      t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
      z = torch.randn_like(batch)
      mean, std = sde.marginal_prob(batch, t)
      perturbed_data = mean + std[:, None, None, None] * z
      score = net_fn(perturbed_data, t)
      losses = torch.square(score * std[:, None, None, None] + z)

      losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
      loss = torch.mean(losses)
      return loss

  return loss_fn



def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, method_name=None):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `methods.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.

  Returns:
    A one-step function for training or evaluation.
  """

  loss_fn = get_loss_fn(sde, train, reduce_mean=reduce_mean,
                            continuous=True, method_name=method_name)

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the PFGM or score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    if train:
      optimizer = state['optimizer']
      optimizer.zero_grad()
      loss = loss_fn(model, batch)
      loss.backward()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())
    else:
      with torch.no_grad():
        ema = state['ema']
        ema.store(model.parameters())
        ema.copy_to(model.parameters())
        loss = loss_fn(model, batch)
        ema.restore(model.parameters())

    return loss

  return step_fn
