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

# Lint as: python3
"""Training NCSN++ on CIFAR-10 with PFGM."""
from configs.default_cifar10_configs import get_default_configs


def get_config():
  config = get_default_configs()
  # training
  training = config.training
  training.sde = 'poisson'
  training.continuous = True
  training.batch_size = 4096
  training.small_batch_size = 128
  training.gamma = 5
  training.restrict_M = True
  training.tau = 0.03
  training.snapshot_freq = 50000
  training.model = 'nscnpp'

  # data
  data = config.data
  data.channels = 3
  data.centered = True

  # sampling
  sampling = config.sampling
  sampling.method = 'ode'
  sampling.ode_solver = 'rk45'
  #sampling.ode_solver = 'forward_euler'
  #sampling.ode_solver = 'improved_euler'
  sampling.N = 100

  sampling.z_max = 40
  sampling.z_min = 1e-3
  sampling.upper_norm = 3000.
  sampling.vs = False

  # model
  model = config.model
  model.name = 'ncsnpp'
  model.scale_by_sigma = False
  model.ema_rate = 0.999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 2)
  model.num_res_blocks = 4
  model.attn_resolutions = (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = True
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive = 'none'
  model.progressive_input = 'residual'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.conv_size = 3
  model.sigma_end = 0.01
  model.z_max = 40
  model.z_min = 1e-2
  return config
