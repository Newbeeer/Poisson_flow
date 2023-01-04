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
"""Config file for reproducing the results of DDPM on bedrooms."""

from configs.default_audio_configs import get_default_configs


def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'poisson'
  training.continuous = True
  training.batch_size = 2 # 1024 for rtx 6000 and 64mels, small = bs/8
  training.small_batch_size = 2
  training.gamma = 5
  training.restrict_M = True
  training.tau = 0.03
  training.snapshot_freq = 10000
  training.model = 'attunet1d'
  training.reduce_mean = True
  training.amp = True
  training.accum_iter = 16 # gradient accumulations

  # data
  data = config.data
  data.channels = 1
  data.category = 'audio' # audio, mel, tfmel
  data.image_height = 1
  data.image_width = 16000
  data.centered = True

  # sampling
  sampling = config.sampling
  sampling.method = 'ode'
  #sampling.ode_solver = 'rk45'
  #sampling.ode_solver = 'forward_euler'
  sampling.ode_solver = 'improved_euler'
  sampling.N = 100
  sampling.z_max = 150 #TODO find good value
  sampling.z_min = 1e-3
  sampling.upper_norm = 5000
  sampling.vs = False
  sampling.ckpt_number = 75000 # number of ckpt to load for sampling

  # model
  model = config.model
  model.name = 'attunet1d' # ncsnpp_audio OR attunet1d
  model.scale_by_sigma = False
  model.sigma_end = 0.01
  model.ema_rate = 0.9999
  model.nf = 16 
  # optim
  optim = config.optim
  optim.lr = 2e-5
  

  return config
