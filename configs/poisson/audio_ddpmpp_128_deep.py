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
import ml_collections
from configs.default_audio_configs import get_default_configs
from configs.mel_configs import get_mels_128

def get_config():
  config = get_default_configs()

  # training
  training = config.training
  training.sde = 'poisson'
  training.continuous = True
  training.batch_size = 128
  training.small_batch_size = training.batch_size // 8
  training.gamma = 5
  training.restrict_M = True
  training.tau = 0.03
  training.snapshot_freq = 10000
  training.model = 'ddpmpp'
  training.reduce_mean = True
  training.amp = False

  # data
  data = config.data
  data.tfrecords_path = 'sc09_128.tfrecords' # set 64 or 128, also set the data.spec field right
  data.spec = ml_collections.ConfigDict()
  data.spec = get_mels_128()
  data.image_height = data.spec.image_size
  data.image_width = data.spec.image_size

  data.channels = 1
  data.category = 'tfmel' # audio, mel, tfmel
  data.centered = False

  # sampling
  sampling = config.sampling
  sampling.method = 'ode'
  #sampling.ode_solver = 'rk45'
  #sampling.ode_solver = 'forward_euler'
  sampling.ode_solver = 'improved_euler'
  sampling.N = 100
  sampling.z_max = 80 #TODO find good value
  sampling.z_min = 1e-3
  sampling.upper_norm = 12000
  sampling.vs = False
  sampling.ckpt_number = 180000 # number of ckpt to load for sampling

  # model TODO adapt a 1d attention unet not a 
  model = config.model
  model.name = 'ncsnpp_audio'
  model.scale_by_sigma = False
  model.ema_rate = 0.9999
  model.normalization = 'GroupNorm'
  model.nonlinearity = 'swish'
  model.nf = 128
  model.ch_mult = (1, 2, 2, 4) # initial (1, 1, 2, 2, 4, 4)
  model.num_res_blocks = 8 # initial 2
  model.attn_resolutions = (16,) # initial (16,)
  model.resamp_with_conv = True
  model.conditional = True
  model.fir = False
  model.fir_kernel = [1, 3, 3, 1]
  model.skip_rescale = True
  model.resblock_type = 'biggan'
  model.progressive_combine = 'sum'
  model.attention_type = 'ddpm'
  model.init_scale = 0.
  model.fourier_scale = 16
  model.embedding_type = 'positional'
  model.conv_size = 3
  model.sigma_end = 0.01

  # optim
  optim = config.optim
  optim.lr = 2e-5

  return config