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

from configs.default_audio_configs import get_default_configs, get_mels_64
import ml_collections
from . import get_configs

@get_configs.register_config(name='64_deep')
def get_config():
    config = get_default_configs()

    # training
    training = config.training
    training.sde = 'poisson'
    training.continuous = True
    training.batch_size = 128  # 1024 for rtx 6000 and 64mels, small = bs/8
    training.small_batch_size = training.batch_size // 8
    training.gamma = 5
    training.restrict_M = True
    training.M = 268
    training.tau = 0.03
    training.snapshot_freq = 10000
    training.model = 'ddpmpp'
    training.reduce_mean = True
    training.accum_iter = 1024 // training.batch_size

    # data
    data = config.data
    data.spec = ml_collections.ConfigDict()
    data.spec = get_mels_64()
    data.image_height = data.spec.image_size
    data.image_width = data.spec.image_size
    data.mel_root = 'mel_datasets/sc09_64'

    data.channels = 1
    data.category = 'mel'  # audio, mel
    data.centered = False

    # sampling
    sampling = config.sampling
    sampling.method = 'ode'
    #sampling.ode_solver = 'rk45'
    # sampling.ode_solver = 'forward_euler'
    sampling.ode_solver = 'improved_euler'
    sampling.N = 100
    sampling.z_max = 22
    sampling.z_min = 1e-3
    sampling.upper_norm = 1800
    sampling.vs = False
    sampling.ckpt_number = 150000  # number of ckpt to load for sampling

    # model TODO adapt a 1d attention unet not a
    model = config.model
    model.name = 'ncsnpp_audio'
    model.scale_by_sigma = False
    model.ema_rate = 0.995
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 1, 2, 2, 4, 4)  # initial (1, 1, 2, 2, 4, 4)
    model.num_res_blocks = 6  # initial 2
    model.attn_resolutions = (16,)  # initial (16,)
    model.resamp_with_conv = True
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
    optim.lr = 1e-4

    return config
