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

from configs.default_audio_configs import get_default_configs, get_mels_128
import ml_collections
from . import get_configs

@get_configs.register_config(name='sd_128')
def get_config():
    config = get_default_configs()

    # training
    training = config.training
    training.sde = 'poisson'
    training.continuous = True
    training.batch_size = 16  # 1024 for rtx 6000 and 64mels, small = bs/8
    training.small_batch_size = 2
    training.gamma = 5
    training.restrict_M = True
    training.M = 293
    training.tau = 0.03
    training.snapshot_freq = 10000
    training.model = 'stablediff'
    training.reduce_mean = True
    training.accum_iter = 16  # gradient accumulations

    # data
    data = config.data
    data.spec = ml_collections.ConfigDict()
    data.spec = get_mels_128()
    data.image_height = data.spec.image_size
    data.image_width = data.spec.image_size
    data.mel_root = 'mel_datasets/sc09_128_v2'
    data.channels = 1
    data.category = 'mel'  # audio, mel
    data.centered = False

    # sampling
    config.eval.batch_size = 1
    sampling = config.sampling
    sampling.method = 'ode'
    # sampling.ode_solver = 'rk45'
    sampling.ode_solver = 'forward_euler'
    # sampling.ode_solver = 'improved_euler'
    sampling.N = 100
    sampling.z_max = 46  # TODO find good value
    sampling.z_min = 1e-3
    sampling.upper_norm = 7400
    sampling.vs = False
    sampling.ckpt_number = 70000  # number of ckpt to load for sampling

    # model
    model = config.model
    model.name = 'stablediff'
    model.scale_by_sigma = False
    model.ema_rate = 0.995
    model.nf = 128
    model.conv_size = 3
    model.sigma_end = 0.01

    # stable diffusion settings
    model.channels = 128  # channels of the features = nf value
    model.d_cond = 128  # like nf, size of conditional embeddings => we have none, it would be the CLIP embed size
    model.n_res_blocks = 2
    model.attention_levels = [3] # 128 -> 64 -> 32 -> 16
    model.channel_multipliers = [1, 2, 2, 2] # 128, 256, 256, 256
    model.n_heads = 1
    model.transformer_depth = 1

    # optim
    optim = config.optim
    optim.lr = 1e-4

    return config
