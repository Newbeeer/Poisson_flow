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
from . import get_configs


@get_configs.register_config(name='diffwave')
def get_config():
    config = get_default_configs()

    # training
    training = config.training
    training.sde = 'poisson'
    training.continuous = True
    training.n_iters = 5000000
    training.batch_size = 16
    training.small_batch_size = 16
    training.gamma = 5
    training.M = 377
    training.restrict_M = True
    training.tau = 0.03
    training.snapshot_freq = 50000
    training.snapshot_freq_for_preemption = 10000
    training.model = 'diffwave'
    training.reduce_mean = True
    training.accum_iter = 4

    # data
    data = config.data
    data.image_height = 1
    data.image_width = 16000
    data.channels = 1
    data.spec = get_mels_128()
    data.category = 'audio'  # audio, mel
    data.centered = True # data is scaled from -1 to 1

    # sampling
    sampling = config.sampling
    sampling.method = 'ode'
    # sampling.ode_solver = 'rk45'
    sampling.ode_solver = 'forward_euler'
    # sampling.ode_solver = 'improved_euler'
    sampling.N = 100
    sampling.z_max = 550
    sampling.z_min = 1e-3
    sampling.upper_norm = 1000
    sampling.vs = False
    sampling.ckpt_number = 50000  # number of ckpt to load for sampling

    # model TODO adapt a 1d attention unet not a
    model = config.model
    model.name = 'diffwave'
    model.scale_by_sigma = False
    model.ema_rate = 0.995
    model.sigma_end = 0.01
    model.nf = 128
    # diffwave
    # from paper : We use a 36-layer DiffWave model
    # with kernel size 3 and dilation cycle [1, 2, · · · , 2048]. We set the number of diffusion steps T = 200
    # and residual channels C = 256. 
    model.residual_channels=128
    model.residual_layers=30
    model.dilation_cycle_length=10
    model.unconditional = True # conditioning on mel spec of audio
    
    # optim
    optim = config.optim
    optim.lr = 2e-5

    return config
