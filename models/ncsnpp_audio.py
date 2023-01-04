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

from . import utils, layers, layerspp, normalization
import torch.nn as nn
import functools
import torch
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import functional as F

from .layersad import SkipBlock, FourierFeatures, SelfAttention1d, ResConvBlock
from .up_or_down_sampling import Downsample1d, Upsample1d
from .layerssd import SpatialTransformer


@utils.register_model(name='attunet1d')
class DiffusionAttnUnet1D(nn.Module):
    def __init__(
            self,
            config,
            in_channels=1,
            out_channels=2,
            depth=6,
            n_attn_layers=1,
            c_mults=[128, 128, 256, 256] + [512] * 2
    ):
        super().__init__()
        self.nf = config.model.nf

        attn_layer = depth - n_attn_layers - 1

        block = nn.Identity()

        conv_block = ResConvBlock

        for i in range(depth, 0, -1):
            c = c_mults[i - 1]
            if i > 1:
                c_prev = c_mults[i - 2]
                add_attn = i >= attn_layer and n_attn_layers > 0
                block = SkipBlock(
                    Downsample1d("cubic"),
                    conv_block(c_prev, c, c),
                    SpatialTransformer(channels=c, n_heads=c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SpatialTransformer(channels=c, n_heads=c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SpatialTransformer(channels=c, n_heads=c // 32) if add_attn else nn.Identity(),
                    block,
                    conv_block(c * 2 if i != depth else c, c, c),
                    SpatialTransformer(channels=c, n_heads=c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c),
                    SpatialTransformer(channels=c, n_heads=c // 32) if add_attn else nn.Identity(),
                    conv_block(c, c, c_prev),
                    SpatialTransformer(channels=c_prev, n_heads=c_prev // 32) if add_attn else nn.Identity(),
                    Upsample1d(kernel="cubic")
                )
            else:
                block = nn.Sequential(
                    conv_block(in_channels + self.nf, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, c),
                    block,
                    conv_block(c * 2, c, c),
                    conv_block(c, c, c),
                    conv_block(c, c, out_channels, is_last=True),
                )
        self.net = block

        with torch.no_grad():
            for param in self.net.parameters():
                param *= 0.5

    def forward(self, x, t):
        # timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        timestep_embed = layers.get_positional_embedding(t, self.nf)
        timestep_embed = expand_to_planes(timestep_embed, x.shape)

        # delete the height channel in case of sampling
        if x.ndim == 4:
            x = x.squeeze(-2)

        inputs = torch.cat([x, timestep_embed], dim=1)
        pred = self.net(inputs)

        # TODO add z value
        x_pred = pred[:, 0]  # the first channel (mono) is the prediction
        z_pred = pred[:, -1]  # the second channel is the augmented dimension
        z_pred = F.adaptive_avg_pool1d(z_pred, 1).squeeze()
        return x_pred, z_pred


def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f'input has {x.ndim} dims but target_dims is {target_dims}, which is less')
    return x[(...,) + (None,) * dims_to_append]


# expands the embedding along the time axis
def expand_to_planes(input, shape):
    return input[..., None].repeat([1, 1, shape[-1]])


###########################################################################################
########################################NCSNPP#############################################
###########################################################################################


ResnetBlockDDPM = layerspp.ResnetBlockDDPMpp
ResnetBlockBigGAN = layerspp.ResnetBlockBigGANpp
Combine = layerspp.Combine
conv3x3 = layerspp.conv3x3
conv1x1 = layerspp.conv1x1
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init


@utils.register_model(name='ncsnpp_audio')
class NCSNpp(nn.Module):
    """NCSN++ model"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.act = act = get_act(config)
        self.register_buffer('sigmas', torch.tensor(utils.get_sigmas(config)))

        self.nf = nf = config.model.nf
        ch_mult = config.model.ch_mult
        self.num_res_blocks = num_res_blocks = config.model.num_res_blocks
        self.attn_resolutions = attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        resamp_with_conv = config.model.resamp_with_conv
        self.num_resolutions = num_resolutions = len(ch_mult)
        self.all_resolutions = all_resolutions = [config.data.image_height // (2 ** i) for i in range(num_resolutions)]

        self.conditional = conditional = config.model.conditional  # noise-conditional
        fir = config.model.fir
        fir_kernel = config.model.fir_kernel
        self.skip_rescale = skip_rescale = config.model.skip_rescale
        self.resblock_type = resblock_type = config.model.resblock_type.lower()
        self.embedding_type = embedding_type = config.model.embedding_type.lower()
        init_scale = config.model.init_scale
        combine_method = config.model.progressive_combine.lower()
        combiner = functools.partial(Combine, method=combine_method)

        modules = []
        # z/noise_level embedding; only for continuous training
        # positional embedding means the embedding dim is the number of features
        embed_dim = nf

        # conditioning on the noise levels
        if conditional:
            modules.append(nn.Linear(embed_dim, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)
            modules.append(nn.Linear(nf * 4, nf * 4))
            modules[-1].weight.data = default_initializer()(modules[-1].weight.shape)
            nn.init.zeros_(modules[-1].bias)

        AttnBlock = functools.partial(layerspp.AttnBlockpp, init_scale=init_scale, skip_rescale=skip_rescale)

        Upsample = functools.partial(layerspp.Upsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        Downsample = functools.partial(layerspp.Downsample, with_conv=resamp_with_conv, fir=fir, fir_kernel=fir_kernel)

        if resblock_type == 'ddpm':
            ResnetBlock = functools.partial(ResnetBlockDDPM,
                                            act=act,
                                            dropout=dropout,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            zemb_dim=nf * 4)

        elif resblock_type == 'biggan':
            ResnetBlock = functools.partial(ResnetBlockBigGAN,
                                            act=act,
                                            dropout=dropout,
                                            fir=fir,
                                            fir_kernel=fir_kernel,
                                            init_scale=init_scale,
                                            skip_rescale=skip_rescale,
                                            zemb_dim=nf * 4)

        else:
            raise ValueError(f'resblock type {resblock_type} unrecognized.')

        # Downsampling block
        channels = config.data.num_channels

        modules.append(conv3x3(channels, nf))
        hs_c = [nf]

        in_ch = nf
        for i_level in range(num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(num_res_blocks):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch, out_ch=out_ch))
                in_ch = out_ch

                # add attention when the feature map gets the right size (16 default)
                if all_resolutions[i_level] in attn_resolutions:
                    modules.append(AttnBlock(channels=in_ch))
                hs_c.append(in_ch)

            if i_level != num_resolutions - 1:
                if resblock_type == 'ddpm':
                    modules.append(Downsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(down=True, in_ch=in_ch))

                hs_c.append(in_ch)

        in_ch = hs_c[-1]
        modules.append(ResnetBlock(in_ch=in_ch))
        modules.append(AttnBlock(channels=in_ch))
        modules.append(ResnetBlock(in_ch=in_ch))

        # Upsampling block
        for i_level in reversed(range(num_resolutions)):
            for i_block in range(num_res_blocks + 1):
                out_ch = nf * ch_mult[i_level]
                modules.append(ResnetBlock(in_ch=in_ch + hs_c.pop(),
                                           out_ch=out_ch))
                in_ch = out_ch

            if all_resolutions[i_level] in attn_resolutions:
                modules.append(AttnBlock(channels=in_ch))

            if i_level != 0:
                if resblock_type == 'ddpm':
                    modules.append(Upsample(in_ch=in_ch))
                else:
                    modules.append(ResnetBlock(in_ch=in_ch, up=True))

        assert not hs_c

        modules.append(nn.GroupNorm(num_groups=min(in_ch // 4, 32), num_channels=in_ch, eps=1e-6))

        # output an extra channel for PFGM z-dimension
        modules.append(conv3x3(in_ch, config.data.num_channels + 1, init_scale=init_scale))

        self.all_modules = nn.ModuleList(modules)

    def forward(self, x, cond):
        # x is the disturbed poisson field vector, cond is the disturbed z value
        modules = self.all_modules
        m_idx = 0

        zemb = layers.get_positional_embedding(cond, self.nf)

        if self.conditional:
            zemb = modules[m_idx](zemb)
            m_idx += 1
            zemb = modules[m_idx](self.act(zemb))
            m_idx += 1
        else:
            zemb = None

        if not self.config.data.centered:
            # If input data is in [0, 1]
            x = 2 * x - 1.

        hs = [modules[m_idx](x)]
        m_idx += 1
        for i_level in range(self.num_resolutions):
            # Residual blocks for this resolution
            for i_block in range(self.num_res_blocks):
                h = modules[m_idx](hs[-1], zemb)
                m_idx += 1
                if h.shape[-1] in self.attn_resolutions:
                    h = modules[m_idx](h)
                    m_idx += 1

                hs.append(h)

            if i_level != self.num_resolutions - 1:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](hs[-1])
                    m_idx += 1
                else:
                    h = modules[m_idx](hs[-1], zemb)
                    m_idx += 1

                hs.append(h)

        h = hs[-1]
        h = modules[m_idx](h, zemb)
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1
        h = modules[m_idx](h, zemb)
        m_idx += 1

        # Upsampling block
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = modules[m_idx](torch.cat([h, hs.pop()], dim=1), zemb)
                m_idx += 1

            if h.shape[-1] in self.attn_resolutions:
                h = modules[m_idx](h)
                m_idx += 1

            if i_level != 0:
                if self.resblock_type == 'ddpm':
                    h = modules[m_idx](h)
                    m_idx += 1
                else:
                    h = modules[m_idx](h, zemb)
                    m_idx += 1

        assert not hs

        h = self.act(modules[m_idx](h))
        m_idx += 1
        h = modules[m_idx](h)
        m_idx += 1

        assert m_idx == len(modules)

        # Predict the direction on the extra z dimension
        scalar = F.adaptive_avg_pool2d(h[:, -1], (1, 1))
        return h[:, :-1], scalar.reshape(len(scalar))
