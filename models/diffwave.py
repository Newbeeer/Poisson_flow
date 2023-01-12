import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from . import utils
from .layers import get_positional_embedding


Linear = nn.Linear
ConvTranspose2d = nn.ConvTranspose2d


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class DiffusionEmbedding(nn.Module):
    def __init__(self, nf):
        super().__init__()
        self.nf = nf
        self.projection1 = Linear(nf, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, z):
        x = get_positional_embedding(z, self.nf)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        '''
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)

        # unconditional model
        self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        # using a unconditional model
        y = self.dilated_conv(y)

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


@utils.register_model(name='diffwave')
class DiffWave(nn.Module):
    def __init__(self, config):
        super().__init__()
        model_params = config.model
        self.params = model_params
        self.input_projection = Conv1d(1, model_params.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(model_params.nf)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(model_params.residual_channels, 2 ** (i % model_params.dilation_cycle_length))
            for i in range(model_params.residual_layers)
        ])
        self.skip_projection = Conv1d(model_params.residual_channels, model_params.residual_channels, 1)
        self.output_projection = Conv1d(model_params.residual_channels, 2, 1)
        self.out_pooling = torch.nn.AdaptiveAvgPool1d(1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram=None):

        # delete one of the image dimensions if it was there
        if audio.ndim == 4:
            audio = audio.squeeze(-2)    
        x = self.input_projection(audio)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)

        x_out = x[:, 0, :]
        z_direction = self.out_pooling(x[:, 1, :]).squeeze()
        # add channel information to audio
        return x_out, z_direction
