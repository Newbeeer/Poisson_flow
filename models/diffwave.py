import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
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
        self.projection1 = Linear(128, 512)
        self.projection2 = Linear(512, 512)

    def forward(self, z):
        x = get_positional_embedding(z, self.nf)
        x = self.projection1(x)
        x = silu(x)
        x = self.projection2(x)
        x = silu(x)
        return x


class SpectrogramUpsampler(nn.Module):
    def __init__(self, n_mels):
        super().__init__()
        self.conv1 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])
        self.conv2 = ConvTranspose2d(1, 1, [3, 32], stride=[1, 16], padding=[1, 8])

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x = self.conv1(x)
        x = F.leaky_relu(x, 0.4)
        x = self.conv2(x)
        x = F.leaky_relu(x, 0.4)
        x = torch.squeeze(x, 1)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, n_mels, residual_channels, dilation, uncond=False):
        '''
        :param n_mels: inplanes of conv1x1 for spectrogram conditional
        :param residual_channels: audio conv
        :param dilation: audio conv dilation
        :param uncond: disable spectrogram conditional
        '''
        super().__init__()
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        self.diffusion_projection = Linear(512, residual_channels)
        if not uncond:  # conditional model
            self.conditioner_projection = Conv1d(n_mels, 2 * residual_channels, 1)
        else:  # unconditional model
            self.conditioner_projection = None

        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step, conditioner=None):
        assert (conditioner is None and self.conditioner_projection is None) or \
               (conditioner is not None and self.conditioner_projection is not None)

        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        y = x + diffusion_step
        if self.conditioner_projection is None:  # using a unconditional model
            y = self.dilated_conv(y)
        else:
            conditioner = self.conditioner_projection(conditioner)
            y = self.dilated_conv(y) + conditioner

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
        if self.params.unconditional:  # use unconditional model
            self.spectrogram_upsampler = None
        else:
            self.spectrogram_upsampler = SpectrogramUpsampler(model_params.n_mels)

        self.residual_layers = nn.ModuleList([
            ResidualBlock(model_params.n_mels, model_params.residual_channels,
                          2 ** (i % model_params.dilation_cycle_length), uncond=model_params.unconditional)
            for i in range(model_params.residual_layers)
        ])
        self.skip_projection = Conv1d(model_params.residual_channels, model_params.residual_channels, 1)
        self.output_projection = Conv1d(model_params.residual_channels, 2, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, audio, diffusion_step, spectrogram=None):
        assert (spectrogram is None and self.spectrogram_upsampler is None) or \
               (spectrogram is not None and self.spectrogram_upsampler is not None)
        print(audio.shape)
        x = audio.unsqueeze(1)
        x = self.input_projection(x)
        x = F.relu(x)

        diffusion_step = self.diffusion_embedding(diffusion_step)
        if self.spectrogram_upsampler:  # use conditional model
            spectrogram = self.spectrogram_upsampler(spectrogram)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(x, diffusion_step, spectrogram)
            skip = skip_connection if skip is None else skip_connection + skip

        x = skip / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)
        x_out = x[:, :, 0]
        z_direction = x[:, :, 1]
        return x_out, z_direction
