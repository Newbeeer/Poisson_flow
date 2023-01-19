# https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion
# /stable_diffusion/model/unet.py
import math
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from . import utils, layers
from .layerssd import SpatialTransformer


@utils.register_model(name='stablediff')
class UNetModel(nn.Module):
    """
    ## U-Net model
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # set paramters from config
        self.d_cond = config.model.nf
        channels = config.model.nf
        self.channels = channels

        # self.channels = channels = config.model.channels
        # self.d_cond = config.model.channels
        in_channels = config.data.num_channels
        out_channels = in_channels + 1
        n_res_blocks = config.model.n_res_blocks
        attention_levels = config.model.attention_levels
        channel_multipliers = config.model.channel_multipliers
        n_heads = config.model.n_heads
        tf_layers = config.model.transformer_depth

        # Number of levels
        levels = len(channel_multipliers)
        # Size time embeddings
        d_time_emb = config.model.nf * 4

        self.time_embed = nn.Sequential(
            nn.Linear(config.model.nf, d_time_emb),
            nn.SiLU(),
            nn.Linear(d_time_emb, d_time_emb),
        )

        # Input half of the U-Net
        self.input_blocks = nn.ModuleList()
        # Initial $3 \times 3$ convolution that maps the input to `channels`.
        # The blocks are wrapped in `TimestepEmbedSequential` module because
        # different modules have different forward function signatures;
        # for example, convolution only accepts the feature map and
        # residual blocks accept the feature map and time embedding.
        # `TimestepEmbedSequential` calls them accordingly.
        self.input_blocks.append(TimestepEmbedSequential(nn.Conv2d(in_channels, channels, 3, padding=1)))
        # Number of channels at each block in the input half of U-Net
        input_block_channels = [channels]
        # Number of channels at each level
        channels_list = [channels * m for m in channel_multipliers]
        # Prepare levels
        for i in range(levels):
            # Add the residual blocks and attentions
            for _ in range(n_res_blocks):
                # Residual block maps from previous number of channels to the number of
                # channels in the current level
                layers_block = [ResBlock(channels, d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers_block.append(SpatialTransformer(channels, n_heads, tf_layers))
                # Add them to the input half of the U-Net and keep track of the number of channels of
                # its output
                self.input_blocks.append(TimestepEmbedSequential(*layers_block))
                input_block_channels.append(channels)
            # Down sample at all levels except last
            if i != levels - 1:
                self.input_blocks.append(TimestepEmbedSequential(DownSample(channels)))
                input_block_channels.append(channels)

        # The middle of the U-Net
        self.middle_block = TimestepEmbedSequential(
            ResBlock(channels, d_time_emb),
            SpatialTransformer(channels, n_heads, tf_layers),
            ResBlock(channels, d_time_emb),
        )

        # Second half of the U-Net
        self.output_blocks = nn.ModuleList([])
        # Prepare levels in reverse order
        for i in reversed(range(levels)):
            # Add the residual blocks and attentions
            for j in range(n_res_blocks + 1):
                # Residual block maps from previous number of channels plus the
                # skip connections from the input half of U-Net to the number of
                # channels in the current level.
                layers_block = [
                    ResBlock(channels + input_block_channels.pop(), d_time_emb, out_channels=channels_list[i])]
                channels = channels_list[i]
                # Add transformer
                if i in attention_levels:
                    layers_block.append(SpatialTransformer(channels, n_heads, tf_layers))
                # Up-sample at every level after last residual block
                # except the last one.
                # Note that we are iterating in reverse; i.e. `i == 0` is the last.
                if i != 0 and j == n_res_blocks:
                    layers_block.append(UpSample(channels))
                # Add to the output half of the U-Net
                self.output_blocks.append(TimestepEmbedSequential(*layers_block))

        # Final normalization and $3 \times 3$ convolution
        self.out = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

    def time_step_embedding(self, time_steps: torch.Tensor, max_period: int = 10000):
        """
        ## Create sinusoidal time step embeddings
        :param time_steps: are the time steps of shape `[batch_size]`
        :param max_period: controls the minimum frequency of the embeddings.
        """
        # $\frac{c}{2}$; half the channels are sin and the other half is cos,
        half = self.channels // 2
        # $\frac{1}{10000^{\frac{2i}{c}}}$
        frequencies = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
                                ).to(device=time_steps.device)
        # $\frac{t}{10000^{\frac{2i}{c}}}$
        args = time_steps[:, None].float() * frequencies[None]
        # $\cos\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$ and $\sin\Bigg(\frac{t}{10000^{\frac{2i}{c}}}\Bigg)$
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x, time_steps):
        """
        :param x: is the input feature map of shape `[batch_size, channels, width, height]`
        :param time_steps: are the time steps of shape `[batch_size]`
        :param cond: conditioning of shape `[batch_size, n_cond, d_cond]`
        """

        # x is the disturbed image of right size and cond is the "timestep"
        # To store the input half outputs for skip connections
        x_input_block = []
        # get conditional embedding, this is always the same => cond is none for us
        t_emb = layers.get_positional_embedding(time_steps, self.d_cond)
        # embedding of the timesteps from BS to self.channels and then to self.channels*4 with linear leayers => same
        # as first 2 layers in ncsnpp
        # t_emb = self.time_step_embedding(time_steps)
        t_emb = self.time_embed(t_emb)
        cond = None

        # Input half of the U-Net
        for module in self.input_blocks:
            x = module(x, t_emb, cond)
            x_input_block.append(x)
        # Middle of the U-Net
        x = self.middle_block(x, t_emb, cond)
        # Output half of the U-Net
        for module in self.output_blocks:
            x = th.cat([x, x_input_block.pop()], dim=1)
            x = module(x, t_emb, cond)

        x = self.out(x)

        z_pred = F.adaptive_avg_pool2d(x[:, -1], (1, 1))
        return x[:, :-1], z_pred.reshape(len(z_pred))


class TimestepEmbedSequential(nn.Sequential):
    """
    ### Sequential block for modules with different inputs
    This sequential module can compose of different modules suck as `ResBlock`,
    `nn.Conv` and `SpatialTransformer` and calls them with the matching signatures
    """

    def forward(self, x, t_emb, cond=None):
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x)
            else:
                x = layer(x)
        return x


class UpSample(nn.Module):
    """
    ### Up-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution mapping
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Up-sample by a factor of $2$
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        # Apply convolution
        return self.conv(x)


class DownSample(nn.Module):
    """
    ## Down-sampling layer
    """

    def __init__(self, channels: int):
        """
        :param channels: is the number of channels
        """
        super().__init__()
        # $3 \times 3$ convolution with stride length of $2$ to down-sample by a factor of $2$
        self.op = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor):
        """
        :param x: is the input feature map with shape `[batch_size, channels, height, width]`
        """
        # Apply convolution
        return self.op(x)


class ResBlock(nn.Module):
    """
    ## ResNet Block
    """

    def __init__(self, channels: int, d_t_emb: int, *, out_channels=None):
        """
        :param channels: the number of input channels
        :param d_t_emb: the size of timestep embeddings
        :param out_channels: is the number of out channels. defaults to `channels.
        """
        super().__init__()
        # `out_channels` not specified
        if out_channels is None:
            out_channels = channels

        # First normalization and convolution
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv2d(channels, out_channels, 3, padding=1),
        )

        # Time step embeddings
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(d_t_emb, out_channels),
        )
        # Final convolution layer
        self.out_layers = nn.Sequential(
            normalization(out_channels),
            nn.SiLU(),
            nn.Dropout(0.),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )

        # `channels` to `out_channels` mapping layer for residual connection
        if out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(channels, out_channels, 1)

    def forward(self, x: torch.Tensor, z_emb: torch.Tensor):
        # Initial convolution
        h = self.in_layers(x)
        # Time step embeddings => from d_t_emb to out_channels
        z_emb = self.emb_layers(z_emb).type(h.dtype)
        # Add time step embeddings
        h = h + z_emb[:, :, None, None]
        # Final convolution
        h = self.out_layers(h)
        # Add skip connection
        return self.skip_connection(x) + h


class GroupNorm32(nn.GroupNorm):
    """
    ### Group normalization with float32 casting
    """

    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)


def normalization(channels):
    """
    ### Group normalization
    This is a helper function, with fixed number of groups..
    """
    return GroupNorm32(32, channels)
