import torch.nn as nn
import torch
import torch.nn.functional as F

# Function ported from StyleGAN2

def get_weight(module,
               shape,
               weight_var='weight',
               kernel_init=None):
    """Get/create weight tensor for a convolution or fully-connected layer."""

    return module.param(weight_var, kernel_init, shape)


def naive_upsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H, 1, W, 1))
    x = x.repeat(1, 1, 1, factor, 1, factor)
    return torch.reshape(x, (-1, C, H * factor, W * factor))


def naive_downsample_2d(x, factor=2):
    _N, C, H, W = x.shape
    x = torch.reshape(x, (-1, C, H // factor, factor, W // factor, factor))
    return torch.mean(x, dim=(3, 5))