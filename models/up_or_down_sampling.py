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


# layers from harmonai diffusion
_kernels = {
    'linear':
        [1 / 8, 3 / 8, 3 / 8, 1 / 8],
    'cubic':
        [-0.01171875, -0.03515625, 0.11328125, 0.43359375,
         0.43359375, 0.11328125, -0.03515625, -0.01171875],
    'lanczos3':
        [0.003689131001010537, 0.015056144446134567, -0.03399861603975296,
         -0.066637322306633, 0.13550527393817902, 0.44638532400131226,
         0.44638532400131226, 0.13550527393817902, -0.066637322306633,
         -0.03399861603975296, 0.015056144446134567, 0.003689131001010537]
}


class Downsample1d(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel])
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer('kernel', kernel_1d)

    def forward(self, x):
        x = F.pad(x, (self.pad,) * 2, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv1d(x, weight, stride=2)


class Upsample1d(nn.Module):
    def __init__(self, kernel='linear', pad_mode='reflect'):
        super().__init__()
        self.pad_mode = pad_mode
        kernel_1d = torch.tensor(_kernels[kernel]) * 2
        self.pad = kernel_1d.shape[0] // 2 - 1
        self.register_buffer('kernel', kernel_1d)

    def forward(self, x):
        x = F.pad(x, ((self.pad + 1) // 2,) * 2, self.pad_mode)
        weight = x.new_zeros([x.shape[1], x.shape[1], self.kernel.shape[0]])
        indices = torch.arange(x.shape[1], device=x.device)
        weight[indices, indices] = self.kernel.to(weight)
        return F.conv_transpose1d(x, weight, stride=2, padding=self.pad * 2 + 1)
