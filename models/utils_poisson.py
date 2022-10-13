import torch
import numpy as np


def forward_pz(sde, config, samples_batch, m):
    """Perturbing the augmented training data. See Algorithm 2 in PFGM paper.

    Args:
      sde: An `methods.SDE` object that represents the forward SDE.
      config: `True` for training loss and `False` for evaluation loss.
      samples_batch: A mini-batch of un-augmented training data
      m: A 1D torch tensor. The exponents of (1+\tau).

    Returns:
      Perturbed samples
    """
    tau = config.training.tau
    z = torch.randn((len(samples_batch), 1, 1, 1)).to(samples_batch.device) * config.model.sigma_end
    z = z.abs()

    # Confine the norms of perturbed data.
    # see Appendix B.1.1 of https://arxiv.org/abs/2209.11178
    if config.training.restrict_M:
        idx = (z < 0.005).squeeze()
        num = int(idx.int().sum())
        restrict_m = 250 if config.data.dataset == 'LSUN' else 200
        m[idx] = torch.rand((num,), device=samples_batch.device) * restrict_m

    data_dim = config.data.channels * config.data.image_size * config.data.image_size
    multiplier = (1+tau) ** m

    noise = torch.randn_like(samples_batch).reshape(len(samples_batch), -1) * config.model.sigma_end
    norm_m = torch.norm(noise, p=2, dim=1) * multiplier
    z_m = z.squeeze() * multiplier
    gaussian = torch.randn(len(samples_batch), data_dim).to(samples_batch.device)
    unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
    init_samples = unit_gaussian * norm_m[:, None]
    init_samples = init_samples.view_as(samples_batch)

    perturbed_samples = samples_batch + init_samples
    perturbed_samples_vec = torch.cat((perturbed_samples.reshape(len(samples_batch), -1),
                                       z_m[:, None]), dim=1)
    return perturbed_samples_vec
