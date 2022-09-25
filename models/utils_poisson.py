import torch
import numpy as np


def forward_ode(sde, config, samples_batch, samples_full, T):
    # given a batch of training samples: B * sample_shape
    # and a full batch of training samples for calculating electric flux
    # and corresponding stopping time t: B
    # returen ode samples: N * sampler_shape

    eps = config.training.eps

    noise = torch.randn_like(samples_batch) * config.model.sigma_end
    perturbed_samples = samples_batch + noise


    z = torch.randn((len(samples_batch), 1, 1, 1)).to(samples_batch.device) * config.model.sigma_end
    if config.training.restrict_T:
        # restrict the perturbation steps when initial z is too small
        idx = (z < 0.005).squeeze()
        num = int(idx.int().sum())
        if config.data.dataset == 'CIFAR10':
            restrict_T = int(np.log(200./(np.sqrt(3072) * 0.01)) / np.log(1+config.training.eps)) + 1
        elif config.data.dataset == 'CELEBA':
            restrict_T = int(np.log(400./(np.sqrt(3072 * 4) * 0.01)) / np.log(1+config.training.eps)) + 1
        T[idx] = torch.randint(int(0), restrict_T, (num,), device=samples_batch.device)

    z = z.abs()
    z = z.repeat((1, 1, config.data.image_size, config.data.image_size))

    perturbed_samples = torch.cat((perturbed_samples, z), dim=1)
    perturbed_samples_vec = torch.cat((perturbed_samples[:, :-1].view(len(samples_batch), -1),
                                       perturbed_samples[:, -1].view(len(samples_batch), -1).mean(1).unsqueeze(1)),
                                      dim=1)
    real_samples_vec = torch.cat((samples_full.reshape(len(samples_full), -1),
                                  torch.zeros((len(samples_full), 1)).to(samples_full.device)),
                                  dim=1)
    real_samples_vec_batch = torch.cat((samples_batch.reshape(len(samples_batch), -1),
                                        torch.zeros((len(samples_batch), 1)).to(samples_batch.device)),
                                        dim=1)

    s_0 = torch.norm(perturbed_samples_vec - real_samples_vec_batch, p=2, dim=1, keepdim=True)
    data_dim = config.data.image_size * config.data.image_size * config.data.channels
    # forward ODE
    step = 0
    step_size_list = [(1+eps) ** i for i in range(sde.T)]

    with torch.no_grad():
        while torch.max(T) > step:
            idx = T > step

            gt_distance = torch.sum((perturbed_samples_vec[idx].unsqueeze(1) - real_samples_vec) ** 2,
                                    dim=[-1]).sqrt()
            distance = torch.min(gt_distance, dim=1, keepdim=True)[0] / (gt_distance + 1e-7)
            distance = distance ** (data_dim + 1)
            distance = distance[:, :, None]

            diff = - (perturbed_samples_vec[idx].unsqueeze(1) - real_samples_vec)

            gt_direction = torch.sum(distance * diff, dim=1)
            gt_direction = gt_direction.view(gt_direction.size(0), -1)
            gt_direction /= (gt_direction.norm(p=2, dim=1, keepdim=True) + 1e-7)

            perturbed_samples_vec[idx] = perturbed_samples_vec[idx] - eps * step_size_list[step] * s_0[idx] * gt_direction
            step += 1

    return perturbed_samples_vec, T

def forward_pz(sde, config, samples_batch, T):

    eps = config.training.eps
    z = torch.randn((len(samples_batch), 1, 1, 1)).to(samples_batch.device) * config.model.sigma_end
    if config.training.restrict_T:
        idx = (z < 0.005).squeeze()
        num = int(idx.int().sum())
        restrict_T = 250 if config.data.dataset == 'LSUN' else 200
        T[idx] = torch.rand((num,), device=samples_batch.device) * restrict_T
    z = z.abs()
    data_dim = config.data.channels * config.data.image_size * config.data.image_size
    multiplier = (1+eps) ** T

    noise = torch.randn_like(samples_batch).reshape(len(samples_batch), -1) * config.model.sigma_end
    norm_T = torch.norm(noise, p=2, dim=1) * multiplier
    z_T = z.squeeze() * multiplier
    gaussian = torch.randn(len(samples_batch), data_dim).to(samples_batch.device)
    unit_gaussian = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)
    init_samples = unit_gaussian * norm_T[:, None]
    init_samples = init_samples.view_as(samples_batch)

    perturbed_samples = samples_batch + init_samples
    perturbed_samples_vec = torch.cat((perturbed_samples.reshape(len(samples_batch), -1),
                                       z_T[:, None]), dim=1)
    return perturbed_samples_vec, T
