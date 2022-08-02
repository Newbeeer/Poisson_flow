import torch
import tensorflow as tf
import os
import logging


def restore_checkpoint(ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)


def cal_scores(sde, sigmas, perturbed_samples, samples_full):

  perturbed_samples_vec = perturbed_samples.reshape((len(perturbed_samples), -1))
  samples_full_vec = samples_full.reshape((len(samples_full), -1))

  with torch.no_grad():
    if sde.config.training.sde == 'vesde':
      gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - samples_full_vec) ** 2,
                              dim=[-1])
      gt_distance = - gt_distance / (2 * sigmas.unsqueeze(1) ** 2)
      distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
      distance = torch.exp(distance)
      distance = distance[:, :, None]
      distance = distance / (torch.sum(distance, dim=1, keepdim=True) + 1e-7)
      diff = - (perturbed_samples_vec.unsqueeze(1) - samples_full_vec)

      gt_direction = torch.sum(distance * diff, dim=1)
      gt_direction /= sigmas.unsqueeze(1)

    elif sde.config.training.sde == 'vpsde':
      mean_coeff = torch.sqrt(1 - sigmas ** 2)
      samples_full_vec_mean = samples_full_vec.unsqueeze(0) * mean_coeff[:, None, None]
      gt_distance = torch.sum((perturbed_samples_vec.unsqueeze(1) - samples_full_vec_mean) ** 2,
                              dim=[-1])
      gt_distance = - gt_distance / (2 * sigmas.unsqueeze(1) ** 2)
      distance = - torch.max(gt_distance, dim=1, keepdim=True)[0] + gt_distance
      distance = torch.exp(distance)
      distance = distance[:, :, None]
      distance = distance / (torch.sum(distance, dim=1, keepdim=True) + 1e-7)
      diff = - (perturbed_samples_vec.unsqueeze(1) - samples_full_vec_mean)

      gt_direction = torch.sum(distance * diff, dim=1)
      gt_direction /= sigmas.unsqueeze(1)
    else:
      raise NotImplementedError
    return gt_direction