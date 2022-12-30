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
"""Training and evaluation for PFGM or score-based generative models. """
import gc
import io
import os
import time
import copy
import logging
import numpy as np
import tensorflow as tf
# Keep the import below for registering all model definitions
from models import ncsnv2, ncsnpp, ncsnpp_audio
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import likelihood
import methods
from absl import flags
import torch
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import wandb

torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True

FLAGS = flags.FLAGS
gpus = tf.config.list_physical_devices('GPU')

if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  # Initialize model.
  net = mutils.create_model(config)
  ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
  optimizer, scheduler = losses.get_optimizer(config, net.parameters())
  state = dict(optimizer=optimizer, model=net, ema=ema, scheduler=scheduler, step=0)

  # logging to weights and biases
  wandb.init(config=config, settings=wandb.Settings(start_method='thread'))
  wandb.watch(net, log_freq=1000)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)
  
  # Setup methods
  if config.training.sde.lower() == 'poisson':
    sde = methods.Poisson(config=config)
    sampling_eps = config.sampling.z_min
  else:
    raise NotImplementedError(f"Method {config.training.sde} unknown.")

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  reduce_mean = config.training.reduce_mean
  method_name = config.training.sde.lower()
  train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn, reduce_mean=reduce_mean, method_name=method_name)
  eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn, reduce_mean=reduce_mean, method_name=method_name)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (25, config.data.num_channels, config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))
  for step in range(initial_step, num_train_steps + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    if config.data.dataset == 'speech_commands' and config.data.category != 'tfmel':
      try:
        batch = next(train_iter).cuda()
        if len(batch) != config.training.batch_size:
          continue
      except StopIteration:
        train_iter = iter(train_ds)
        batch = next(train_iter).cuda()
    else:
      batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device).float()
      # change to channel first only for original tf datasets but not for mel datasets
      if config.data.category != 'tfmel':
        batch = batch.permute(0, 3, 1, 2)
    batch = scaler(batch)
    # Execute one training step
    loss = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      wandb.log({"train_loss": loss.item()}, step = step // config.training.log_freq)
      if scheduler is not None:
        lr = scheduler.get_last_lr()[0]
      else:
        lr = optimizer.param_groups[0]['lr']
      wandb.log({"lr": lr}, step = step // config.training.log_freq)
      logging.info("step: %d, training_loss: %.5e, lr: %f" % (step, loss.item(), lr))

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      if config.data.dataset == 'CELEBA':
        try:
          eval_batch = next(eval_iter)[0].cuda()
          if len(eval_batch) != config.eval.batch_size:
            continue
        except StopIteration:
          eval_iter = iter(eval_ds)
          eval_batch = next(eval_iter)[0].cuda()
      # pytorch dataloader case
      elif config.data.dataset == 'speech_commands' and not config.data.category == 'tfmel':
        try:
          eval_batch = next(eval_iter).cuda()
          if len(batch) != config.eval.batch_size:
            continue
        except StopIteration:
          eval_iter = iter(eval_ds)
          eval_batch = next(eval_iter).cuda()
      else:
        eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device).float()
        if not config.data.category == 'tfmel':
          eval_batch = eval_batch.permute(0, 3, 1, 2)
      eval_batch = scaler(eval_batch)
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      wandb.log({"val_loss": eval_loss.item()}, step = step // config.training.eval_freq)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

      # # Generate and save samples
      if config.training.snapshot_sampling:
        ema.store(net.parameters())
        ema.copy_to(net.parameters())
        sample, n = sampling_fn(net)
        ema.restore(net.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        # make img in case of img dataset only
        if not config.data.category in ['audio', 'mel', 'tfmel']:
          sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        else:
          sample = sample.permute(0, 2, 3, 1).cpu().numpy()
        
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.np"), "wb") as fout:
          np.save(fout, sample)

        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          save_image(image_grid, fout)


def evaluate(config,
             workdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  # Create directory to eval_folder

  # set random seed
  tf.random.set_seed(config.seed)

  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Build data pipeline

  if not config.eval.save_images:
    if config.data.dataset == 'CELEBA':
      train_ds, eval_ds = datasets_utils.celeba.get_celeba(config)
    else:
      train_ds, eval_ds, _ = datasets.get_dataset(config,
                                                  uniform_dequantization=config.data.uniform_dequantization,
                                                  evaluation=True)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  net = mutils.create_model(config)
  optimizer, scheduler = losses.get_optimizer(config, net.parameters())
  ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=net, ema=ema, scheduler=scheduler, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints")

  # Setup methods
  if config.training.sde.lower() == 'vpsde':
    sde = methods.VPSDE(config=config, beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.sampling.N)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'subvpsde':
    sde = methods.subVPSDE(config=config, beta_min=config.model.beta_min, beta_max=config.model.beta_max, N=config.model.num_scales)
    sampling_eps = 1e-3
  elif config.training.sde.lower() == 'vesde':
    sde = methods.VESDE(config=config, sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  elif config.training.sde.lower() == 'poisson':
    # PFGM
    sde = methods.Poisson(config=config)
    sampling_eps = config.sampling.z_min
    print("--- sampling eps:", sampling_eps)
  else:
    raise NotImplementedError(f"Method {config.training.sde} unknown.")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   method_name=config.training.sde.lower())


  # Build the likelihood computation function when likelihood is enabled
  if config.eval.enable_bpd:
    # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config,
                                                        uniform_dequantization=True, evaluation=True)
    if config.eval.bpd_dataset.lower() == 'train':
      ds_bpd = train_ds_bpd
      bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
      # Go over the dataset 5 times when computing likelihood on the test dataset
      ds_bpd = eval_ds_bpd
      bpd_num_repeats = 5
    else:
      raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")
    if config.training.sde.lower() == 'poisson':
      likelihood_fn = likelihood.get_likelihood_fn_pfgm(sde)
    else:
      likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled
  if config.eval.enable_sampling:
    sampling_shape = (config.eval.batch_size,
                      config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  begin_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))

  for ckpt in range(begin_ckpt, config.eval.end_ckpt + 1):

    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed_all(config.seed)

    if config.training.sde == 'poisson':
      if config.sampling.ckpt_number > 0:
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(config.sampling.ckpt_number))
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{config.sampling.ckpt_number}.pth')
      else:
        ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt * config.training.snapshot_freq))
        ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt * config.training.snapshot_freq}.pth')
      #ckpt_filename = os.path.join(checkpoint_dir, "checkpoint50000.pth")
      #ckpt_path = os.path.join(checkpoint_dir, 'checkpoint50000.pth')
    else:
      ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(ckpt * config.training.snapshot_freq))
      ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{ckpt * config.training.snapshot_freq}.pth')
    
    if not tf.io.gfile.exists(ckpt_filename):
      print(f"{ckpt_filename} does not exist")
      continue

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    print("loading from ", ckpt_path)
    try:
      state = restore_checkpoint(ckpt_path, state, device=config.device)
    except:
      logging.info("Loading Failed!")
      time.sleep(60)
      try:
        state = restore_checkpoint(ckpt_path, state, device=config.device)
      except:
        time.sleep(120)
        state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(net.parameters())
    # Compute the loss function on the full evaluation dataset if loss computation is enabled
    if config.eval.enable_loss:
      print("please don't set the config.eval.save_images flag, or the datasets wouldn't be loaded.")
      all_losses = []
      eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
      for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
          logging.info("Finished %dth step loss evaluation" % (i + 1))

      # Save loss values to disk or Google Cloud Storage
      all_losses = np.asarray(all_losses)
      with tf.io.gfile.GFile(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        for batch_id in range(len(ds_bpd)):
          batch = next(bpd_iter)
          eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
          eval_batch = eval_batch.permute(0, 3, 1, 2)
          eval_batch = scaler(eval_batch)
          bpd = likelihood_fn(net, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

    if config.eval.enable_interpolate:

      from scipy.spatial import geometric_slerp
      repeat = 6
      inter_num = 10

      sampling_shape = (inter_num,
                        config.data.num_channels,
                        config.data.image_size, config.data.image_size)
      sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)
      imgs = torch.empty((repeat * inter_num, config.data.num_channels, config.data.image_size, config.data.image_size))

      for i in range(repeat):
        N = np.prod(sampling_shape[1:])
        gaussian = torch.randn(2, N).cuda()
        unit_vec = gaussian / torch.norm(gaussian, p=2, dim=1, keepdim=True)

        t_vals = np.linspace(0, 1, inter_num)
        # spherical interpolations
        unit_vec = unit_vec.detach().cpu().numpy().astype(np.double)
        unit_vec /= np.sqrt(np.sum(unit_vec ** 2, axis=1, keepdims=True))
        result = geometric_slerp(unit_vec[0], unit_vec[1], t_vals)
        result = result * config.sampling.upper_norm
        result = torch.from_numpy(result).cuda()

        samples, n = sampling_fn(net, x=result)
        imgs[i * inter_num: (i + 1) * inter_num] = torch.clamp(samples, 0.0, 1.0).to('cpu')

      image_grid = make_grid(imgs, nrow=inter_num)
      save_image(image_grid, os.path.join(eval_dir, f'interpolation_{ckpt}.png'))

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = config.eval.num_samples // config.eval.batch_size + 1
      # Directory to save samples. Different for each host to avoid writing conflicts
      this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
      tf.io.gfile.makedirs(this_sample_dir)
      logging.info(f"Sampling for {num_sampling_rounds} rounds...")
      for r in range(num_sampling_rounds):
        logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
        samples, n = sampling_fn(net)
        logging.info(f"nfe: {n}")
        logging.info(f"sample shape: {samples.shape}")
        samples_torch = copy.deepcopy(samples)
        samples_torch = samples_torch.view(-1, config.data.num_channels, config.data.image_size, config.data.image_size)

        # sample the output matrices differently for pictures vs mel spectograms
        if config.data.category in ['audio', 'mel', 'tfmel']:
          samples = samples.permute(0, 2, 3, 1).cpu().numpy()
          logging.info("Saving images as raw mel specs.")
        else:
          samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        
        samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))

        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
          io_buffer = io.BytesIO()
          np.savez_compressed(io_buffer, samples=samples)
          fout.write(io_buffer.getvalue())

        if config.eval.save_images:
          # Saving a few generated images for debugging / visualization
          image_grid = make_grid(samples_torch, nrow=int(np.sqrt(len(samples_torch))))
          save_image(image_grid, os.path.join(eval_dir, f'ode_images_{ckpt}.png'))