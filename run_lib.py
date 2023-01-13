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
import os
import time
import copy
import logging
import numpy as np
# Keep the import below for registering all model definitions
from models import ncsnpp_audio, stablediff
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import methods
import torch
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint
import wandb
import torch.distributed as dist
import gc

def train(gpu, args):
    """Runs the training pipeline.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    config = args.config 
    workdir = args.workdir
    if args.DDP:
        args.rank = gpu
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=args.world_size,
            rank=args.rank
        )
        torch.cuda.set_device(gpu)
        args.gpu = gpu
    
    if gpu == 0:
        gfile_stream = open(os.path.join(args.workdir, 'stdout.txt'), 'w')
        handler = logging.StreamHandler(gfile_stream)
        formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
        handler.setFormatter(formatter)
        logger = logging.getLogger()
        logger.addHandler(handler)
        logger.setLevel('INFO')
    
    # Create directories for experimental logs
    sample_dir = os.path.join(workdir, "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # Initialize model.
    net = mutils.create_model(args)
    ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
    optimizer, scheduler = losses.get_optimizer(config, net.parameters())
    state = dict(optimizer=optimizer, model=net, ema=ema, scheduler=scheduler, step=0)

    if gpu == 0:
        wandb.init(config=args.config)
        # Create checkpoints directory
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        # Intermediate checkpoints to resume training after pre-emption in cloud environments
        checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Resume training when intermediate checkpoints are detected
    # TODO make one for DDP
    if not args.DDP:
        state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds, _ = datasets.get_dataset(args)
    train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Setup methods
    if config.training.sde.lower() == 'poisson':
        sde = methods.Poisson(args=args)
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
        sampling_shape = (16, config.data.num_channels, config.data.image_height, config.data.image_width)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    num_train_steps = config.training.n_iters

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))
    for step in range(initial_step, num_train_steps + 1):
        # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
        if config.data.dataset == 'speech_commands' and config.data.category != 'tfmel':
            try:
                batch = next(train_iter).cuda(non_blocking=True)
                if len(batch) != config.training.batch_size:
                    continue
            except StopIteration:
                train_iter = iter(train_ds)
                batch = next(train_iter).cuda(non_blocking=True)
        else:
            batch = torch.from_numpy(next(train_iter)['image']._numpy()).to(config.device, non_blocking=True).float()
            # change to channel first only for original tf datasets but not for mel datasets
            if config.data.category != 'tfmel':
                batch = batch.permute(0, 3, 1, 2)
        
        batch = scaler(batch)
        # Execute one training step
        loss = train_step_fn(state, batch)
        if step % config.training.log_freq == 0 and gpu==0:
            wandb.log({"train_loss": loss.item()}, step=step // config.training.log_freq)
            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']
            wandb.log({"lr": lr}, step=step // config.training.log_freq)
            logging.info("step: %d, training_loss: %.5e, lr: %f" % (step, loss.item(), lr))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0 and gpu==0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0 and gpu==0:
            if config.data.dataset == 'speech_commands' and not config.data.category == 'tfmel':
                eval_batch = next(eval_iter).cuda()
            else:
                eval_batch = torch.from_numpy(next(eval_iter)['image']._numpy()).to(config.device, non_blocking=True).float()
                if not config.data.category == 'tfmel':
                    eval_batch = eval_batch.permute(0, 3, 1, 2)
            
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            wandb.log({"val_loss": eval_loss.item()}, step=step // config.training.eval_freq)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps and gpu==0:
            # Save the checkpoint.
            save_step = step
            save_checkpoint(os.path.join(workdir, "checkpoints", f'checkpoint_{save_step}.pth'), state)

            # # Generate and save samples
            if config.training.snapshot_sampling and not args.DDP:
                ema.store(net.parameters())
                ema.copy_to(net.parameters())
                sample, n = sampling_fn(net)
                ema.restore(net.parameters())
                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                os.makedirs(this_sample_dir, exist_ok=True)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = sample.permute(0, 2, 3, 1).cpu().numpy()

                np.save(os.path.join(this_sample_dir, "sample"), sample)
                save_image(image_grid, os.path.join(this_sample_dir, "sample.png"))
        
    wandb.finish()


def evaluate(args):
    """Evaluate trained models.

    Args:
      config: Configuration to use.
      workdir: Working directory for checkpoints.
      eval_folder: The subfolder for storing evaluation results. Default to
        "eval".
    """
    # Create directory to eval_folder
    config = args.config
    workdir = args.workdir
    checkpoint_dir = args.checkpoint_dir
    eval_folder = args.eval_folder
    eval_dir = os.path.join(workdir, eval_folder)
    os.makedirs(eval_dir, exist_ok=True)

    # Build data pipeline

    if not config.eval.save_images:
        train_ds, eval_ds, _ = datasets.get_dataset(args, evaluation=True)

    # Create data normalizer and its inverse
    scaler = datasets.get_data_scaler(config)
    inverse_scaler = datasets.get_data_inverse_scaler(config)

    # Initialize model
    net = mutils.create_model(args)
    optimizer, scheduler = losses.get_optimizer(config, net.parameters())
    ema = ExponentialMovingAverage(net.parameters(), decay=config.model.ema_rate)
    state = dict(optimizer=optimizer, model=net, ema=ema, scheduler=scheduler, step=0)

    # Setup methods
    if config.training.sde.lower() == 'poisson':
        sde = methods.Poisson(args=args)
        sampling_eps = config.sampling.z_min
        print("--- sampling eps:", sampling_eps)
    else:
        raise NotImplementedError(f"Method {config.training.sde} unknown.")

    # Create the one-step evaluation function when loss computation is enabled
    if config.eval.enable_loss:
        optimize_fn = losses.optimization_manager(config)
        reduce_mean = config.training.reduce_mean
        eval_step = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn, reduce_mean=reduce_mean,
                                       method_name=config.training.sde.lower())

    # Build the sampling function when sampling is enabled
    if config.eval.enable_sampling:
        sampling_shape = (config.eval.batch_size,
                          config.data.num_channels,
                          config.data.image_height, config.data.image_width)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

    # Wait if the target checkpoint doesn't exist yet
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    if config.training.sde == 'poisson':
        if config.sampling.ckpt_number > 0:
            ckpt_filename = os.path.join(checkpoint_dir, "checkpoint_{}.pth".format(config.sampling.ckpt_number))
            ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_{config.sampling.ckpt_number}.pth')
        else:
            raise ValueError("Please provide a ckpt_number!")

    if not os.path.exists(ckpt_filename):
        print(f"{ckpt_filename} does not exist! Loading from meta-checkpoint")
        ckpt_filename = os.path.join(checkpoint_dir, os.pardir, 'checkpoints-meta', 'checkpoint.pth')
        if not os.path.exists(ckpt_filename):
            print("No checkpoints-meta")
            return

    # Wait for 2 additional mins in case the file exists but is not ready for reading
    print("Loading from ", ckpt_path)
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

    ckpt = config.sampling.ckpt_number
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
        np.savez_compressed(os.path.join(eval_dir, f"ckpt_{ckpt}_loss.npz"), all_losses=all_losses,
                            mean_loss=all_losses.mean())

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size
        # Directory to save samples. Different for each host to avoid writing conflicts
        this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}/mels")
        os.makedirs(this_sample_dir, exist_ok=True)
        logging.info(f"Sampling for {num_sampling_rounds} rounds...")
        for r in range(num_sampling_rounds):
            logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
            samples, n = sampling_fn(net)
            logging.info(f"nfe: {n}")
            logging.info(f"sample shape: {samples.shape}")
            samples_torch = copy.deepcopy(samples)
            samples_torch = samples_torch.view(-1, config.data.num_channels, config.data.image_height, config.data.image_width)

            # sample the output matrices differently for pictures vs mel spectograms
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()
            logging.info("Saving images as raw mel specs.")

            samples = samples.reshape((-1, config.data.image_height, config.data.image_width, config.data.num_channels))

            # Write samples to disk or Google Cloud Storage
            np.savez_compressed(os.path.join(this_sample_dir, f"samples_{r}.npz"), samples=samples)

            #if config.eval.save_images:
                # Saving a few generated images for debugging / visualization
            #    image_grid = make_grid(samples_torch, nrow=int(np.sqrt(len(samples_torch))))
            #    save_image(image_grid, os.path.join(eval_dir, f'ode_images_{ckpt}.png'))
