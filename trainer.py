import os
import time
import copy
import logging
import numpy as np
# Keep the import below for registering all model definitions
from models import ncsnpp_audio, stablediff, diffwave
import losses
from evaluation import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
import methods
import torch
from torchvision.utils import make_grid, save_image
from utils.checkpoint import save_checkpoint, restore_checkpoint
import wandb
import torch.distributed as dist
import gc
import torchaudio


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
            init_method=args.dist_url,
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

    if args.wandb:
        wandb.init(config=args.config, group=args.wandb_group, name=f"{args.wandb_group}_{gpu}")

    if gpu == 0:
        # Create checkpoints directory
        checkpoint_dir = os.path.join(workdir, "checkpoints")
        # Intermediate checkpoints to resume training after pre-emption in cloud environments
        checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)

    # Resume training when intermediate checkpoints are detected
    # TODO make one for DDP
    if not args.DDP:
        state = restore_checkpoint(checkpoint_meta_dir, state, map_location=config.device)
    else:
        checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
        os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
        map_location = {'cuda:%d' % 0: 'cuda:%d' % args.rank}
        state = restore_checkpoint(checkpoint_meta_dir, state, map_location=map_location)

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
    train_step_fn = losses.get_step_fn(sde, train=True, optimize_fn=optimize_fn, reduce_mean=reduce_mean,
                                       method_name=method_name)
    eval_step_fn = losses.get_step_fn(sde, train=False, optimize_fn=optimize_fn, reduce_mean=reduce_mean,
                                      method_name=method_name)

    # Building sampling functions
    if config.training.snapshot_sampling:
        sampling_shape = (16, config.data.num_channels, config.data.image_height, config.data.image_width)
        sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps, net)

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
        if step % config.training.log_freq == 0:

            if scheduler is not None:
                lr = scheduler.get_last_lr()[0]
            else:
                lr = optimizer.param_groups[0]['lr']

            if args.wandb:
                wandb.log({"lr": lr}, step=step // config.training.log_freq)
                wandb.log({"train_loss": loss.item()}, step=step // config.training.log_freq)
            logging.info("gpu: %d, step: %d, training_loss: %.5e, lr: %f" % (gpu, step, loss.item(), lr))

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            if args.DDP:
                if gpu == 0:
                    save_checkpoint(checkpoint_meta_dir, state)
                dist.barrier()

            else:
                save_checkpoint(checkpoint_meta_dir, state)
            gc.collect()

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0 and gpu == 0:
            if config.data.dataset == 'speech_commands':
                eval_batch = next(eval_iter).cuda()
            eval_batch = scaler(eval_batch)
            eval_loss = eval_step_fn(state, eval_batch)
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            if args.wandb:
                wandb.log({"val_loss": eval_loss.item()}, step=step // config.training.eval_freq)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps and gpu == 0:
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
    if args.wandb:
        wandb.finish()
