"""Training and evaluation"""
# collections fix for python 10
import collections
import collections.abc

collections.Container = collections.abc.Container
collections.Mapping = collections.abc.Mapping
collections.MutableMapping = collections.abc.MutableMapping
collections.Iterable = collections.abc.Iterable
collections.MutableSet = collections.abc.MutableSet
collections.Callable = collections.abc.Callable

from configs import audio_ddpmpp_128_deep, audio_ddpmpp_64_deep, audio_diffwave_128, audio_sd_128, audio_sd_64, audio_ddpmpp_128_tiny
import trainer
import os
import argparse
from configs.get_configs import get_config
import torch.multiprocessing as mp
import torch
import wandb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True, help="config file")
    parser.add_argument("--workdir", required=True, help="working directory")
    parser.add_argument("--eval_folder", default="eval", help="folder for evaluation outputs")
    parser.add_argument("--test", action="store_true", help="test mode")
    parser.add_argument("--sampling", action="store_true", help="generate samples")
    parser.add_argument("--sampling_bs", default=8, help="batch size for sampling")
    parser.add_argument("--ckpt", default=None, help="checkpoint number for sampling")
    parser.add_argument("--DDP", action='store_true', default=False, help="use DDP (Distr Data Parallel)")
    parser.add_argument("--dist_file", default="ddp_sync_", help="file for DDP sync")
    parser.add_argument("--wandb", action='store_true', default=False, help="use wandb")
    args = parser.parse_args()

    args.config = get_config(args)
    args.wandb_group = args.workdir

    if args.sampling:
        args.config.eval.enable_sampling = True
        args.config.eval.save_images = True
        args.config.eval.batch_size = args.sampling_bs
        if args.ckpt is not None:
            args.config.sampling.ckpt_number = int(args.ckpt)

    ngpus = torch.cuda.device_count()
    print("Number of GPUs: {}".format(ngpus))
    # setup for DDP
    if args.DDP is True:
        if args.wandb:
            wandb.require("service")
            args.wandb_group += "_DDP"
        args.gpus = ngpus
        args.world_size = args.gpus
        job_id = os.environ["SLURM_JOBID"]
        args.dist_url = "file://{}.{}".format(os.path.realpath(args.dist_file), job_id)

    os.makedirs(args.workdir, exist_ok=True)
    if args.DDP:
        mp.spawn(trainer.train, nprocs=args.gpus, args=(args,))
    else:
        trainer.train(0, args)


if __name__ == "__main__":
    main()
