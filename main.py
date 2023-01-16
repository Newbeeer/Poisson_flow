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

import logging
import run_lib
import os
import argparse
from configs.get_configs import get_config
import torch.multiprocessing as mp
import torch 
import wandb 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    parser.add_argument("--eval_folder", default="eval")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sampling", action="store_true")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--DDP", action='store_true', default=False)
    parser.add_argument("--dist_file", default="ddp_sync_")
    parser.add_argument("--wandb", action='store_true', default=False)
    args = parser.parse_args()

    args.config = get_config(args)
    args.wandb_group = args.workdir
    

    if args.sampling:
        print("Parsing sampling args...")
        args.config.eval.enable_sampling = True
        args.config.eval.save_images = True
        args.config.eval.batch_size = 32
        if args.ckpt is not None:
            args.config.sampling.ckpt_number = int(args.ckpt)
        
    # setup for DDP
    if args.DDP is True:
        if args.wandb:
            wandb.require("service")
            args.wandb_group += "_DDP"
        args.gpus = torch.cuda.device_count()
        args.world_size = args.gpus
        

        job_id = os.environ["SLURM_JOBID"]
        args.dist_url = "file://{}.{}".format(os.path.realpath(args.dist_file), job_id)

    if args.mode == "train":
        print("START TRAINING")
        # Create the working directory
        os.makedirs(args.workdir, exist_ok=True)
        # Set logger so that it outputs to both console and file
        # Make logging work for both disk and Google Cloud Storage
        # Run the training pipeline
        if args.DDP:
            mp.spawn(run_lib.train, nprocs=args.gpus, args=(args,))
        else:
            run_lib.train(0, args)
            
    elif args.mode == "eval":
        print("START EVALUATION")
        # Run the evaluation pipeline
        run_lib.evaluate(args)
    else:
        raise ValueError(f"Mode {args.mode} not recognized.")


if __name__ == "__main__":
    main()
