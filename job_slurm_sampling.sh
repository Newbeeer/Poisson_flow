#!/bin/bash

#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
wandb offline
python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode eval --workdir pfgm_sc09_64_DEEP --config.eval.enable_sampling --config.eval.save_images --config.eval.batch_size 100
