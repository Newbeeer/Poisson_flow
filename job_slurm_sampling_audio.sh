#!/bin/bash

#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
wandb offline
python3 main.py --config ./configs/poisson/audio_ddpmpp_128_deep.py --mode eval --workdir pfgm_sc09_128 --config.eval.enable_sampling --config.eval.save_images --config.eval.batch_size 32
