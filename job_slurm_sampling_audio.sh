#!/bin/bash

#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g

python3 main.py --config ./configs/poisson/audio_ddpmpp.py --mode eval --workdir pfgm_sc09_v1 --config.eval.enable_sampling --config.eval.save_images --config.eval.batch_size 32
