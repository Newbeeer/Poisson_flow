#!/bin/bash
#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
wandb offline
python3 main.py --conf sd_128 --mode eval --workdir sd_128 --ckpt 50000 --sampling
