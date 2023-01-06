#!/bin/bash
#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --gres=gpumem:20g
wandb offline
python3 main.py --conf 128_deep --mode eval --workdir pfgm_128_deep --ckpt 70000 --sampling
