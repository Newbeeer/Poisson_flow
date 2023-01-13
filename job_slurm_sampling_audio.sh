#!/bin/bash
#SBATCH -n 8
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
wandb offline
python3 main.py --conf diffwave --mode eval --workdir diffwave --ckpt 50000 --sampling
