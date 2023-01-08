#!/bin/bash
#SBATCH -n 16
#SBACTH --mem-per-cpu=4G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:16g
#SBATCH --time=00:10:00
wandb offline
python3 main.py --conf 128_deep --mode train --workdir pfgm_128_deep_v2 --DDP
