#!/bin/bash
#SBATCH -n 8
#SBACTH --mem-per-cpu=2048
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
#SBATCH --time=00:10:00
wandb offline
python3 main.py --config 128_deep --mode train --workdir test_dir --test
