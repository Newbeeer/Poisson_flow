#!/bin/bash
#SBATCH -n 24
#SBACTH --mem-per-cpu=2048
#SBATCH --gpus=4
#SBATCH --gres=gpumem:16g
#SBATCH --time=00:10:00
wandb offline
python3 main.py --conf sd_128 --mode train --workdir test_dir --DDP
