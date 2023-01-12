#!/bin/bash

#SBATCH -n 8
#SBACTH --mem-per-cpu=8G
#SBATCH --gpus=8
#SBATCH --gres=gpumem:20g
#SBATCH --nodes=1
#SBATCH --time=100:00:00
wandb online
python3 main.py --conf diffwave --mode train --workdir diffwave --DDP
