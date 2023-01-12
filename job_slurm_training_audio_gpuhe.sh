#!/bin/bash

#SBATCH -n 12
#SBACTH --mem-per-cpu=8G
#SBATCH --gpus=2
#SBATCH --gres=gpumem:20g
#SBATCH --nodes=1
#SBATCH --time=100:00:00
wandb online
python3 main.py --conf diffwave --mode train --workdir diffwave_he --DDP
