#!/bin/bash
#SBATCH -n 4
#SBACTH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpumem:10g
#SBATCH --time=00:10:00
export WANDB_MODE=disabled
python3 main.py --conf dw_128 --mode train --workdir test_dir --test
