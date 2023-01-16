#!/bin/bash
#SBATCH -n 6
#SBACTH --mem-per-cpu=4G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:20g
#SBATCH --time=4:00:00
python eval_mathias.py