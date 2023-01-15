#!/bin/bash
#SBATCH -n 6
#SBATCH --gpus=titan_rtx:1
#SBATCH --time=4:00:00
python eval.py