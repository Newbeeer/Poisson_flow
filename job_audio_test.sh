#!/bin/bash
#SBATCH -n 16
#SBATCH --gpus=1
#SBATCH --gres=gpumem:8g
#SBATCH --time=00:10:00
python3 main.py --config ./configs/poisson/audio_test.py --mode train --workdir pfgm_test
