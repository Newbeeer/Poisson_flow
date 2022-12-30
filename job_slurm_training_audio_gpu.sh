#!/bin/bash

#SBATCH -n 16
#SBATCH --gpus=rtx_2080_ti:8
#SBATCH --time=24:00:00
python3 main.py --config ./configs/poisson/audio_ddpmpp.py --mode train --workdir pfgm_sc09_v2.3
