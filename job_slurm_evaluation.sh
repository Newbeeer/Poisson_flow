#!/bin/bash
#SBATCH -n 6
#SBACTH --mem-per-cpu=4G
#SBATCH --gpus=4
#SBATCH --gres=gpumem:20g
#SBATCH --time=4:00:00
python eval.py --conf 128_deep --workdir 128_deep  --checkpoint_dir checkpoints/pfgm/128 --ckpt_number 500000 --sampling --save_audio --enable_benchmarking