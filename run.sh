#!/bin/sh
#SBATCH --gpus=1
#SBATCH --output=/cluster/scratch/krasnopk/data/log/%j.out     
#SBATCH --error=/cluster/scratch/krasnopk/data/log/%j.err
#SBATCH --gres=gpumem:12G
#SBATCH --mem-per-cpu=24G
#SBATCH --time=240

# module load gcc/8.2.0 python_gpu/3.10.4 ninja libsndfile eth_proxy hdf5

# module load gcc/6.3.0 python_gpu/3.7.4 ninja libsndfile eth_proxy hdf5

echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode train --workdir ../Poissound_flow/test

CUDA_VISIBLE_DEVICES=0,1 python3 eval_script.py --config ./configs/poisson/cifar10_ddpmpp.py --mode eval --workdir ../Poisson_flow/test --config.eval.enable_sampling --config.sampling.N 50

# CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode eval \
# --workdir ../Poisson_flow/test --config.eval.enable_sampling --config.sampling.N 20

# CUDA_VISIBLE_DEVICES=0,1 python3 main.py --config ./configs/poisson/cifar10_ddpmpp.py --mode eval \
# --workdir ../Poisson_flow/test --config.eval.enable_sampling --config.sampling.N 100

echo "Finished at:     $(date)"

exit 0