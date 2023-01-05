#!/bin/bash
#SBATCH -n 16
module load lz4
tar -cf - /cluster/work/igp_psr/ai4good/group-2a/mathias/poisson/mel_datasets/ | lz4 > datasets.tar.lz4