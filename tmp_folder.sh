#!/bin/bash
#SBATCH -n 2
export PYTHONPATH=$PYTHONPATH:/cluster/home/krasnopk/Poissound_flow 
python3 convert_folder.py -d /cluster/scratch/krasnopk/data/poisson/datasets/SpeechCommands/speech_commands_v0.02/two -t /cluster/scratch/krasnopk/data/poisson/datasets/SpeechCommands/speech_commands_v0.02_mel