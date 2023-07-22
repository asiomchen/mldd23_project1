#!/bin/bash
#SBATCH --job-name=RL_GRU_128
#SBATCH --partition=student
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
python -u RL.py > log_rl.out
echo done