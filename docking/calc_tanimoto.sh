#!/bin/bash
#SBATCH --job-name=GRU
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

source ~/miniconda3/bin/activate
conda init bash
conda activate mldd

python calc_tanimoto.py