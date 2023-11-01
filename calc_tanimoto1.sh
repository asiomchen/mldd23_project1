#!/bin/bash
#SBATCH --job-name=GRU
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

source ~/miniconda3/bin/activate
conda init bash
conda activate mldd

python calc_tanimoto.py -d results/d2_SVC_lolek_150_18-28-52/preds_20230927-095252/predictions.csv