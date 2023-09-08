#!/bin/bash
#SBATCH --job-name=bayesian_search
#SBATCH --partition=student
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
python bayesian_search.py -m models/discr_d2_tatra_epoch_80/epoch_150.pt -n 100 -i 100 > log_bayes.out
echo done
