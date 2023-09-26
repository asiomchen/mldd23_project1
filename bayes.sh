#!/bin/bash
#SBATCH --job-name=bayesian_search
#SBATCH --partition=student
#SBATCH --qos=normal
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
python bayesian_search.py -m models/SVC_eggman/model.pkl -n 100000 -i 20 -b 4 > log_bayes.out

echo done
