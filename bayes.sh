#!/bin/bash
#SBATCH --job-name=bayesian_search
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
python bayesian_search.py -m models/SVC_sonic/model.pkl -n 100000 -i 20 -b 4 > log_bayes.out

echo done
