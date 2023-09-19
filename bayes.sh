#!/bin/bash
#SBATCH --job-name=bayesian_search
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
python bayesian_search.py -m models/SVC_gelu_200_23-57-38/model.pkl -t sklearn -n 1000 -i 20 -b 4 > log_bayes.out
python bayesian_search.py -m models/SVC_tapir_200_23-55-54/model.pkl -t sklearn -n 1000 -i 20 -b 4 > log_bayes.out

echo done
