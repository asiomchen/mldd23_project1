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
python bayesian_search.py -m models/SVC_gelu_200_23-57-38/model.pkl -t sklearn -n 1000 -i 40 -b 4 -l 32 > log_bayes.out
python bayesian_search.py -m models/SVC_tapir_200_23-55-54/model.pkl -t sklearn -n 1000 -i 40 -b 4 -l 64 > log_bayes.out

echo done
