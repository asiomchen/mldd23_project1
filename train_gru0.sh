#!/bin/bash
#SBATCH --job-name=GRU0
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
wandb login
python -u train_gru.py -c gru_config0.ini > log_gru0.out
echo done
