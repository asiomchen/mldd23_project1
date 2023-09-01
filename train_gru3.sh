#!/bin/bash
#SBATCH --job-name=GRU3
#SBATCH --partition=student
#SBATCH --qos=big
#SBATCH --gpus=1
#SBATCH --cpus-per-task=3
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
wandb login
python -u train_gru.py -c gru_config3.ini > log_gru3.out
echo done
