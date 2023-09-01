#!/bin/bash
#SBATCH --job-name=GRU2
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
python -u train_gru.py -c gru_config2.ini > log_gru2.out
echo done
