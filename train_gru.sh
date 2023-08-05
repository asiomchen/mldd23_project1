#!/bin/bash
#SBATCH --job-name=GRU_256_nograd_encoder
#SBATCH --partition=student
#SBATCH --qos=big
#SBATCH --gpus=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python -u train_gru.py -c gru_config.ini > log_gru_nograd.out
echo done
