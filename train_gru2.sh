#!/bin/bash
#SBATCH --job-name=5ht1a
#SBATCH --partition=dgx_A100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
nvidia-smi -L
source /raid/soft/miniconda/bin/activate

conda init bash
conda activate mldd
wandb login
python -u train_gru.py -c gru_config2.ini > log_gru2.out
echo done
