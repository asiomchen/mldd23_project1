#!/bin/bash
#SBATCH --job-name=VAE5
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
wandb login
python -u train_vae.py -c vae5.ini > log_vae5.out
echo done
