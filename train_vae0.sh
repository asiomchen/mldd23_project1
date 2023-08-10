#!/bin/bash
#SBATCH --job-name=VAE0
#SBATCH --partition=student
#SBATCH --qos=quick
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
wandb login 505ce3ad45fdf9309c3d8ec1d9764262ae6929c1
python -u train_vae.py -c vae0.ini > log_vae0.out
echo done
