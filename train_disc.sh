#!/bin/bash
#SBATCH --job-name=discriminator
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
python -u train_disc.py -c disc_config0.ini > log_disc0.out
python -u train_disc.py -c disc_config1.ini > log_disc1.out
echo done
