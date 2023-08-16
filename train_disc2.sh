#!/bin/bash
#SBATCH --job-name=discriminator2
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
python -u train_disc.py -c disc_config4.ini > log_disc4.out
python -u train_disc.py -c disc_config5.ini > log_disc5.out
echo done
