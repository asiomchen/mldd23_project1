#!/bin/bash
#SBATCH --job-name=CVAE
#SBATCH --partition=student
#SBATCH --qos=big
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32G

nvidia-smi -L
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
python -u train_cvae.py -c cave_5ht1a.ini > log_cvae_5ht1a.out
python -u train_cvae.py -c cave_5ht7.ini> log_cvae_5ht7.out
python -u train_cvae.py -c cave_beta2.ini> log_cvae_beta2.out
python -u train_cvae.py -c cave_d2.ini> log_cvae_d2.out
python -u train_cvae.py -c cave_h1.ini> log_cvae_h1.out
echo done
