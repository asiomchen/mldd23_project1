#!/bin/bash
#SBATCH --job-name=predictions
#SBATCH --partition=student
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32G

source ~/miniconda3/bin/activate
conda init bash
conda activate mldd

python fp_sampler.py -t 5ht1a -n 500000
python fp_sampler.py -t 5ht7 -n 500000
python fp_sampler.py -t d2 -n 500000
python fp_sampler.py -t beta2 -n 500000
python fp_sampler.py -t h1 -n 500000
python predict.py -c pred_config.ini