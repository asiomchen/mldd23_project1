#!/bin/bash
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
echo "training VAE"
python -u train_vae.py -c v_2.ini > log_vae_2.out
echo "training GRU"
python -u train_gru.py -c g_2.ini > log_gru_2.out
echo "run finished"