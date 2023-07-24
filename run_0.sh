#!/bin/bash
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
echo "training VAE"
python -u train_vae.py -c v_0.ini > log_vae_0.out
echo "training GRU"
python -u train_gru.py -c g_0.ini > log_gru_0.out
echo "run finished"