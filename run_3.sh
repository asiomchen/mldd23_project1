#!/bin/bash
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
echo "training VAE"
python -u train_vae.py -c v_3.ini > log_vae_3.out
echo "training GRU"
python -u train_gru.py -c g_3.ini > log_gru_3.out
echo "run finished"