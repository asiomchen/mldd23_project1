#!/bin/bash
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
echo "training VAE"
python -u train_vae.py -c v_1.ini > log_vae_1.out
echo "training GRU"
python -u train_gru.py -c g_1.ini > log_gru_1.out
echo "training GRU with RL"
python -u RL.py -c rl_1.ini > log_rl_1.out
echo "run finished"