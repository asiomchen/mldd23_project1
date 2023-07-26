#!/bin/bash
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
echo "training GRU"
python -u train_gru.py -c g_4.ini > log_gru_4.out
echo "training GRU with RL"
python -u RL.py -c rl_4.ini > log_rl_4.out
echo "run finished"