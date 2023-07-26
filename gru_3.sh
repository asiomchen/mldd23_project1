#!/bin/bash
source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
echo "training GRU"
python -u train_gru.py -c g_3.ini > log_gru_3.out
echo "training GRU with RL"
python -u RL.py -c rl_3.ini > log_rl_3.out
echo "run finished"