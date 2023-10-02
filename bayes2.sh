#!/bin/bash

source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
python bayesian_search.py -m models/SVC_tola_150/5ht1a.pkl -n 10000 -i 40 -b 4 > log_bayes2.out

echo done
