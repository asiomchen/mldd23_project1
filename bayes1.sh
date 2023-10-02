#!/bin/bash

source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
python bayesian_search.py -m models/SVC_sonic/c1.pkl -n 10000 -i 40 -b 4 > log_bayes1.out

echo done
