#!/bin/bash

source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
python bayesian_search.py -m models/SVC_bolek_150/model.pkl -n 100000 -i 40 -b 4 > log_bayes5.out

echo done
