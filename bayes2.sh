#!/bin/bash

source ~/miniconda3/bin/activate
conda init bash
conda activate mldd
python bayesian_search.py -m models/SVC_eggman/model.pkl -n 100000 -i 40 -b 4 > log_bayes2.out

echo done
