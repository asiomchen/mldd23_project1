#!/bin/bash
./get_datasets.sh
sleep 100
sbatch train_gru0.sh
sbatch train_gru1.sh
sbatch train_gru2.sh
sbatch train_gru3.sh
echo 'All jobs submitted!'