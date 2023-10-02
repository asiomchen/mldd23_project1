#!/bin/bash -l
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=500MB
#SBATCH --time=72:00:00
#SBATCH -A plgporphconj-cpu
#SBATCH -p plgrid

cd $SLURM_SUBMIT_DIR
conda activate /net/pr2/projects/plgrid/plggjmdgroup/mldd
python bayesian_search.py -m models/SVC_sonic_200/5ht1a.pkl -n 10000 -i 40 -b 4 > log_bayes0.out

echo done
