#!/bin/bash -l
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=500MB
#SBATCH --time=72:00:00
#SBATCH -A plgporphconj-cpu
#SBATCH -p plgrid

cd $SLURM_SUBMIT_DIR
conda activate /net/pr2/projects/plgrid/plggjmdgroup/mldd
python submit_jobs.py -ns 1000 -nc 10 -d results/SVC_lolek_150_18-28-52/preds_20230927-095252/predictions.csv
echo "Jobs submitted!"