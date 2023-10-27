#!/bin/bash -l
#SBATCH -N 1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=500MB
#SBATCH --time=72:00:00
#SBATCH -A plgporphconj-cpu
#SBATCH -p plgrid

cd $SLURM_SUBMIT_DIR
conda activate /net/pr2/projects/plgrid/plggjmdgroup/mldd
python submit_jobs.py -ns 1000 -nc 10 -d results/5ht1a_SVC_20231003_152751/preds_20231005-172945/predictions.csv
python submit_jobs.py -ns 1000 -nc 10 -d results/c1_SVC_20231003_193344/preds_20231005-195627/predictions.csv
echo "Jobs submitted!"