#!/bin/bash
#SBATCH --job-name=docking_0
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=500MB
#SBATCH --time=24:00:00
#SBATCH -A plgporphconj-cpu
#SBATCH -p plgrid
cd $SLURM_SUBMIT_DIR
conda activate activate /net/pr2/projects/plgrid/plggjmdgroup/mldd
python docking/docking.py -d docking/inputs/chunk_0.csv
echo Done