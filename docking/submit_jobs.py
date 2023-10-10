import pandas as pd
import argparse
import numpy as np
import os


def main():
    # split data into chunks

    parser = argparse.ArgumentParser()
    parser.add_argument('-nc', '--n_chunks', type=int, default=10)
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-ns', '--n_samples', type=int, default=1000)
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    n_chunks = args.n_chunks
    if args.data_path.split('.')[-1] == 'csv':
        df = pd.read_csv(args.data_path).sample(args.n_samples, random_state=args.seed)
    elif args.data_path.split('.')[-1] == 'parquet':
        df = pd.read_parquet(args.data_path).sample(args.n_samples, random_state=args.seed)
    else:
        raise ValueError('Data format not supported')
    df_list = np.split(df, n_chunks)
    os.makedirs('docking/inputs', exist_ok=True)
    for n, df in enumerate(df_list):
        df.to_csv(f'docking/inputs/chunk_{n}.csv', index=False)

    # generate shell scripts and submit jobs

    for i in range(n_chunks):
        with open(f'docking/docking_{i}.sh', 'w') as f:
            header = [
                f'''#!/bin/bash''',
                f'''#SBATCH --job-name=docking_{i}''',
                '''#SBATCH -N 1''',
                '''#SBATCH --cpus-per-task=16''',
                '''#SBATCH --mem-per-cpu=500MB''',
                '''#SBATCH --time=24:00:00''',
                '''#SBATCH -A plgporphconj-cpu''',
                '''#SBATCH -p plgrid''',
                '''cd $SLURM_SUBMIT_DIR''',
                '''conda activate activate /net/pr2/projects/plgrid/plggjmdgroup/mldd\n'''
            ]
            f.write('\n'.join(header))
            f.write(f'''python docking/docking.py -d docking/inputs/chunk_{i}.csv -n {args.name}\n''')
            f.write('''echo Done''')
        os.system(f'sbatch docking/docking_{i}.sh')
        os.remove(f'docking/docking_{i}.sh')
    return

if __name__ == '__main__':
    main()

