import pandas as pd
import argparse
import multiprocessing
from joblib import Parallel, delayed
from rdkit import Chem

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--file', type=str, required=True, help='(str) path to the .csv file')

parser.add_argument('-s', '--smiles_col', type=str, default='SMILES', help='(str) name of the smiles column in .csv file')

parser.add_argument('-n', '--n_jobs', type=int, default=-1, help='(int) how many processes to use for fp generation')

parser.add_argument('-o', '--output', type=str, default='./outputs/out.csv', help='(str) output filename, optional')

parser.add_argument('-k', '--keys', type=str, default='../data/KlekFP_keys.txt', help='(str) path to SMARTS keys')

args = parser.parse_args()

file = args.file
smiles_col = args.smiles_col
output = args.output
n_jobs = args.n_jobs
keys = args.keys

klek_keys = [line.strip() for line in open(keys)]
klek_keys_mols = list(map(Chem.MolFromSmarts, klek_keys))


def calculate_sparse_fingerprint(mol):
    mol_list = []
    for i, key in enumerate(klek_keys_mols):
        if mol.HasSubstructMatch(key):
            mol_list.append(i)
    return str(mol_list)


def process_molecules(mols):
    results = [delayed(calculate_sparse_fingerprint)(mol) for mol in mols]
    return compute(*results)


df = pd.read_csv(file)

mols = df[smiles_col].apply(Chem.MolFromSmiles)

if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        multiprocessing.set_start_method('spawn', force=True)

    molecules_list = list(mols)

    chunk_size = 512
    chunks = [molecules_list[i:i + chunk_size] for i in range(0, len(molecules_list), chunk_size)]

    with Parallel(n_jobs=n_jobs, prefer="processes") as parallel:
        def process_chunk(chunk):
            return [calculate_sparse_fingerprint(mol) for mol in chunk]

        all_results = sum(parallel(delayed(process_chunk)(chunk) for chunk in chunks), [])

    output_df = pd.DataFrame({'SMILES': df[smiles_col], 'fps': all_results})

    output_df.to_csv(output, sep=',', index=False, header=True)
