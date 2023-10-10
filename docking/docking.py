import os
import subprocess
import pandas as pd
import sys
from rdkit import Chem
from rdkit.Chem import AllChem
import re
import argparse
import warnings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    parser.add_argument('-n', '--name', type=str, required=True)
    args = parser.parse_args()
    name = args.name
    chunk_idx = args.data_path.split('_')[-1].split('.')[0]
    df = pd.read_csv(args.data_path)
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['mol'] = df['mol'].apply(optimize_conformation)
    os.makedirs(f'docking/outputs_{name}/{chunk_idx}', exist_ok=True)
    df = dock_molecules(pd.DataFrame(df), chunk_idx, f"outputs_{name}")
    df.sort_index(inplace=True)
    os.makedirs(f'docking/outputs_{name}', exist_ok=True)
    df.to_csv(f'docking/outputs_{name}/chunk_{chunk_idx}.csv')
    os.remove(f'docking/inputs/chunk_{chunk_idx}.csv')
    return


def optimize_conformation(mol):
    mol = Chem.AddHs(mol)  # Adds hydrogens to make optimization more accurate
    AllChem.EmbedMolecule(mol)  # Adds 3D positions
    AllChem.MMFFOptimizeMolecule(mol)  # Improves the 3D positions using a force-field method
    return mol


def dock_molecules(df: pd.DataFrame, chunk_idx: int = 0, folder: str = 'outputs'):
    scores = []
    df_ = df.copy()
    for i, row in df_.iterrows():
        scores.append(get_docking_score(row['mol'], f'docked_{i}', chunk_idx, folder))
    df_['score'] = scores
    return df_


def get_docking_score(mol: Chem.Mol, output_name: str = 'molecule_docked', chunk_idx: int = 0, folder: str = 'outputs'):
    try:
        Chem.MolToMolFile(mol, f'docking/molecule_{chunk_idx}.mol')
        os.system(f'obabel -imol docking/molecule_{chunk_idx}.mol -omol2 -O docking/molecule_{chunk_idx}.mol2')
        os.remove(f'docking/molecule_{chunk_idx}.mol')
        os.system(
            f'smina -r docking/6luq_preprocessed.pdb -l docking/molecule_{chunk_idx}.mol2 --autobox_ligand docking/d2_ligand.pdb --autobox_add 8 --exhaustiveness 16 --out docking/{folder}/{chunk_idx}/{output_name}.mol2')
        output = subprocess.check_output(
            f'smina -r docking/6luq_preprocessed.pdb -l docking/{folder}/{chunk_idx}/{output_name}.mol2 --score_only', shell=True)
        score = float(re.findall(r'Affinity:\s*(\-?[\d\.]+)', str(output))[0])
        print('Score:', score)
    except Exception as e:
        warnings.warn(f'Error: {e} \nScore set to 0')
        score = 0

    return score


if __name__ == '__main__':
    main()
