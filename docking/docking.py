import os
import subprocess
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import re
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_path', type=str, required=True)
    args = parser.parse_args()
    chunk_idx = args.data_path.split('_')[-1].split('.')[0]
    df = pd.read_csv(args.data_path)
    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['mol'] = df['mol'].apply(optimize_conformation)
    df = dock_molecules(pd.DataFrame(df))
    df.sort_index(inplace=True)
    os.makedirs('docking/outputs', exist_ok=True)
    df.to_csv(f'docking/outputs/chunk_{chunk_idx}.csv')
    os.remove(f'docking/input/chunk_{chunk_idx}.csv')
    return


def optimize_conformation(mol):
    mol = Chem.AddHs(mol)  # Adds hydrogens to make optimization more accurate
    AllChem.EmbedMolecule(mol)  # Adds 3D positions
    AllChem.MMFFOptimizeMolecule(mol)  # Improves the 3D positions using a force-field method
    return mol


def dock_molecules(df: pd.DataFrame):
    scores = []
    df_ = df.copy()
    for i, row in df_.iterrows():
        scores.append(get_docking_score(row['mol'], f'docked_{i}'))
    df_['score'] = scores
    return df_


def get_docking_score(mol: Chem.Mol, output_name: str = 'molecule_docked'):
    Chem.MolToMolFile(mol, f'docking/molecule.mol')
    os.system('obabel - imol docking/molecule.mol - omol2 - O docking/molecule.mol2')
    os.remove(f'docking/molecule.mol')
    os.system(
        f'smina -r docking/6luq_preprocessed.pdb -l docking/molecule.mol2 --autobox_ligand docking/d2_ligand.pdb --autobox_add 8 --exhaustiveness 16 --out docking/outputs/{output_name}.mol2')
    output = subprocess.check_output(
        f'smina -r docking/6luq_preprocessed.pdb -l docking/molecule_docked.mol2 --score_only', shell=True)
    score = float(re.findall(r'Affinity:\s*(\-?[\d\.]+)', str(output))[0])
    print('Score:', score)
    return score


if __name__ == '__main__':
    main()
