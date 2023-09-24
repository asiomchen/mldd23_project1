import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors
import rdkit.Chem.rdmolops as rdmolops


def molecule_score(mol, max_ring_size=7):
    """
    Used to choose the most druglike mol from a collection of dropout prediction mols.
    Args:
        mol: rdkit mol object.
        max_ring_size (int): maximum ring size.
    Returns:
        float: score.
    """
    fragments = ['[C]1-[C]=[C]-1',  # cyclopropene
                 '[C]1-[C]#[C]-1',  # cyclopropyne
                 '[*]=[C]1-[C]-[C]-1'  # cyclopropane bound to anything with double bond
                 '[*]-[C]1-[C]-[C]-1-[*]',  # 1,2-disubstituted cyclopropane,
                 ]

    score = QED.qed(mol)
    if get_largest_ring(mol) > max_ring_size:
        score = score * (0.8)
    if rdMolDescriptors.CalcNumRings(mol) > 8:
        score = score * (0.8)
    for fragment in fragments:
        if mol.HasSubstructMatch(Chem.MolFromSmarts(fragment)):
            score = score * (0.8)

    return score


def molecule_filter(dataframe, config):
    """
    Filters out non-druglike molecules from a list of SMILES.
    Args:
        dataframe (pd.DataFrame): Dataframe containing 'smiles' and 'fps' columns.
        config (ConfigParser): Configuration file.
    Returns:
        pd.DataFrame: Dataframe containing druglike molecules.
    """

    qed = float(config['FILTER']['qed'])
    max_ring_size = int(config['FILTER']['max_ring_size'])
    max_num_rings = int(config['FILTER']['max_num_rings'])
    df = dataframe.copy(deep=True)

    # generate mol object for each smiles
    df['mols'] = df.smiles.apply(Chem.MolFromSmiles)

    # print(f"Original size of dataset: {len(df)}")

    if max_num_rings is not None:
        df['num_rings'] = df['mols'].apply(Chem.rdMolDescriptors.CalcNumRings)
        df = df[df['num_rings'] <= max_num_rings].reset_index(drop=True)

    if max_ring_size is not None and len(df) > 0:
        df['max_ring'] = df['mols'].apply(get_largest_ring)
        df = df[df['max_ring'] <= max_ring_size].reset_index(drop=True)
        # print(f"Dataset size after ring size check: {len(df)}")

    if qed is not None and len(df) > 0:
        df['qed'] = df['mols'].apply(QED.qed)
        df = df[df['qed'] > qed].reset_index(drop=True)
        # print(f"Dataset size after QED check: {len(df)}")

    return df


def workup(dataframe):
    df = dataframe.copy(deep=True)
    df['mol'] = df.smiles.apply(Chem.MolFromSmiles)

    # delete common artifacts
    df['mol'].apply(delete_terminal_artifacts, inplace=True)

    # sanitize
    df['mol'].apply(try_sanitize, inplace=True)

    # remove invalid mols
    df = df[df.mol.notnull()]

    df.drop(columns=['mol'], inplace=True)

    return df


def delete_terminal_artifacts(mol):
    smarts = ['[C]1-[C]=[C]-1',  # cyclopropene
              '[C]1-[C]#[C]-1',  # cyclopropyne
              '[*]=[C]1-[C]-[C]-1'  # cyclopropane bound to anything with double bond
              # TODO: add more common artifacts
              ]
    fragment_mols = [Chem.MolFromSmarts(smart) for smart in smarts]
    for fragment_mol in fragment_mols:
        mol = rdmolops.DeleteSubstructs(mol, fragment_mol)
    return mol


def get_largest_ring(mol):
    ri = mol.GetRingInfo()
    rings = []
    for b in mol.GetBonds():
        ring_len = [len(ring) for ring in ri.BondRings() if b.GetIdx() in ring]
        rings += ring_len
    return max(rings) if rings else 0


def try_sanitize(mol):
    try:
        output = mol
        Chem.SanitizeMol(output)
        return output
    except:
        return mol


