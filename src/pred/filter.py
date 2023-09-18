import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, rdMolDescriptors
from src.pred.tanimoto import TanimotoSearch


def get_largest_ring(mol):
    ri = mol.GetRingInfo()
    rings = []
    for b in mol.GetBonds():
        ring_len = [len(ring) for ring in ri.BondRings() if b.GetIdx() in ring]
        rings += ring_len
    return max(rings) if rings else 0


def check_substructures(mol):
    value = False
    smarts = ['[C]1-[C]=[C]-1',  # cyclopropene
              '[C]1-[C]#[C]-1',  # cyclopropyne
              '[*]=[C]1-[C]-[C]-1',  # cyclopropane bound to anything with double bond
              '[*]-[C]1-[C]-[C]-1-[*]',  # epoxide-type cyclopropane
              '[C]1=[C]-[C]=[C]-1',  # cyclobutadiene
              '[C]1#[C]~[C]~[C]-1',  # cyclobutyne
              '[C]1=[C]-[C]=[C]-[C]-[C]-1',  # 1,3-cyclohexadiene
              '[C]1=[C]-[C]-[C]=[C]-[C]-1',  # 1,4-cyclohexadiene
              '[c]1#[c]~[c]~[c]~[c]~[c]~1'  # any pseudo-aromatic 6-membered ring with triple bond
              ]
    for substructure in smarts:
        pattern = Chem.MolFromSmarts(substructure)
        value = mol.HasSubstructMatch(pattern)
        if value:
            break
    return value


def molecule_filter(dataframe, config, return_list):
    """
    Filters out non-druglike molecules from a list of SMILES.
    Args:
        dataframe (pd.DataFrame): Dataframe containing 'smiles' and 'fps' columns.
        config (ConfigParser): Configuration file.
        return_list (list): List to which the results are appended.
    Returns:
        pd.DataFrame: Dataframe containing druglike molecules.
    """

    qed = float(config['FILTER']['qed'])
    max_tanimoto = float(config['FILTER']['max_tanimoto'])
    check_sub = config['FILTER'].getboolean('check_sub')
    calc_tanimoto = config['FILTER'].getboolean('calc_tanimoto')
    verbose = config['SCRIPT'].getboolean('verbose')
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

    if max_tanimoto is not None and calc_tanimoto and len(df) > 0:
        search_agent = TanimotoSearch(return_smiles=False, progress_bar=verbose)
        df['tanimoto'] = df['mols'].apply(search_agent)
        df = df[df['tanimoto'] < max_tanimoto].reset_index(drop=True)
        # print(f"Dataset size after Tanimoto check: {len(df)}")

    if check_sub and len(df) > 0:
        mask = df['mols'].apply(check_substructures)
        df = df[~mask].reset_index(drop=True)
        # print(f"Dataset size after substructure check: {len(df)}")

    return_list.append(df)
    return None
