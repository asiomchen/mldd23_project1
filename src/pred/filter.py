import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, QED, Crippen
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
from src.pred.tanimoto import TanimotoSearch


def get_largest_ring(mol):
    ri = mol.GetRingInfo()
    rings = []
    for b in mol.GetBonds():
        ring_len = [len(ring) for ring in ri.BondRings() if b.GetIdx() in ring]
        rings += ring_len
    return max(rings) if rings else 0


def calculate_ro5(mol):
    prop = {
        'MW': Chem.Descriptors.MolWt(mol),
        'LogP': Chem.Crippen.MolLogP(mol),
        'HBD': Lipinski.NumHDonors(mol),
        'HBA': Lipinski.NumHAcceptors(mol)
    }
    return pd.Series(prop)


def check_pains(df_):
    params_all = FilterCatalogParams()
    params_all.AddCatalog(FilterCatalogParams.FilterCatalogs.ALL)
    catalog_all = FilterCatalog(params_all)
    pains_ = []
    for mol in df_.mols:
        pains_.append([entry.GetProp('FilterSet') for entry in catalog_all.GetMatches(mol)])
    mask = pd.Series([bool(pains_[i]) for i in range(len(pains_))])
    for value, smiles, pain in zip(mask, df_.smiles, pains_):
        if value:
            print(value, smiles, pain)
    df_ = df_[~mask]
    return df_


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


def molecule_filter(df, config):
    """
    Filters out non-druglike molecules from a list of SMILES.
    Args:
        df (pd.DataFrame): Dataframe containing 'smiles' and 'fps' columns.
        config (ConfigParser): Configuration file.
    Returns:
        pd.DataFrame: Dataframe containing druglike molecules.
    """

    qed = float(config['FILTER']['qed'])
    tanimoto = float(config['FILTER']['tanimoto'])
    pains = config['FILTER'].getboolean('pains')
    ro5 = config['FILTER'].getboolean('ro5')
    check_sub = config['FILTER'].getboolean('check_sub')
    calc_tanimoto = config['FILTER'].getboolean('calc_tanimoto')
    progress_bar = config['SCRIPT'].getboolean('progress_bar')
    max_ring_size = int(config['FILTER']['max_ring_size'])

    # generate mol object for each smiles
    df['mols'] = df.smiles.apply(Chem.MolFromSmiles)
    print(f"Original size of dataset: {len(df)}")

    if max_ring_size is not None:
        df['max_ring'] = df['mols'].apply(get_largest_ring)
        df = df[df['max_ring'] <= max_ring_size].reset_index(drop=True)
        print(f"Dataset size after ring size check: {len(df)}")

    if qed is not None and len(df) > 0:
        df['qed'] = df['mols'].apply(QED.qed)
        df = df[df['qed'] > qed].reset_index(drop=True)
        print(f"Dataset size after QED check: {len(df)}")

    if tanimoto is not None and calc_tanimoto and len(df) > 0:
        search_agent = TanimotoSearch(return_smiles=False, progress_bar=progress_bar)
        df['tanimoto'] = df['mols'].apply(search_agent)
        df = df[df['tanimoto'] < tanimoto].reset_index(drop=True)
        print(f"Dataset size after Tanimoto check: {len(df)}")

    if pains and len(df) > 0:
        df = check_pains(df).reset_index(drop=True)
        print(f"Dataset size after PAINS check: {len(df)}")

    if ro5 and len(df) > 0:
        properties = df['mols'].apply(calculate_ro5)
        df = pd.concat([df, properties], axis=1)
        df = df[np.logical_and(df['MW'] < 500, df['LogP'] < 5)]
        df = df[np.logical_and(df['HBD'] < 10, df['HBA'] < 5)]
        df.reset_index(drop=True, inplace=True)
        print(f"Dataset size after Ro5 check: {len(df)}")

    if check_sub and len(df) > 0:
        mask = df['mols'].apply(check_substructures)
        df = df[~mask].reset_index(drop=True)
        print(f"Dataset size after substructure check: {len(df)}")

    return df
