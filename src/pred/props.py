import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.rdMolDescriptors as rdMolDescriptors
import rdkit.Chem.Crippen as Crippen
import rdkit.Chem.QED as QED

def get_properties(dataframe):
    df = dataframe.copy(deep=True)

    df['mol'] = df['smiles'].apply(Chem.MolFromSmiles)
    df['mol_wt'] = df['mol'].apply(rdMolDescriptors.CalcExactMolWt)
    df['num_HBA'] = df['mol'].apply(rdMolDescriptors.CalcNumHBA)
    df['num_HBD'] = df['mol'].apply(rdMolDescriptors.CalcNumHBD)
    df['logP'] = df['mol'].apply(Crippen.MolLogP)
    df['num_rotatable_bonds'] = df['mol'].apply(rdMolDescriptors.CalcNumRotatableBonds)
    df['TPSA'] = df['mol'].apply(rdMolDescriptors.CalcTPSA)
    df['QED'] = df['mol'].apply(QED.qed)
    df.drop(columns=['mol'], inplace=True)

    return df