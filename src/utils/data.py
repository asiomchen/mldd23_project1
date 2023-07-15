import pandas as pd
import rdkit.Chem as Chem
import rdkit.DataStructs as DataStructs
import src.utils.finger

'''
returns the highest tanimoto score between given molecule
and all of the molecules from the training set

input: mol, molecule
returns: float, highest tanimoto score when compared with
                all the molecules from the training set.
if returnSmiles:
returns: float, highest tanimoto score
         string, smiles of the molecule with the highest tanimoto score in train
'''

def closest_in_train(mol, returnSmiles=False):
    data_path = 'data/GRU_data/train_morgan_512bits.parquet'
    high_tan = 0
    high_idx = 0
    train_morgan_fps = pd.read_parquet(data_path).fps.apply(eval).tolist()
    query_fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)

    for idx, fp in enumerate(train_morgan_fps):
        fp = src.utils.finger.dense2sparse(fp, fp_len=512).tolist()
        bitstring = ''.join([str(x) for x in fp])
        ebv = DataStructs.CreateFromBitString(bitstring)
        tan = DataStructs.TanimotoSimilarity(query_fp, ebv)
        if tan > high_tan:
            high_tan = tan
            high_idx = idx
    
    if returnSmiles:
        high_smiles = get_smiles_from_train(high_idx)
        return high_tan, high_smiles
    else:
        return high_tan
    
'''
get smiles of a molecule by idx in the training set
'''
def get_smiles_from_train(idx):
    data_path = 'data/GRU_data/train_dataset.parquet'
    train = pd.read_parquet(data_path).smiles
    smiles = train.iloc[idx]
    return(smiles)