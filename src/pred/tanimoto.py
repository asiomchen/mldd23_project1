import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect

def fp2bitstring(fp):
    """
    Changes the molecules fingerprint into a bitstring.

    Args:
        fp (list): list containing active bits of a vector

    Returns:
        bitstring (str): bitstring vector

    """
    bitstring = ['0'] * 512
    for x in fp:
        bitstring[x] = '1'
    return ''.join(bitstring)


def get_smiles_from_train(idx):
    """
    Returns smiles of a molecule by idx in the training set

    Args:
        idx (int): index of the molecule in the training set

    Returns:
        smiles (str): SMILES of the molecule
    """

    data_path = 'data/train_data/train_dataset.parquet'
    train = pd.read_parquet(data_path).smiles
    smiles = train.iloc[idx]
    return smiles


class TanimotoSearch():
    """
    Returns the highest tanimoto score between given molecule
    and all the molecules from the training set

    Args:
        return_smiles (bool): if True, returns the smiles of the molecule
        progress_bar (bool): if True, shows the progress bar

    Returns:
        high_tan (float): highest tanimoto score

    """

    def __init__(self, return_smiles=False, progress_bar=True):
        data_path = 'data/train_data/train_morgan_512bits.parquet'
        self.train_morgan_fps = pd.read_parquet(data_path).fps.apply(eval).tolist()
        self.return_smiles = return_smiles
        self.progress_bar = progress_bar

    def __call__(self, mol):

        high_tan = 0
        high_idx = 0
        query_fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=512)
        for idx, fp in enumerate(self.train_morgan_fps):
            bitstring = fp2bitstring(fp)
            ebv = DataStructs.CreateFromBitString(bitstring)
            tan = DataStructs.TanimotoSimilarity(query_fp, ebv)
            if tan > high_tan:
                high_tan, high_idx = tan, idx

        if self.return_smiles:
            high_smiles = get_smiles_from_train(high_idx)
            return high_tan, high_smiles
        else:
            return high_tan
