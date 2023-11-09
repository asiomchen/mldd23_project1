import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from src.utils.finger import dense2sparse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_path', type=str, required=True)
path = parser.parse_args().data_path
df = pd.read_csv(path)

class SimilarityCalculator():
    def __init__(self):
        train = pd.read_parquet('data/train_data/train_morgan_512bits.parquet')
        train['fps'] = train['fps'].apply(eval).apply(lambda x: dense2sparse(x, 512))
        self.fps = train['fps'].values
    def get_max_train_similarity(self, smile):
        query_mol = Chem.MolFromSmiles(smile)
        query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=512)
        tanimotos = []
        for fp in self.fps:
            train_fp = self.np_to_bv(fp)
            sim = DataStructs.TanimotoSimilarity(train_fp, query_fp)
            tanimotos.append(sim)
        return max(tanimotos)
    def np_to_bv(self,fv):
        bv = DataStructs.ExplicitBitVect(len(fv))
        for i,v in enumerate(fv):
            if v:
                bv.SetBit(i)
        return bv

calculator = SimilarityCalculator()
df['max_train_similarity'] = df['smiles'].apply(calculator.get_max_train_similarity)
df.to_parquet('path')