from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Union
from multiprocessing import Pool
import numpy as np
import time

klek_keys = 'keys/KlekFP_keys.txt'
maccs_keys = 'keys/MACCSFP_keys_v2.txt'
sub_keys = 'keys/SubFP_keys.txt'
klek_keys = [line.strip() for line in open(klek_keys)]
maccs_keys = [line.strip() for line in open(maccs_keys)]
sub_keys = [line.strip() for line in open(sub_keys)]
klek_keys_mols = list(map(Chem.MolFromSmarts, klek_keys))
maccs_keys_mols = list(map(Chem.MolFromSmarts, maccs_keys))
sub_keys_mols = list(map(Chem.MolFromSmarts, sub_keys))



class FPGenerator:
    def __init__(self, fp_type: str, n_jobs: int = 1):
        self.fp_type = fp_type
        self.n_jobs = n_jobs
        if fp_type not in ['klek', 'maccs', 'sub']:
            raise ValueError('Invalid fingerprint type')
    def transform(self, x: Union[List[Chem.Mol], Chem.Mol]) -> Chem.Mol:
        if isinstance(x, Chem.Mol):
            return self.get_fp(x)
        elif isinstance(x, list):
            with Pool(self.n_jobs) as p:
                return p.map(self.get_fp, x)
        elif isinstance(x, np.ndarray):
            with Pool(self.n_jobs) as p:
                return p.map(self.get_fp, x)


    def get_fp(self, x):
        if self.fp_type == 'klek':
            return self.klek_fp(x)
        elif self.fp_type == 'maccs':
            return self.maccs_fp(x)
        elif self.fp_type == 'sub':
            return self.sub_fp(x)


class KlekFPGenerator(FPGenerator):
    def __init__(self, n_jobs: int = 1):
        super().__init__('klek', n_jobs)
    def klek_fp(self, x: Chem.Mol) -> List[int]:
        fp = np.zeros(len(klek_keys))
        for i, key in enumerate(klek_keys_mols):
            if x.HasSubstructMatch(key):
                fp[i] = 1
        return fp
    
class MACCSFPGenerator(FPGenerator):
    def __init__(self, n_jobs: int = 1):
        super().__init__('maccs', n_jobs)
    def maccs_fp(self, x: Chem.Mol) -> List[int]:
        fp = np.zeros(len(maccs_keys))
        for i, key in enumerate(maccs_keys_mols):
            if x.HasSubstructMatch(key):
                fp[i] = 1
        return fp


class SubFPGenerator(FPGenerator):
    def __init__(self, n_jobs: int = 1):
        super().__init__('sub', n_jobs)
    def sub_fp(self, x: Chem.Mol) -> List[int]:
        fp = np.zeros(len(sub_keys))
        for i, key in enumerate(sub_keys_mols):
            if x.HasSubstructMatch(key):
                fp[i] = 1
        return fp


    

if __name__ == '__main__':


    mol2 = Chem.MolFromSmiles('CN1CC[C@]23C4=C5C=CC(O)=C4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5')
    mols_list = [mol2] * 1000
    mols_list = np.array(mols_list)
    klek = KlekFPGenerator(n_jobs=8)
    maccs = MACCSFPGenerator(n_jobs=8)
    sub = SubFPGenerator(n_jobs=8)
    start = time.time()
    klek.transform(mols_list)
    print(f'Time: {(time.time() - start)}s per 1000 mol to get klek')
    start = time.time()
    maccs.transform(mols_list)
    print(f'Time: {(time.time() - start)}s per 1000 mol to get maccs')
    start = time.time()
    sub.transform(mols_list)
    print(f'Time: {(time.time() - start)}s per 1000 mol to get sub')

