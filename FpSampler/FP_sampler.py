import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--protein', type=str, required=True, help='selects protein to use')
parser.add_argument('-d', '--dtype', type=str, required=True, help='selects the dataset to use')
parser.add_argument('-n', '--number', type=int, required=True, help='number of fingerprints to generate')
parser.add_argument('-m', '--magnitude', type=int, default=1, help='increases probability of bits from '
                                                                   'active class to be selected')
parser.add_argument('-r', '--random', type=bool, default=False, help='adds random noise to fingerprints')
parser.add_argument('-a', '--av_bits', type=int, default=60, help='average number of active bits in fingerprint')


args = parser.parse_args()

protein = args.protein
dtype = args.dtype
number = args.number
magnitude = args.magnitude
random = args.random
av_bits = args.av_bits


class FpSampler:
    def __init__(self, protein, dtype='100nM', magnitude=1, add_random=False):
        allowed_proteins = ['5ht1a', '5ht7', 'beta2', 'd2', 'h1']
        allowed_dtypes = ['100nM', 'balanced']

        if protein not in allowed_proteins:
            raise ValueError(f"Invalid value for protein. Allowed values are: {allowed_proteins}")

        if dtype not in allowed_dtypes:
            raise ValueError(f"Invalid value for dtype. Allowed values are: {allowed_dtypes}")

        self.protein = protein
        self.dtype = dtype
        self.magnitude = magnitude
        self.add_random = add_random
        self.path = f"./datasets/fp_frequency_{self.dtype}/{self.protein}_frequency.csv"
        self.df = pd.read_csv(self.path, sep=',')
        self.sizes = {
            '5ht1a': 5250,
            '5ht7': 2963,
            'beta2': 782,
            'd2': 10170,
            'h1': 1691
        }

        print(f"{self.dtype} dataset for {self.protein.upper()} loaded")

        self.dummy_df = None
        self.org_df = None
        self.fp_df = None

        self.make_dummy_df()
        self.read_original()
        self.combine_df()
        self.convert_to_proba()

    def make_dummy_df(self):
        dummy_dict = {'KEYS': [f"KLEK_{n}" for n in range(4860)]}
        self.dummy_df = pd.DataFrame(dummy_dict)

    def read_original(self):
        self.org_df = pd.read_csv(f"./datasets/counts_full/counts_full_{self.dtype}.csv", sep=',')
        self.org_df = self.org_df.loc[:, [f"{self.protein}", "KEYS"]]
        self.org_df['Freq'] = self.org_df[f'{self.protein}'] / self.sizes[f"{self.protein}"]

    def combine_df(self):
        self.fp_df = self.dummy_df.merge(self.org_df, on='KEYS')
        self.fp_df = self.fp_df.merge(self.df, on='KEYS').drop(columns=['SMARTS'])

    def convert_to_proba(self):
        self.fp_df['Probability'] = self.fp_df['Freq'] * (
                    (self.fp_df[f"{self.protein}_percentage"] * self.magnitude + 100) / 100)
        self.fp_df['Probability'] = [x if x >= 0 else 0 for x in self.fp_df['Probability']]
        self.fp_df['Probability'] = self.fp_df['Probability'] / self.fp_df['Probability'].apply(np.abs).sum()
        if self.add_random:
            self.fp_df['Probability'] = pd.Series([0.0001 if x < 0.0001 else x for x in self.fp_df['Probability']])
            self.fp_df['Probability'] = self.fp_df['Probability'] / self.fp_df['Probability'].sum()

    def generate_fingerprints(self, av_bits=60, n=1000):
        fps = []
        length = []
        for fp in range(n):
            vec1 = np.array(self.fp_df['Probability']) * av_bits
            vec2 = np.random.rand(4860)
            fp = (vec1 > vec2).astype('int')
            length.append(np.sum(fp))
            fp = str([int(x) for x in np.where(fp == 1)[0]])
            fps.append(fp)
        print(f"Generated {n} vectors with mean length of {np.mean(length):.3f} and SD of {np.std(length):.3f}")
        return fps


sampler = FpSampler(protein=protein, magnitude=magnitude, add_random=random)
samples = sampler.generate_fingerprints(av_bits=av_bits, n=number)
samples_df = pd.DataFrame({'fps': samples})
samples_df.to_parquet(f'./{protein}_{dtype}_samples.parquet')
