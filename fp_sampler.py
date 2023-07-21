import numpy as np
import pandas as pd
import time
import argparse

# parse arguments

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--target', type=str, default='5ht1a',
                    help='(str) select target to use: 5ht1a,n 5ht7, beta2, d2, h1')
parser.add_argument('-d', '--dataset', type=str, default='100nM',
                    help='(str) selects the dataset to use for sampling: 100nM, balanced')
parser.add_argument('-n', '--number', type=int, default=10000,
                    help='(int) number of fingerprints to generate')
parser.add_argument('-m', '--magnitude', type=int, default=1,
                    help='(int) increase probability of bits from active class to be selected (1-5), default=1')
parser.add_argument('-r', '--random', type=bool, default=False,
                    help='(bool) adds random noise to fingerprints')
parser.add_argument('-a', '--av_bits', type=int, default=60,
                    help='(int) average number of active bits in fingerprint')

args = parser.parse_args()
target = args.target
dataset = args.dataset
number = args.number
magnitude = args.magnitude
random = args.random
av_bits = args.av_bits

class FpSampler:
    def __init__(self, target, dataset, magnitude, add_random):
        allowed_targets = ['5ht1a', '5ht7', 'beta2', 'd2', 'h1']
        allowed_datasets = ['100nM', 'balanced']

        if target not in allowed_targets:
            raise ValueError(f"Invalid value for target. Allowed values are: {allowed_targets}")

        if dataset not in allowed_datasets:
            raise ValueError(f"Invalid value for dataset. Allowed values are: {allowed_datasets}")

        self.target = target
        self.dataset = dataset
        self.magnitude = magnitude
        self.add_random = add_random
        self.path = f"data/sampler_data/fp_frequency_{self.dataset}/{self.target}_frequency.csv"
        self.df = pd.read_csv(self.path, sep=',')
        self.sizes = {
            '5ht1a': 5250,
            '5ht7': 2963,
            'beta2': 782,
            'd2': 10170,
            'h1': 1691
        }

        print(f"{self.dataset} dataset for {self.target.upper()} loaded")

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
        self.org_df = pd.read_csv(f"data/sampler_data/counts_full/counts_full_{self.dataset}.csv", sep=',')
        self.org_df = self.org_df.loc[:, [f"{self.target}", "KEYS"]]
        self.org_df['Freq'] = self.org_df[f'{self.target}'] / self.sizes[f"{self.target}"]

    def combine_df(self):
        self.fp_df = self.dummy_df.merge(self.org_df, on='KEYS')
        self.fp_df = self.fp_df.merge(self.df, on='KEYS').drop(columns=['SMARTS'])

    def convert_to_proba(self):
        self.fp_df['Probability'] = self.fp_df['Freq'] * (
                    (self.fp_df[f"{self.target}_percentage"] * self.magnitude + 100) / 100)
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

def main():
    sampler = FpSampler(target=target, magnitude=magnitude, add_random=random, dataset=dataset)
    samples = sampler.generate_fingerprints(av_bits=av_bits, n=number)
    samples_df = pd.DataFrame({'fps': samples})
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    samples_df.to_parquet(f'results/{timestamp}_{target}_{dataset}.parquet')
    return None


if __name__ == '__main__':
    main()