import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE


def main(encoded_data_path, results_path):
    df = pd.read_parquet(encoded_data_path)
    preds_df = pd.read_csv(results_path)

    activity = df['label'].to_numpy()
    d2_numpy = df.drop(['label', 'smiles'], axis=1).to_numpy()
    preds_numpy = preds_df.drop(['score', 'norm'], axis=1).to_numpy()

    random_state = 42
    cat = np.concatenate((d2_numpy, preds_numpy), axis=0)
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=40, n_jobs=-1)
    results = tsne.fit_transform(cat)

    d2_df = pd.DataFrame((results[:len(d2_numpy)]), columns=['x', 'y'])
    preds_df = pd.DataFrame((results[len(d2_numpy):]), columns=['x', 'y'])
    d2_df['activity'] = activity
    d2_df['activity'] = ['D2 active' if x == 1 else 'D2 inactive' for x in d2_df['activity']]

    sns.set(rc={'figure.figsize': (10, 8)})
    sns.set_style("white")
    with sns.color_palette("Paired"):
        sns.scatterplot(
            data=d2_df,
            x="x",
            y="y",
            hue="activity",
            s=5
        )
    sns.scatterplot(
        data=preds_df,
        x="x",
        y="y",
        c='crimson',
        s=20,
        label='Predictions'
    )
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')

    plt.legend()

    plt.savefig(f'GRUv3_std_tails_epoch_140_preds.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--encoded_data_path', type=str, required=True,
                        help='Path to .parquet file containing encoded data of known ligands')
    parser.add_argument('p', '--predictions_path', type=str, required=True,
                        help='Path to .csv file containing predicted latent vectors')
    args = parser.parse_args()
    main(args.encoded_data_path, args.predictions_path)
