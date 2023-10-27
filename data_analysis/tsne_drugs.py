import argparse
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
import seaborn as sns
import selfies as sf
import torch
from adjustText import adjust_text
from sklearn.manifold import TSNE

from src.utils.finger import encode
from src.utils.finger import smiles2sparse
from src.utils.modelinit import initialize_model
from src.utils.vectorizer import SELFIESVectorizer


def main(model_path, data_path, drugs_path, seed):
    random.seed(seed)
    templist = model_path.split('/')
    config_path = '/'.join(templist[:-1]) + '/hyperparameters.ini'
    model_name = model_path.split('/')[1]
    receptor = drugs_path.split('/')[-1].split('_')[0]
    epoch = model_path.split('_')[-1].split('.')[0]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = initialize_model(config_path, dropout=False, device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    vectorizer = SELFIESVectorizer(pad_to_len=128)

    drugs = pd.read_csv(drugs_path)
    smiles = drugs['smiles'].to_list()
    molecule_names = drugs['name'].to_list()

    fps = [torch.Tensor(smiles2sparse(smile)) for smile in smiles]
    fps = [fp.unsqueeze(0).to(device) for fp in fps]
    fps_tensor = torch.cat(fps, dim=0)
    fp_encoded, _ = model.encoder(fps_tensor)

    preds, _ = model(fp_encoded, None, omit_encoder=True)
    preds = preds.detach().cpu().numpy()
    preds = [vectorizer.devectorize(pred, remove_special=True) for pred in preds]
    preds = [sf.decoder(x) for x in preds]
    preds = [Chem.MolFromSmiles(pred) for pred in preds]
    img = Draw.MolsToGridImage(preds, molsPerRow=3, subImgSize=(300, 300), legends=molecule_names)
    img.save(f'plots/{model_name}_epoch_{epoch}_{receptor}_structures.png')
    print('Images saved')

    df = pd.read_parquet(data_path)
    d2_encoded, _ = encode(df, model, device)
    fp_encoded_numpy = fp_encoded.detach().cpu().numpy()

    random_state = random.randint(0, 100000)
    cat = np.concatenate((fp_encoded_numpy, d2_encoded), axis=0)

    print('Running t-SNE...')
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=40, n_jobs=-1)

    results = tsne.fit_transform(cat)

    all_df = pd.DataFrame((results[-len(d2_encoded):]), columns=['x', 'y'])
    activity = [f'{receptor} active' if x == 1 else f'{receptor} inactive' for x in df['activity']]
    all_df['activity'] = activity
    drugs_df = pd.DataFrame((results[:-len(d2_encoded)]), columns=['x', 'y'])
    drugs_df['name'] = molecule_names
    drugs_df['n'] = range(1, len(drugs_df) + 1)

    sns.set(rc={'figure.figsize': (10, 8)})
    sns.set_style("white")
    sns.set_context("paper")
    with sns.color_palette("Paired"):
        sns.scatterplot(
            data=all_df,
            x="x", y="y", hue="activity", hue_order=[f"{receptor} inactive", f"{receptor} active"], s=10
        )
    sns.scatterplot(
        data=drugs_df,
        x="x", y="y", c='black', marker='o', s=30
    )
    plt.xlabel('t-SNE dim 1')
    plt.ylabel('t-SNE dim 2')

    annotation_list = []
    for line in range(0, drugs_df.shape[0]):
        annotation_list.append(
            plt.annotate(drugs_df['name'][line], xy=(drugs_df['x'][line], drugs_df['y'][line]), size=12,
                         c='black', bbox=dict(boxstyle='round', fc='white', ec='black'))
        )
    adjust_text(annotation_list, drugs_df['x'], drugs_df['y'],
                expand_points=(1.5,3), expand_text=(1, 3),
                arrowprops=dict(arrowstyle="-", color='black', lw=1), only_move={'points':'x', 'text':'xy'})
    plt.xlim(-99, 99)
    plt.ylim(-99, 99)
    plt.legend(loc='upper right')
    plt.savefig(f'plots/{model_name}_epoch_{epoch}_{receptor}_tsne.png', dpi=300, bbox_inches='tight')
    print('Plot saved to plots/')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', '-m', type=str, required=True)
    parser.add_argument('--data_path', '-d', type=str, required=True)
    parser.add_argument('--drugs_path', '-dr', type=str, required=True)
    parser.add_argument('--random_seed', '-r', type=int, default=42)
    args = parser.parse_args()
    main(args.model_path, args.data_path, args.drugs_path, args.random_seed)
