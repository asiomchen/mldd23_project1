from src.gen.generator import EncoderDecoderV3
import torch
import seaborn as sns
import pandas as pd
from gru_pca import encode
from src.utils.finger import smiles2sparse
from sklearn.manifold import TSNE
import argparse
import configparser
import rdkit.Chem as Chem
import rdkit.Chem.Draw as Draw
from src.utils.vectorizer import SELFIESVectorizer
import numpy as np
import selfies as sf
import random
import matplotlib.pyplot as plt
from adjustText import adjust_text

config = configparser.ConfigParser()

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', '-m', type=str, required=True)
parser.add_argument('--random_seed', '-r', type=int, default=42)
parser.add_argument('--data', '-d', type=str, default='d2_drugs.csv')
seed = parser.parse_args().random_seed
random.seed(seed)

model_path = parser.parse_args().model_path
templist = model_path.split('/')
config_path = '/'.join(templist[:-1]) + '/hyperparameters.ini'
config.read(config_path)
model_name = model_path.split('/')[1]
epoch = model_path.split('_')[-1].split('.')[0]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

try:
    encoder_activation = str(config['MODEL']['encoder_activation'])
except KeyError:
    encoder_activation = 'relu'

model = EncoderDecoderV3(fp_size=int(config['MODEL']['fp_len']),
                         encoding_size=int(config['MODEL']['encoding_size']),
                         hidden_size=int(config['MODEL']['hidden_size']),
                         num_layers=int(config['MODEL']['num_layers']),
                         output_size=42,
                         dropout=0,
                         teacher_ratio=0.0,
                         fc1_size=int(config['MODEL']['fc1_size']),
                         fc2_size=int(config['MODEL']['fc2_size']),
                         fc3_size=int(config['MODEL']['fc3_size']),
                         random_seed=seed,
                         encoder_activation=encoder_activation
                         ).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))

vectorizer = SELFIESVectorizer()

drugs = pd.read_csv(parser.parse_args().data)
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
img.save(f'plots/{model_name}_epoch_{epoch}_drugs.png')

df = pd.read_parquet('data/activity_data/d2_klek_100nM.parquet')
d2_encoded, _ = encode(df, model, device)
fp_encoded_numpy = fp_encoded.detach().cpu().numpy()

random_state = random.randint(0, 100000)
cat = np.concatenate((fp_encoded_numpy, d2_encoded), axis=0)
tsne = TSNE(n_components=2, random_state=random_state, perplexity=40, n_jobs=-1)

results = tsne.fit_transform(cat)

all_df = pd.DataFrame((results[-len(d2_encoded):]), columns=['x', 'y'])
activity = ['D2 active' if x == 1 else 'D2 inactive' for x in df['Class']]
all_df['activity'] = activity
drugs_df = pd.DataFrame((results[:-len(d2_encoded)]), columns=['x', 'y'])
drugs_df['name'] = molecule_names
drugs_df['n'] = range(1, len(drugs_df) + 1)

sns.set(rc={'figure.figsize': (10, 8)})
sns.set_style("white")
with sns.color_palette("Paired"):
    sns.scatterplot(
        data=all_df,
        x="x", y="y", hue="activity", s=5
    )
sns.scatterplot(
    data=drugs_df,
    x="x", y="y", c='black', s=20
)
plt.xlabel('t-SNE dim 1')
plt.ylabel('t-SNE dim 2')

annotation_list = []
for line in range(0, drugs_df.shape[0]):
    annotation_list.append(
        plt.annotate(drugs_df['name'][line], xy=(drugs_df['x'][line], drugs_df['y'][line]), size=10,
                     c='black', weight='bold')
    )
adjust_text(annotation_list)

plt.savefig(f'plots/{model_name}_epoch_{epoch}_tsne.png')