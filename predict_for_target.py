from src.gru.generator import EncoderDecoder
from src.gru.dataset import PredictionDataset
from src.utils.vectorizer import SELFIESVectorizer
import selfies as sf
from torch.utils.data import DataLoader
import rdkit.Chem as Chem
from rdkit.Chem import QED
import os
import torch
import pandas as pd
import argparse
from src.utils.data import closest_in_train
vectorizer = SELFIESVectorizer(pad_to_len=128)
torch.cuda.empty_cache()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
parser = argparse.ArgumentParser()
parser.add_argument("-t", "--Target", type=str, help="Target name")
args = parser.parse_args()
#--------------------------------------------------------------------------#

model_path = 'models/fixed_cce_3_layers/epoch_175.pt'
data_path = 'data/GRU_data/5ht1a_fp.parquet'

encoding_size = 512
hidden_size = 512
num_layers = 3
dropout = 0.2 # dropout must be equal 0 if num_layers = 1
teacher_ratio = 0.5

batch_size = 100
#-------------------------------------------------------------------------#
def get_predictions(model, df):
    dataset = PredictionDataset(df, vectorizer)
    loader = DataLoader(dataset, shuffle=False, batch_size=batch_size, drop_last=True)
    preds_smiles = []
    with torch.no_grad():
        model.eval()
        for X in loader:
            X = X.to(device)
            preds = model(X, None, teacher_forcing=False)
            preds = preds.detach().cpu().numpy()

            #sf.set_semantic_constraints("hypervalent")
            for seq in preds:
                selfie = vectorizer.devectorize(seq, remove_special=True)
                try:
                    preds_smiles.append(sf.decoder(selfie))
                except:
                    preds_smiles.append('C')
    return preds_smiles

def filter_out_nondruglike(predictions, threshold):
    raw_mols = [Chem.MolFromSmiles(s) for s in predictions]
    raw_qeds = [QED.qed(m) for m in raw_mols]
    filtered_mols = []
    filtered_qeds = []
    for i, value in enumerate(raw_qeds):
        mol = raw_mols[i]
        ri = mol.GetRingInfo()
        largest_ring_size = max((len(r) for r in ri.AtomRings()), default=0)
        if value > threshold and largest_ring_size < 8:
            filtered_mols.append(mol)
            filtered_qeds.append(value)
    return filtered_mols, filtered_qeds

#-------------------------------------------------------------------------#
name = args.Target

model = EncoderDecoder(
    fp_size=4860,
    encoding_size=encoding_size,
    hidden_size=hidden_size,
    num_layers=num_layers,
    dropout=dropout,
    teacher_ratio = teacher_ratio).to(device)

model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
print(f'Loaded model from {model_path}')

df = pd.read_parquet(data_path)
print('Getting predictions...')
predictions = get_predictions(model, df)
print('Filtering out non-druglike molecules...')
predictions_druglike, qeds = filter_out_nondruglike(predictions, 0.7)

print(f'Calculating tanimoto scores for {len(predictions_druglike)} molecules...')
tanimoto_scores = []
for mol in predictions_druglike:
    tanimoto = closest_in_train(mol)
    tanimoto_scores.append(tanimoto)

if not os.path.isdir(f'imgs/{name}'):
    os.mkdir(f'imgs/{name}')

# save data as csv
data_to_save = pd.DataFrame({'smiles': predictions_druglike, 'fp': df['fps'], 'qed': qeds, 'tanimoto': tanimoto_scores})
data_to_save.to_csv(f'imgs/{name}/{name}.csv', index=False)
print(f'Saved data to imgs/{name}/{name}.csv')

i = 0
while i < 1000:
    mol4 = predictions_druglike[i:(i+4)]
    qed4 = qeds[i:(i+4)]
    qed4 = ['{:.2f}'.format(x) for x in qed4]
    tan4 = tanimoto_scores[i:(i+4)]
    tan4 = ['{:.2f}'.format(x) for x in tan4]
    img = Chem.Draw.MolsToGridImage(mol4, molsPerRow=2, subImgSize=(400, 400),
                                    legends=[f'QED: {qed}, Tan: {tan}' for qed, tan in zip(qed4, tan4)],
                                    returnPNG=False
                                   )
    img.save(f'imgs/{name}/{i}.png')
    i += 4
print(f'Saved images to imgs/{name}/')