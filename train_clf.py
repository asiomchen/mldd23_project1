from src.utils.modelinit import initialize_model
from sklearn.metrics import roc_auc_score, confusion_matrix
from src.gen.dataset import VAEDataset
from src.gen.generator import EncoderDecoderV3
import sklearn.model_selection
from sklearn.svm import SVC
import torch.utils.data as D
import time
import pandas as pd
import pickle
import numpy as np
import wandb
import os
import argparse
import torch


def main(data_path, c_param, kernel=50, degree=3, gamma='scale'):
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = pd.read_parquet(data_path)
    print(f'Loaded data from {data_path}')
    activity = data['activity']
    model = initialize_model('models/GRUv3_ECFP_tola/hyperparameters.ini',
                             dropout=False,
                             device=device)
    model.load_state_dict(torch.load('models/GRUv3_ECFP_tola/epoch_200.pt', map_location=device))

    print('Encoding data...')
    mus, _ = encode(data, model, device)
    data = pd.DataFrame(mus)
    data['activity'] = activity
    train, test = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=42)

    SV_params = {'C': c_param,
                 'kernel': kernel,
                 'degree': degree,
                 'gamma': gamma,
                 'shrinking': True,
                 'probability': True,
                 'max_iter': -1}

    svc = SVC(**SV_params)

    train_X = train.drop('activity', axis=1)
    train_y = train['activity']
    test_X = test.drop('activity', axis=1)
    test_y = test['activity']

    print('Training...')
    svc.fit(train_X, train_y)

    name_extended = f'{svc.__class__.__name__}_C_{c_param}_kernel_{kernel}_degree_{degree}_gamma_{gamma}'
    # save model

    if not os.path.exists(f'models/{name_extended}'):
        os.mkdir(f'models/{name_extended}')
    with open(f'./models/{name_extended}/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    # evaluate
    print('Evaluating...')
    metrics = evaluate(svc, test_X, test_y)
    # wandb

    wandb.init(
        project='sklearn-clf',
        config=SV_params,
        name=name_extended
    )
    wandb.log(metrics)
    wandb.finish()

    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(f'models/{name_extended}/metrics.csv', index=False)

    time_elapsed = round((time.time() - start_time), 2)
    if time_elapsed < 60:
        print(f'Done in {time_elapsed} seconds')
    else:
        print(f'Done in {round(time_elapsed / 60, 2)} minutes')
    return


def encode(df, model, device):
    """
    Encodes the fingerprints of the molecules in the dataframe using VAE encoder.
    Args:
        df (pd.DataFrame): dataframe containing 'fps' column with Klekota&Roth fingerprints
            in the form of a list of integers (dense representation)
        model (EncoderDecoderV3): model to be used for encoding
        device (torch.device): device to be used for encoding
    Returns:
        mus (np.ndarray): array of means of the latent space
        logvars (np.ndarray): array of logvars of the latent space
    """
    dataset = VAEDataset(df, fp_len=2048)
    dataloader = D.DataLoader(dataset, batch_size=1024, shuffle=False)
    mus = []
    logvars = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            X = batch.to(device)
            mu, logvar = model.encoder(X)
            mus.append(mu.cpu().numpy())
            logvars.append(logvar.cpu().numpy())

        mus = np.concatenate(mus, axis=0)
        logvars = np.concatenate(logvars, axis=0)
    return mus, logvars


def evaluate(model, test_X, test_y):
    predictions = model.predict_proba(test_X)[:, 1]
    df = pd.DataFrame()
    df['pred'] = predictions
    df['label'] = test_y.values
    df['pred'] = df['pred'].apply(lambda x: 1 if x > 0.5 else 0)
    accuracy = df[df['pred'] == df['label']].shape[0] / df.shape[0]
    roc_auc = roc_auc_score(df['label'], df['pred'])
    tn, fp, fn, tp = confusion_matrix(df['label'], df['pred']).ravel()
    metrics = {
        'accuracy': round(accuracy, 4),
        'roc_auc': round(roc_auc, 4),
        'true_positive': round(tp / df.shape[0], 4),
        'true_negative': round(tn / df.shape[0], 4),
        'false_positive': round(fp / df.shape[0], 4),
        'false_negative': round(fn / df.shape[0], 4)
    }
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', '-d', type=str, required=True,
                        help='Path to data file')
    parser.add_argument('--c_param', '-c', type=float, default=50,
                        help='C parameter for SVM. Commonly a float in range [0.01, 1000]')
    parser.add_argument('--kernel', '-k', type=str, default='rbf',
                        help='Kernel type for SVM',
                        choices=['linear', 'poly', 'rbf', 'sigmoid'])
    parser.add_argument('--degree', '-deg', type=int, default=3,
                        help='Degree of polynomial kernel (ignored by other kernels)')
    parser.add_argument('--gamma', '-g', type=str, default='scale',
                        help='Gamma parameter for SVM. Can be "scale" or "auto" or float.')
    args = parser.parse_args()
    main(args.data_path, args.c_param, args.kernel, args.degree, args.gamma)
