import time
import pandas as pd
import pickle
from sklearn.svm import SVC
import sklearn.model_selection
from sklearn.metrics import roc_auc_score, confusion_matrix
import wandb
import os
import argparse


def main(data_path, c_param, kernel=50, degree=3, gamma='scale'):
    data = pd.read_parquet(data_path)
    model_name = data_path.split('/')[-2].split('_')[-1]
    model_epoch = data_path.split('/')[-1].split('_')[-1].split('.')[0]
    train, test = sklearn.model_selection.train_test_split(data, test_size=0.1, random_state=42)

    SV_params = {'C': c_param,
                 'kernel': kernel,
                 'degree': degree,
                 'gamma': gamma,
                 'shrinking': True,
                 'probability': True,
                 'max_iter': -1}

    SV = SVC(**SV_params)
    model = SV
    model_params = SV_params

    train_X = train.drop(['label', 'smiles'], axis=1)
    train_y = train['label']
    test_X = test.drop(['label', 'smiles'], axis=1)
    test_y = test['label']

    model.fit(train_X, train_y)

    name = model.__str__().split('(')[0]
    timestamp = (str(time.localtime()[3]) + '-' +
                 str(time.localtime()[4]) + '-' +
                 str(time.localtime()[5])
                 )
    name_extended = (name + '_'
                     + model_name + '_'
                     + model_epoch + '_'
                     + timestamp)

    # save model

    if not os.path.exists(f'models/{name_extended}'):
        os.mkdir(f'models/{name_extended}')
    with open(f'./models/{name_extended}/model.pkl', 'wb') as file:
        pickle.dump(model, file)

    metrics = evaluate(model, test_X, test_y)
    # wandb

    wandb.init(
        project='sklearn-clf',
        config=model_params,
        name=name_extended
    )
    wandb.log(metrics)
    wandb.finish()

    metrics_df = pd.DataFrame(metrics, index=[0])
    metrics_df.to_csv(f'models/{name_extended}/metrics.csv', index=False)


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
                        help='Path to encoded data file')
    parser.add_argument('--c_param', '-c', type=str, default=50,
                        help='C parameter for SVM')
    parser.add_argument('--kernel', '-k', type=str, default='rbf',
                        help='Kernel type for SVM')
    parser.add_argument('--degree', '-deg', type=int, default=3,
                        help='Degree of polynomial kernel')
    parser.add_argument('--gamma', '-g', type=str, default='scale',
                        help='Gamma parameter for SVM')
    args = parser.parse_args()
    main(args.data_path, args.c_param, args.kernel, args.degree, args.gamma)