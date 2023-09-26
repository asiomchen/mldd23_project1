# Fragments from the known ligands as a source of building blocks for enumeration of new target-focused compound libraries supported by deep learning algoritms
## Authors: Mateusz Iwan, Hubert Rybka, Anton Siomchen
## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Interface](#interface)
* [Data sources and tools](#data-sources-and-tools)

## General info
(W.I.P.)

## Setup
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) following the instructions for your operating system.
2. Download this repository. You need to have [Git](https://git-scm.com/) installed.
3. Install environment from the YAML file: `conda env create -n mldd -f environment.yml`

## Usage
1. Activate the environment: `conda activate mldd `

Prepare a dataset (preferably over 1,000 compunds). 

Put the data into pandas.DataFrame object. The dataframe must contain the following columns:
'smiles' - SMILES strings of known ligands.
'fps' - Klekota&Roth or Morgan (radius=2) fingerprints of the ligands.
    The fingerprints have to be saved as ordinary pthon lists, in dense format (a list of ints
    designating the indices of active bits in the original fingerprint). 
    For help, see src.utils.finger.sparse2dense().
'activity' - Activity class (True, False). By default, we define active compounds as those having
    Ki value <= 100nM and inactive as those of Ki > 100nM.
Save dataframe to .parquet file using:

import pandas as pd
df = pd.DataFrame(columns=['smiles', 'fps', 'activity'])

# ... load data into the dataframe
name = '5ht7_ECFP' # example name for the dataset

df.to_parquet(f'data/activity_data/{name}.parquet', index=False)

2. Encode the dataset into latent space, and train the activity predictor.
Use the following command:
  
    python train_clf.py

and provide the path to the dataset file using the -d flag.
Other parameters are optional and can be set using the command line arguments.
  
--data_path DATA_PATH, -d DATA_PATH
                        Path to data file (prepared as described in above)
  --c_param C_PARAM, -c C_PARAM
                        C parameter for SVM (default: 50)
                        Commonly a float in the range [0.01, 100]
  --kernel KERNEL, -k KERNEL
                        Kernel type for SVM
                        One of: linear, poly, rbf, sigmoid
  --degree DEGREE, -deg DEGREE
                        Degree of polynomial kernel
                        Ignored by other kernels
  --gamma GAMMA, -g GAMMA
                        Gamma parameter for SVM

Now a file with the trained model should be saved in the 'models' directory. Locate the directory,
and save path to a model.pkl file created by the training script inside.

It should look like this:
    
   models/name_of_the_model/model.pkl

3. Bayesian search on the latent space

The trained activity predictor can be used to perform bayesian search on the latent space
in order to identify latent representations of potential novel ligands.
To perform bayesian search on the latent space, use the following command:

    python bayesian_search.py

and provide the path to the model file using the -m flag.

    python bayesian_search.py -m models/name_of_the_model/model.pkl

Other parameters can be set using the command line arguments:

  -m MODEL_PATH, --model_path MODEL_PATH
                        Path to the saved activity predictor model
  -n N_SAMPLES, --n_samples N_SAMPLES
                        Number of samples to generate
  -p INIT_POINTS, --init_points INIT_POINTS
                        Number of initial points to sample
  -i N_ITER, --n_iter N_ITER
                        Number of iterations to perform
  -b BOUNDS, --bounds BOUNDS
                        Bounds for the latent space search
  -v VERBOSITY, --verbosity VERBOSITY
                        Verbosity: 0 - silent, 1 - normal, 2 - verbose
  -w N_WORKERS, --n_workers N_WORKERS
                        Number of workers to use. Default: -1 (all available CPU cores)

Results of the search will be saved in the 'results' directory.

Directory 'SVC_{timestamp}' will be created on /results, containing the following files:
    latent_vectors.csv - latent vectors found by the search
    info.txt - information about the search



## Data sources and tools
### Data sources:
* CHEMBL32
  ~1M subset of compounds with molecular mass in range of 200-450 Da, with no RO5 violations.
* ZINC-250k

  Datasets are available on [dropbox](https://www.dropbox.com/sh/7sop2qzz4n38o06/AAA1QXeD3cXO__02RnmsVV-Aa?dl=0) in .parquet format
### Python packages:
* machine learning: pytorch
* cheminformatics: selfies
* other: numpy, pandas

