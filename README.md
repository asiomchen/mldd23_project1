# Fragments from the known ligands as a source of building blocks for enumeration of new target-focused compound libraries supported by deep learning algoritms
## Authors: Mateusz Iwan, Hubert Rybka, Anton Siomchen
## Table of contents
* [General info](#general-info)
* [Setup](#setup)
* [Interface](#interface)
* [Data sources and tools](#data-sources-and-tools)

## General info
This project is a machine learning model for *de novo* generation of ligands for 5HT1A, 5HT7, d2, beta2 and H1 receptors. A chosen number of [Klekota & Roth](https://pubmed.ncbi.nlm.nih.gov/18784118/) molecular fingerprints are sampled in the following way: first, the incidence and percentage point difference between active/inactive classes of each bit are calcualted based on the fingerprints of known ligands. Afterward, a probabilty of each bit being active in a new fingerprint is calculated as: incidence * (p.p. difference * magnitude (adjustable parameter) + 100) / 100. Then, the probability values for each bit are adjusted to sum up to 1. Finally, a random value is drawn for each bit and compared with the calculated probability to decide if the bit will be 1 or 0.
Next, a recurrent neural network decodes the fingerprints into [SELFIES](https://iopscience.iop.org/article/10.1088/2632-2153/aba947) or SMILES notation. The resulting compounds are filtered based on chosen criteria (QED, max ring size, etc.).

## Setup
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) following the instructions for your operating system.
2. Download this repository. You need to have [Git](https://git-scm.com/) installed.
3. Install environment from the YAML file: `conda env create -n mldd -f environment.yml`

## Interface
1. Activate the environment: `conda activate mldd `
2. Generate fingerprints of potential ligands for a chosen target: `python fp_sampler.py`
  
   Target selection:  
     `-t --target` (str) select target to use: 5ht1a, 5ht7, beta2, d2, h1
     
   Other arguments:  
     `-d --dataset` (str) select the dataset to use for sampling: 100nM, balanced (default: 100nM)  
     `-n --number` (int) number of fingerprints to generate (default: 10 000)  
     `-m --magnitude` (float) increase probability of bits from active class to be selected  
     (recommended range: 1.0 - 5.0, default: 1.0)  
     `-r --random` (bool) add random noise to fingerprints (default: False)  
     `-a --av_bits` (int) average number of active bits in fingerprint (default: 60)
     
     The output file will be saved in ./results directory

     Example use: `python fp_sampler.py -t 5ht1a -n 5000 -r True`
     
     Generates 5000 fingerprints of potentially active 5ht1a ligands with random noise added.

3. (W.I.P.) Convert SMILES of ligands of any target to K&R fingerprints: `python Extract.py --file [filepath.csv]`

   Required arguments:
    `-f --file` (str) path to .csv file containing SMILES of molecules; should be put into ./Smiles2Fp/datasets

   Other arguments:  
    `-s --smiles_col` (str) name of column with SMILES of molecues, default: SMILES  
    `-n --n_jobs` (int) number of processes to use during conversion, default: -1  
    `-o --output` (str) name of .csv file to be saved into ./Smiles2Fp/outputs directory  
    `-k --keys` (str) path to SMARTS keys, default file is included in the directory  
   
5. Generate molecules for fingerprints: `python predict.py`
     
   The script will scan ./results folder for .parquet files generated in previous step. For each one, a new directory in ./results will be generated, containing:
   * .csv file with SMILES of the generated compounds, QED (quantitative estimator od drug-likeness) and highest tanimoto similarity in the training set
   * imgs folder with .png files depicting structures of the generated compounds
   * copy of the configuration file containing model hyperparameters

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

