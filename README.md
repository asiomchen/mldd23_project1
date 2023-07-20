# Fragments from the known ligands as a source of building blocks for enumeration of new target-focused compound libraries supported by deep learning algoritms
## Authors: Mateusz Iwan, Hubert Rybka, Anton Siomchen
## Table of contents
* Genertal info
* Setup
* User manual
* Data sources and tools

## General info
This project is a machine learning model for *de novo* generation of ligands for 5HT1A, 5HT7, d2, beta2 and H1 receptors. A chosen number of [Klekota & Roth](https://pubmed.ncbi.nlm.nih.gov/18784118/) molecular fingerprints are sampled from the distribution of active compounds' fingerprints. Next, a recurrent neural network decodes the fingerprints into [SELFIES](https://iopscience.iop.org/article/10.1088/2632-2153/aba947) or SMILES notation. The resulting compounds are filtered based on chosed criteria (QED, max ring size, etc.).

## Setup
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) following the instructions for your operating system.
2. Download this repository. You need to have [Git](https://git-scm.com/) installed.
3. Install environment from the YAML file: `conda env create -n mldd -f environment.yml`
4. Activate the enviroment: `conda activate mldd`

## User manual
TODO

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
