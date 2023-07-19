# Fragments from the known ligands as a source of building blocks for enumeration of new target-focused compound libraries supported by deep learning algoritms
## Authors: Mateusz Iwan, Hubert Rybka, Anton Siomchen
![abstract](https://github.com/asiomchen/mldd23_project1/assets/126616541/f1b73205-6046-49c1-a03b-a0cac37861f5)
## Table of contents
* Genertal info
* Setup
* Data sources and tools

## General info
This project is a

## Setup
1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) following the instructions for your operating system.
2. Download this repository.
   
   You need to have [Git](https://git-scm.com/) installed.
3. Install environment from the YAML file: `conda env create -f environment.yml` (or `conda env create -f environment-gpu.yml` for the GPU version).

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
