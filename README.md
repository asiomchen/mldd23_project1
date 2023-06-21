# Fragments from the known ligands as a source of building blocks for enumeration of new target-focused compound libraries supported by deep learning algoritms
## Authors: Mateusz Iwan, Hubert Rybka, Anton Siomchen
![abstract](https://github.com/asiomchen/mldd23_project1/assets/126616541/f1b73205-6046-49c1-a03b-a0cac37861f5)
##  1. Data sources and tools used
### 1.1.  Data sources:
* CHEMBL32
  ~1M subset of compounds with molecular mass in range of 200-450 Da, with no RO5 violations.
* ZINC-250k

Datasets are available on [dropbox](https://www.dropbox.com/sh/7sop2qzz4n38o06/AAA1QXeD3cXO__02RnmsVV-Aa?dl=0) in .parquet format
### 1.2 Python packages:
* machine learning: pytorch
* cheminformatics: smiles
* other: numpy, pandas, matplotlib, tqdm, jupyter, ipython, re
## 2. Problems to solve:
- Problem 1 - The structure of generated molecules must be chemically correct.
- Problem 2 - Generated molecules should be obtainable through the means of simple organic synmthesis.
- Problem 3 - Generated molecules should not contain functional groups which are known to be unstable at physiological conditions.
- Problem 4 - Generated molecules should be "drug-like" and non-toxic.

## 3. Additional info:
(empty)
