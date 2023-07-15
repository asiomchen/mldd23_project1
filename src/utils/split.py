from rdkit.Chem.Scaffolds import MurckoScaffold

# scaffold split

def get_scaffold(smiles, include_chirality=False):
    
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=include_chirality)
    return scaffold

def scaffold_split(df, frac_train):

    print('grouping compounds by murcko scaffold...')
    # create dict of the form {scaffold_i: [idx1, idx....]}
    
    all_scaffolds = {}
    for i, smiles in enumerate(tqdm(df.smiles)):
        scaffold = get_scaffold(smiles, include_chirality=True)
        if scaffold not in all_scaffolds:
            all_scaffolds[scaffold] = [i]
        else:
            all_scaffolds[scaffold].append(i)

    # sort from largest to smallest sets
    print('sorting keys...')
    all_scaffolds = {key: sorted(value) for key, value in all_scaffolds.items()}
    all_scaffold_sets = [
        scaffold_set for (scaffold, scaffold_set) in sorted(
            all_scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]), reverse=True)
    ]

    # get train, valid indices
    cutoff = frac_train * len(df)

    train_idx, valid_idx = [], []
    
    print('splitting...')
    for scaffold_set in tqdm(all_scaffold_sets):
        if len(train_idx) + len(scaffold_set) > cutoff:
            valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    assert len(set(train_idx).intersection(set(valid_idx))) == 0


    train_df = df[df.index.isin(train_idx)]
    val_df = df[df.index.isin(valid_idx)]

    return train_df, val_df
