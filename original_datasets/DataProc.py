import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import selfies as sf
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.manifold import TSNE
import sys


def plot_TSNE(df, title="", save_path=None, save=False):
    """
    Plot t-SNE visualization for feature vectors.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing feature vectors.
    title : str, optional
        Title of the plot, by default ""
    save_path : str, optional
        File path to save the plot, by default None
    save : bool, optional
        Whether to save the plot, by default False

    Returns
    -------
    None
    """
    fp_list = []
    for i, row in df.iterrows():
        line = row['fps'].strip('[').strip(']').split(",")
        line = [int(x) for x in line]
        vec = np.zeros(4860, dtype='int')
        for fp in line:
            vec[fp] = 1
        fp_list.append(vec)
    array = np.array(fp_list)
    tsne = TSNE()
    tsne_results = tsne.fit_transform(array)
    tsne_dict = {"X": tsne_results[:, 0], "Y": tsne_results[:, 1]}
    fp_df = pd.concat([df["Class"], pd.DataFrame(tsne_dict)], axis=1)
    sns.set_style('white')
    sns.set_context('talk')
    ax = sns.scatterplot(data=fp_df, x='X', y="Y", hue='Class', legend='auto', marker='.')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.title(f'{title}')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    if save:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def selfies_to_smiles(selfies):
    """
    Convert SELFIES representation to SMILES representation.

    Parameters
    ----------
    selfies : str
        SELFIES representation of a molecule.

    Returns
    -------
    str or None
        SMILES representation of the molecule if conversion is successful, None otherwise.
    """
    try:
        smiles = sf.decoder(selfies)
    except sf.DecoderError:
        print(f"The following SELFIES is not valid: {selfies}")
        pass
    return smiles


def properties_from_smiles(smiles):
    """
    Calculate molecular properties from a SMILES representation.

    Parameters
    ----------
    smiles : str
        SMILES representation of a molecule.

    Returns
    -------
    pandas.Series
        Series containing calculated molecular properties.
    """
    mol = Chem.MolFromSmiles(smiles)
    properties = {
        'Molecular weight': Descriptors.MolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'Number of HBD': Descriptors.NumHDonors(mol),
        'Number of HBA': Descriptors.NumHAcceptors(mol),
        'Number of heteroatoms': Descriptors.NumHeteroatoms(mol),
        'TPSA': Descriptors.TPSA(mol),
        'Number of Aromatic Rings': Descriptors.NumAromaticRings(mol),
        'Fraction of sp3 atoms': Descriptors.FractionCSP3(mol),
    }
    return pd.Series(properties)


def could_be_valid(smiles):
    """
    Check if a SMILES representation is valid.

    Parameters
    ----------
    smiles : str
        SMILES representation of a molecule.

    Returns
    -------
    bool
        True if the SMILES representation could be valid, False otherwise.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


def sparse_to_dense(sparse):
    """
    Convert a sparse vector to a dense vector.

    Parameters
    ----------
    sparse
        Sparse vector.

    Returns
    -------
    list
        Dense vector.
    """
    return np.nonzero(sparse)[0].tolist()


class DataProcessor:
    """
    A class for processing data related to protein activities.

    Parameters
    ----------
    protein : str
        The name of the protein for which data processing is performed.

    y_col : str, optional
        The name of the column representing the target variable (default is 'Ki').

    Attributes
    ----------

    proteins_ : list
        A list of available protein names.

    y_col : str
        The name of the column representing the target variable.

    missing : int or None
        The number of missing values removed during data processing.

    duplicated : int or None
        The number of duplicated rows removed during data processing.

    protein : str
        The name of the protein being processed.

    path : str
        The file path for the protein's data.

    activities : dict
        A dictionary mapping protein names to their corresponding activity threshold values.

    threshold : float
        The activity threshold value for the current protein.

    df : pandas.DataFrame
        The loaded dataset for the current protein.

    Methods
    -------
    load_data()
        Load the dataset for the specified protein.

    remove_missing()
        Remove rows with missing values in the target variable.

    remove_duplicates()
        Remove duplicated rows in the dataset.

    add_classification()
        Add a binary classification column based on the target variable.

    write_cleaned()
        Write the cleaned dataset to a CSV file.

    return_parameters()
        Return a list of data processing parameters.

    write_parquet()
        Write the dataset to a Parquet file.
    """

    def __init__(self, protein, y_col='Ki'):
        sys.path.append("..")
        self.proteins_ = ['5ht1a', '5ht7', 'beta2', 'd2', 'h1']
        self.y_col = y_col
        self.missing = None
        self.duplicated = None
        self.protein = protein
        self.path = f"../original_datasets/klek/{self.protein}_klek.csv"
        self.activities = {
            '5ht1a': 54,
            '5ht7': 89,
            'beta2': 270,
            'd2': 240.1,
            'h1': 501
        }
        self.threshold = self.activities[self.protein]

    def load_data(self):
        self.df = pd.read_csv(self.path)
        self.df[self.y_col] = self.df[self.y_col].astype('float')
        print(f"Loaded data for {self.protein} protein")

    def remove_missing(self):
        print(f'The initial size of dataset: {len(self.df)}')
        missing = self.df[self.y_col].isnull()
        zero_or_neg = self.df[self.y_col] <= 0
        to_remove = pd.Series([a or b for a, b in zip(missing, zero_or_neg)])
        print(f'The percent of rows with missing {self.y_col} values: {to_remove.sum() / len(self.df) * 100:.2f} %')
        self.df = self.df[~to_remove]
        print(f'New size of the dataset: {len(self.df)}')
        self.missing = int(to_remove.sum())

    def remove_duplicates(self):
        print(f'The initial size of dataset: {len(self.df)}')
        duplicates = self.df.duplicated(keep='first')
        print(f'The percent of duplicated rows: {duplicates.sum() / len(self.df) * 100:.2f} %')
        self.df = self.df[~duplicates]
        print(f'New size of the dataset: {len(self.df)}')
        self.duplicated = int(duplicates.sum())

    def add_classification(self):
        classes = [1 if x < 100 else 0 for x in self.df[self.y_col]]
        self.df.insert(1, "Class", classes)
        print(f'The percent of compounds classified as active is {self.df["Class"].sum() / len(self.df) * 100:.2f} %')

    def write_cleaned(self):
        write_path = '../original_datasets/klek_clean/' + self.protein + '_klek_100nM.csv'
        self.df.to_csv(path_or_buf=write_path, sep=',', index=False)
        print(f'Cleaned file saved at {write_path}')

    def return_parameters(self):
        parameters = [self.missing, self.duplicated]
        return parameters

    def write_parquet(self):
        path = '..' + self.path.strip('.csv') + '_balanced.parquet'
        self.df.to_parquet(path)


class DataAnalyser:
    """
    A class for analysing and visualizing data related to protein activities.

    Parameters
    ----------
    protein : str
        The name of the protein for which data analysis is performed.

    dtype : str or list of str, optional
        The type(s) of data to be analyzed (one of ['smiles', 'klek_balanced', 'klek_100nM', 'artificial']).

    smiles_col : str, optional
        The name of the column containing SMILES strings (default is 'SMILES').

    Attributes
    ----------
    protein : str
        The name of the protein being analyzed.

    proteins_ : list
        A list of available protein names.

    dtype : str or list of str
        The type(s) of data being analyzed.

    smiles_col : str
        The name of the column containing SMILES strings.

    data_path : str
        The file path for the selected data type.

    df : pandas.DataFrame
        The loaded dataset for the current protein and data type.

    disc_properties : list
        A list of discrete properties to consider during analysis.

    cont_properties : list
        A list of continuous properties to consider during analysis.

    Methods
    -------
    __call__()
        Returns the loaded dataset.

    add_properties()
        Add additional properties to the dataset based on SMILES strings.

    plot_distribution(column, save=False)
        Plot the distribution of a specified column in the dataset.

    TSNE_all(title="", save_path="", save=False)
        Perform t-SNE analysis on all available proteins' data and plot the results.
    """

    def __init__(self, protein, dtype=None, smiles_col='SMILES'):
        if dtype is None:
            dtype = ['smiles', 'klek_balanced', 'klek_100nM', 'artificial']
        sys.path.append('..')
        self.protein = protein
        self.proteins_ = ['5ht1a', '5ht7', 'beta2', 'd2', 'h1']
        self.dtype = dtype
        self.smiles_col = smiles_col
        if dtype == 'smiles':
            self.data_path = f"./smiles/{protein}_{dtype}.csv"
        elif dtype == 'klek_balanced':
            self.data_path = f"./klek_clean/{protein}_{dtype}.csv"
        elif dtype == 'klek_100nM':
            self.data_path = f"./klek_clean/{protein}_{dtype}.csv"
        elif dtype == 'artificial':
            self.data_path = f"./artificial/{protein}_klek_{dtype}.csv"
        else:
            print("Data not found")
        self.df = pd.read_csv(self.data_path, sep=',')
        self.disc_properties = ['Number of HBD', 'Number of HBA', 'Number of heteroatoms', 'Number of Aromatic Rings']
        self.cont_properties = ['LogP', 'Molecular weight', 'TPSA', 'Fraction of sp3 atoms']

    def __call__(self):
        return self.df

    def add_properties(self):
        self.df = pd.concat([self.df, self.df[self.smiles_col].apply(properties_from_smiles)], axis=1)

    def plot_distribution(self, column, save=False, av=False):
        if self.dtype in ['klek_balanced', 'klek_100nM']:
            self.df['pKi'] = self.df['Ki'].apply(np.log10)
            class_0_mean = np.mean(self.df[self.df['Class'] == 0][column])
            class_1_mean = np.mean(self.df[self.df['Class'] == 1][column])
            class_0_std = np.mean(self.df[self.df['Class'] == 0][column])
            class_1_std = np.mean(self.df[self.df['Class'] == 1][column])
            self.df = self.df[np.logical_and(self.df[column] > (class_0_mean - 3 * class_0_std),
                                             self.df[column] < (class_0_mean + 3 * class_0_std))]
            self.df = self.df[np.logical_and(self.df[column] > (class_1_mean - 2 * class_1_std),
                                             self.df[column] < (class_1_mean + 2 * class_1_std))]
        sns.set_style('white')
        sns.set_context('talk')
        if column in self.cont_properties:
            ax = sns.histplot(data=self.df, x=column, kde=True, hue='Class')  # hue="Class"
            plt.title(f"Distribution of {column} for known ligands of {self.protein.upper()}")
            plt.ylabel("Count")
        elif column in self.disc_properties:
            ax = sns.countplot(data=self.df, x=column, hue='Class')  # hue="Class"
            plt.title(f"{column} for known ligands of {self.protein.upper()}")
            plt.xticks(ticks=plt.xticks()[0], labels=[int(x) for x in plt.xticks()[0]])
            plt.ylabel("Count")
        else:
            print(f"Column {column} not found")
        if self.dtype in ['klek_balanced', 'klek_100nM'] and av:
            ax.axvline(class_0_mean, 0, 1, color='blue')
            ax.axvline(class_1_mean, 0, 1, color='orange')
        if save:
            if self.dtype == 'artificial':
                save_path = f"./artificial/{self.protein}_artificial_{column}.png"
            else:
                save_path = f"./figures_{self.dtype}/{self.protein}_{self.dtype}/{self.protein}_{column}.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.clf()
            
    def plot_distribution_no_legend(self, column, save=False, av=False):
        if self.dtype in ['klek_balanced', 'klek_100nM']:
            self.df['pKi'] = self.df['Ki'].apply(np.log10)
            class_0_mean = np.mean(self.df[self.df['Class'] == 0][column])
            class_1_mean = np.mean(self.df[self.df['Class'] == 1][column])
            class_0_std = np.mean(self.df[self.df['Class'] == 0][column])
            class_1_std = np.mean(self.df[self.df['Class'] == 1][column])
            self.df = self.df[np.logical_and(self.df[column] > (class_0_mean - 3 * class_0_std),
                                             self.df[column] < (class_0_mean + 3 * class_0_std))]
            self.df = self.df[np.logical_and(self.df[column] > (class_1_mean - 2 * class_1_std),
                                             self.df[column] < (class_1_mean + 2 * class_1_std))]
        sns.set_style('white')
        sns.set_context('talk')
        if column in self.cont_properties:
            ax = sns.histplot(data=self.df, x=column, kde=True, hue='Class')  # hue="Class"
            #plt.title(f"Distribution of {column} for generated ligands of {self.protein.upper()}")
            plt.ylabel("")
            plt.xlabel("")
            ax.set(xlabel=None)
        elif column in self.disc_properties:
            ax = sns.countplot(data=self.df, x=column, hue='Class')  # hue="Class"
            #plt.title(f"{column} for generated ligands of {self.protein.upper()}")
            plt.xticks(ticks=plt.xticks()[0], labels=[int(x) for x in plt.xticks()[0]])
            plt.ylabel("")
            plt.xlabel("")
            ax.set(xlabel=None)
        else:
            print(f"Column {column} not found")
        if self.dtype in ['klek_balanced', 'klek_100nM'] and av:
            ax.axvline(class_0_mean, 0, 1, color='blue')
            ax.axvline(class_1_mean, 0, 1, color='orange')
        if save:
            if self.dtype == 'artificial':
                save_path = f"./artificial/{self.protein}_artificial_{column}.png"
            else:
                save_path = f"./figures_{self.dtype}/{self.protein}_{self.dtype}/{self.protein}_{column}_no_legend.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.clf()
            
    def TSNE_all(self, title="", save_path="", save=False):
        data_paths = [f"./klek_clean/{x}_{self.dtype}.csv" for x in self.proteins_]
        df_list = []
        for path, protein in zip(data_paths, self.proteins_):
            df = pd.read_csv(path, sep=',')
            df = df[df['Class'] == 1]
            df = df.drop("Class", axis=1)
            df["Class"] = protein
            df_list.append(df)
        self.combined_df = pd.concat([df for df in df_list], axis=0, ignore_index=True)
        plot_TSNE(self.combined_df, title=title, save_path=save_path, save=save)


class FPAnalyser:
    """
    Class for analyzing frequency differences of bits in different proteins.

    Parameters
    ----------
    df_active : pandas.DataFrame
        DataFrame containing active counts.
    df_inactive : pandas.DataFrame
        DataFrame containing inactive counts.

    Attributes
    ----------
    df : pandas.DataFrame
        Merged DataFrame of active and inactive counts.
    proteins : list
        List of proteins.
    sizes : dict
        Dictionary containing sizes of each protein dataset.

    Methods
    -------
    calculate_difference()
        Calculates the difference between active and inactive counts for each protein.
    calculate_percentage()
        Calculates the percentage difference for each protein.
    save_frequency()
        Saves the frequency DataFrame for each protein as a CSV file.
    plot_frequencies(save=False)
        Plots the frequency differences for each protein.
    """

    def __init__(self):
        self.df_active = pd.read_csv(f"./counts_active.csv", sep=',')
        self.df_inactive = pd.read_csv(f"./counts_inactive.csv", sep=',')
        self.df = self.df_active.merge(self.df_inactive, on=['KEYS', 'SMARTS'], how='inner',
                                       suffixes=('_active', '_inactive'))
        self.proteins = ['5ht1a', '5ht7', 'beta2', 'd2', 'h1']
        self.sizes = {
            '5ht1a': 5250,
            '5ht7': 2963,
            'beta2': 782,
            'd2': 10170,
            'h1': 1691
        }

    def calculate_difference(self):
        for protein in self.proteins:
            print(self.df.head())
            col_act = f"{protein}_active"
            col_inact = f"{protein}_inactive"
            self.df[f"{protein}_difference"] = self.df[col_act] - self.df[col_inact]

    def calculate_percentage(self):
        for protein in self.proteins:
            dataset_size = self.sizes[protein]
            self.df[f"{protein}_percentage"] = self.df[f"{protein}_difference"] / dataset_size * 100
            self.df[f"{protein}_percentage"] = self.df[f"{protein}_percentage"].round(3)

    def save_frequency(self):
        for protein in self.proteins:
            self.save_df = pd.concat([self.df[f"{protein}_percentage"], self.df["KEYS"], self.df["SMARTS"]], axis=1)
            #self.save_df = self.save_df.sort_values(by=f"{protein}_percentage", ascending=False)
            save_path = f"./fp_frequency_100nM/{protein}_frequency_not_sorted.csv"
            self.save_df.to_csv(save_path, sep=',', header=True, index=False)

    def plot_frequencies(self, save=False):
        self.plot_df = self.df.iloc[:, [0, 6, -1, -2, -3, -4, -5]]
        for protein in self.proteins:
            df = self.plot_df.loc[:, ['KEYS', 'SMARTS', f"{protein}_percentage"]]
            df = df[np.logical_or(df[f'{protein}_percentage'] > 3, df[f'{protein}_percentage'] < -3)]
            df = df.sort_values(by=f"{protein}_percentage", ascending=False)
            selected_df = df.iloc[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1], :]
            print(selected_df)
            sns.set_style('white')
            sns.set_context('talk')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=selected_df, y='KEYS', x=f"{protein}_percentage", orient='h', ax=ax)
            ax.bar_label(ax.containers[-1], fmt='%.2f', label_type='center')
            plt.ylabel("")
            plt.xlabel("P.p. difference between active and inactive")
            plt.title(f"{protein}".upper())
            if save:
                save_path = f"./fp_frequency_100nM/{protein}_frequency.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
    def plot_abs_frequencies(self, save=False):
        self.plot_df = self.df.iloc[:, [0, 6, -1, -2, -3, -4, -5]]
        for protein in self.proteins:
            df = self.plot_df.loc[:, ['KEYS', 'SMARTS', f"{protein}_percentage"]]
            #df = df[np.logical_or(df[f'{protein}_percentage'] > 3, df[f'{protein}_percentage'] < -3)]
            df[f'{protein}_percentage_abs'] = df[f'{protein}_percentage'].apply(np.absolute)
            df = df.sort_values(by=f"{protein}_percentage_abs", ascending=False)
            selected_df = df.iloc[:10, :]
            print(selected_df.head())
            sns.set_style('white')
            sns.set_context('talk')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=selected_df, y='KEYS', x=f"{protein}_percentage", orient='h', ax=ax)
            ax.bar_label(ax.containers[-1], fmt='%.2f', label_type='center')
            plt.ylabel("")
            plt.xlabel("P.p. difference between active and inactive")
            plt.title(f"{protein}".upper())
            if save:
                save_path = f"./fp_frequency_100nM/{protein}_frequency_abs.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
class FPImbalancedAnalyser:
    """
    Class for analyzing frequency differences of bits in different proteins.

    Parameters
    ----------
    df_active : pandas.DataFrame
        DataFrame containing active counts.
    df_inactive : pandas.DataFrame
        DataFrame containing inactive counts.

    Attributes
    ----------
    df : pandas.DataFrame
        Merged DataFrame of active and inactive counts.
    proteins : list
        List of proteins.
    sizes : dict
        Dictionary containing sizes of each protein dataset.

    Methods
    -------
    calculate_difference()
        Calculates the difference between active and inactive counts for each protein.
    calculate_percentage()
        Calculates the percentage difference for each protein.
    save_frequency()
        Saves the frequency DataFrame for each protein as a CSV file.
    plot_frequencies(save=False)
        Plots the frequency differences for each protein.
    """

    def __init__(self):
        self.df_active = pd.read_csv(f"./counts_active.csv", sep=',')
        self.df_inactive = pd.read_csv(f"./counts_inactive.csv", sep=',')
        self.proteins = ['5ht1a', '5ht7', 'beta2', 'd2', 'h1']
        self.sizes = {
            '5ht1a': 5250,
            '5ht7': 2963,
            'beta2': 782,
            'd2': 10170,
            'h1': 1691,
            '5ht1a_active': 3043,
            '5ht7_active': 1526,
            'beta2_active': 331,
            'd2_active': 3713,
            'h1_active': 641,
            '5ht1a_inactive': 2207,
            '5ht7_inactive': 1437,
            'beta2_inactive': 451,
            'd2_inactive': 6457,
            'h1_inactive': 1050,
        }

    def calcualte_percentage_difference(self):
        for protein in self.proteins:
            self.df_active[f'{protein}_percentage'] = self.df_active[protein] / self.sizes[f'{protein}_active'] * 100
            self.df_inactive[f'{protein}_percentage'] = self.df_inactive[protein] / self.sizes[f'{protein}_inactive'] * 100
        self.df = self.df_active.merge(self.df_inactive, on=['KEYS', 'SMARTS'], how='inner', suffixes=('_active', '_inactive'))
        for protein in self.proteins:
            self.df[f"{protein}_percentage"] = self.df[f"{protein}_percentage_active"] - self.df[f"{protein}_percentage_inactive"]
        print(self.df.head())

    def save_frequency(self):
        for protein in self.proteins:
            self.save_df = pd.concat([self.df[f"{protein}_percentage"], self.df["KEYS"], self.df["SMARTS"]], axis=1)
            #self.save_df = self.save_df.sort_values(by=f"{protein}_percentage", ascending=False)
            save_path = f"./fp_frequency_100nM/{protein}_frequency_imbalanced.csv"
            self.save_df.to_csv(save_path, sep=',', header=True, index=False)

    def plot_frequencies(self, save=False):
        self.plot_df = self.df.iloc[:, [0, 6, -1, -2, -3, -4, -5]]
        for protein in self.proteins:
            df = self.plot_df.loc[:, ['KEYS', 'SMARTS', f"{protein}_percentage"]]
            df = df[np.logical_or(df[f'{protein}_percentage'] > 3, df[f'{protein}_percentage'] < -3)]
            df = df.sort_values(by=f"{protein}_percentage", ascending=False)
            selected_df = df.iloc[[0, 1, 2, 3, 4, -5, -4, -3, -2, -1], :]
            print(selected_df)
            sns.set_style('white')
            sns.set_context('talk')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=selected_df, y='KEYS', x=f"{protein}_percentage", orient='h', ax=ax)
            ax.bar_label(ax.containers[-1], fmt='%.2f', label_type='center')
            plt.ylabel("")
            plt.xlabel("P.p. difference between active and inactive")
            plt.title(f"{protein}".upper())
            if save:
                save_path = f"./fp_frequency_100nM/{protein}_frequency.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
            
    def plot_abs_frequencies(self, save=False):
        self.plot_df = self.df.iloc[:, [0, 6, -1, -2, -3, -4, -5]]
        for protein in self.proteins:
            df = self.plot_df.loc[:, ['KEYS', 'SMARTS', f"{protein}_percentage"]]
            #df = df[np.logical_or(df[f'{protein}_percentage'] > 3, df[f'{protein}_percentage'] < -3)]
            df[f'{protein}_percentage_abs'] = df[f'{protein}_percentage'].apply(np.absolute)
            df = df.sort_values(by=f"{protein}_percentage_abs", ascending=False)
            selected_df = df.iloc[:10, :]
            print(selected_df.head())
            sns.set_style('white')
            sns.set_context('talk')
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.barplot(data=selected_df, y='KEYS', x=f"{protein}_percentage", orient='h', ax=ax)
            ax.bar_label(ax.containers[-1], fmt='%.2f', label_type='center')
            plt.ylabel("")
            plt.xlabel("P.p. difference between active and inactive")
            plt.title(f"{protein}".upper())
            if save:
                save_path = f"./fp_frequency_100nM/{protein}_frequency_abs.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

