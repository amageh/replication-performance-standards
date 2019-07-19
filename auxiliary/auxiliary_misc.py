"""This module contains various additional auxiliary functions that do not belong in plots/tables/predictions which are used in the main notebook."""

import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels as sm


def calculate_bin_frequency(data, bins):
    """
    Calculates the frequency of bins in a dataframe.
    Args:
    data(pd.DataFrame): Dataframe that contains the raw data.
    bins(column): Name of column that contains the bins that should be assessed.
    
    Returns:
    bin_frequency(pd.DataFrame): Dataframe that contains the frequency of each bin in data and and a constant.
    """
    bin_frequency = pd.DataFrame(data[bins].value_counts())
    bin_frequency.reset_index(level=0, inplace=True)
    bin_frequency.rename(columns={"index": "bins", bins: "freq"}, inplace=True)
    bin_frequency = bin_frequency.sort_values(by=['bins'])
    bin_frequency['const'] = 1
    
    return bin_frequency


def create_groups_dict(data, keys, columns):
    """
    Function creates a dictionary containing different subsets of a dataset. Subsets are created using dummies. 
    
    Args:
    data(pd.DataFrame): Dataset that should be split into subsets.
    keys(list): List of keys that should be used in the dataframe.
    columns(list): List of dummy variables in dataset that are used for creating subsets.
    
    Returns:
    groups_dict(dictionary)
    """
    groups_dict = {}
    
    for i in range(len(keys)):
        groups_dict[keys[i]]=data[data[columns[i]] == 1]
            
    return groups_dict