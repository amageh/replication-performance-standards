"""This module contains auxiliary function which we use in the example notebook."""
import json

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import numpy as np

from grmpy.estimate.estimate_output import calculate_mte
from grmpy.read.read import read


def process_data(df, output_file):
    """This function adds squared and interaction terms to the Cainero data set."""

    # Delete redundant columns\n",
    for key_ in ['newid', 'caseid']:
        del df[key_]

    # Add squared terms
    for key_ in ['mhgc', 'cafqt', 'avurate', 'lurate_17', 'numsibs', 'lavlocwage17']:
        str_ = key_ + 'sq'
        df[str_] = df[key_] ** 2

    # Add interaction terms
    for j in ['pub4', 'lwage5_17', 'lurate_17', 'tuit4c']:
        for i in ['cafqt', 'mhgc', 'numsibs']:
            df[j + i] = df[j] * df[i]

    df.to_pickle(output_file + '.pkl')

    
def plot_est_mte(rslt, file):
    """This function calculates the marginal treatment effect for different quartiles of the
    unobservable V. ased on the calculation results."""

    init_dict = read(file)
    data_frame = pd.read_pickle(init_dict['ESTIMATION']['file'])

    # Define the Quantiles and read in the original results
    quantiles = [0.0001] + np.arange(0.01, 1., 0.01).tolist() + [0.9999]
    mte_ = json.load(open('data/mte_original.json', 'r'))
    mte_original = mte_[1]
    mte_original_d = mte_[0]
    mte_original_u = mte_[2]

    # Calculate the MTE and confidence intervals
    mte = calculate_mte(rslt, init_dict, data_frame, quantiles)
    mte = [i / 4 for i in mte]
    mte_up, mte_d = calculate_cof_int(rslt, init_dict, data_frame, mte, quantiles)

    # Plot both curves
    ax = plt.figure(figsize=(17.5, 10)).add_subplot(111)

    ax.set_ylabel(r"$B^{MTE}$", fontsize=24)
    ax.set_xlabel("$u_D$", fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.plot(quantiles, mte, label='grmpy $B^{MTE}$', color='blue', linewidth=4)
    ax.plot(quantiles, mte_up, color='blue', linestyle=':', linewidth=3)
    ax.plot(quantiles, mte_d, color='blue', linestyle=':', linewidth=3)
    ax.plot(quantiles, mte_original, label='original$B^{MTE}$', color='orange', linewidth=4)
    ax.plot(quantiles, mte_original_d, color='orange', linestyle=':',linewidth=3)
    ax.plot(quantiles, mte_original_u, color='orange', linestyle=':', linewidth=3)
    ax.set_ylim([-0.41, 0.51])
    ax.set_xlim([-0.005, 1.005])

    blue_patch = mpatches.Patch(color='blue', label='original $B^{MTE}$')
    orange_patch = mpatches.Patch(color='orange', label='grmpy $B^{MTE}$')
    plt.legend(handles=[blue_patch, orange_patch],prop={'size': 16})
    plt.show()

    return mte

def calculate_cof_int(rslt, init_dict, data_frame, mte, quantiles):
    """This function calculates the confidence interval of the marginal treatment effect."""

    # Import parameters and inverse hessian matrix
    hess_inv = rslt['AUX']['hess_inv'] / data_frame.shape[0]
    params = rslt['AUX']['x_internal']

    # Distribute parameters
    dist_cov = hess_inv[-4:, -4:]
    param_cov = hess_inv[:46, :46]
    dist_gradients = np.array([params[-4], params[-3], params[-2], params[-1]])

    # Process data
    covariates = init_dict['TREATED']['order']
    x = np.mean(data_frame[covariates]).tolist()
    x_neg = [-i for i in x]
    x += x_neg
    x = np.array(x)

    # Create auxiliary parameters
    part1 = np.dot(x, np.dot(param_cov, x))
    part2 = np.dot(dist_gradients, np.dot(dist_cov, dist_gradients))
    # Prepare two lists for storing the values
    mte_up = []
    mte_d = []

    # Combine all auxiliary parameters and calculate the confidence intervals
    for counter, i in enumerate(quantiles):
        value = part2 * (norm.ppf(i)) ** 2
        aux = np.sqrt(part1 + value) / 4
        mte_up += [mte[counter] + norm.ppf(0.95) * aux]
        mte_d += [mte[counter] - norm.ppf(0.95) * aux]

    return mte_up, mte_d
