"""This module contains auxiliary functions for the creation of tables in the main notebook."""

import json

import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels as sm


def estimate_RDD_multiple_outcomes(data, outcomes, regressors):
    """ Regression analysis with standard errors clustered on GPA, on probation cutoff for multiple outcomes contained in ONE dataframe.

    Args:
    data(pd.DataFrame): Dataset containing all data (must contain 'clustervar', 'gpalscutoff', & 'const')
    outcomes(list): List of all outcomes (must correspond to column names in dataset)
    regressors(list): List of all regressors(must correspond to column names in dataset)

    Returns:
    table(pd.DataFrame): Dataframe containing the coefficient, pvalue and standard error for the dummy 
                        'GPA below cutoff' and the constant.
    """
    table = pd.DataFrame({'GPA below cutoff (1)': [], 'P-Value (1)': [], 'Std.err (1)': [],
                          'Intercept (0)': [], 'P-Value (0)': [], 'Std.err (0)': [],
                          'Observations': []})

    table['outcomes'] = outcomes
    table = table.set_index('outcomes')

    for outcome in outcomes:
        data = data.dropna(subset=[outcome])
        model = sm.regression.linear_model.OLS(
            data[outcome], data[regressors], hasconst=True)
        result = model.fit(cov_type='cluster', cov_kwds={
                           'groups': data['clustervar']})
        outputs = [result.params['gpalscutoff'], result.pvalues['gpalscutoff'], result.bse['gpalscutoff'],
                   result.params['const'], result.pvalues[
                       'const'], result.bse['const'],
                   len(data[outcome])]
        table.loc[outcome] = outputs

    table = table.round(3)
    return table


def estimate_RDD_multiple_datasets(dictionary, keys, outcome, regressors):
    """ Regression analysis for ONE outcome with standard errors on GPA and with dicionary of MANY dataframes as input.

    Args:
    dictionary(pd.dict): Dictionary conatining datasets ( datasets must contain 'clustervar', 'gpalscutoff', & 'const')
    outcome(string): Name of outcome variable (must correspond to column name in datasets )
    regressors(list): List of all regressors(must correspond to column names in datasets)
    
    Returns:
    table(pd.DataFrame): Dataframe containing the coefficient, pvalue and standard error for the dummy 
                        'GPA below cutoff' and the constant.
    """
    table = pd.DataFrame({'GPA below cutoff (1)': [], 'P-Value (1)': [], 'Std.err (1)': [],
                          'Intercept (0)': [], 'P-Value (0)': [], 'Std.err (0)': [],
                          'Observations': []})

    table['groups'] = keys
    table = table.set_index('groups')

    for key in keys:
        data = dictionary[key]
        data = data.dropna(subset=[outcome])
        model = sm.regression.linear_model.OLS(
            data[outcome], data[regressors], hasconst=True)
        result = model.fit(cov_type='cluster', cov_kwds={
                           'groups': data['clustervar']})
        outputs = [result.params['gpalscutoff'], result.pvalues['gpalscutoff'], result.bse['gpalscutoff'],
                   result.params['const'], result.pvalues['const'], result.bse['const'],len(data[outcome])]
        table.loc[key] = outputs

    table = table.round(3)
    return table
