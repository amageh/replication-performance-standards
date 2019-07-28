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



def create_table1(data):
    variables = data[['hsgrade_pct', 'totcredits_year1', 'age_at_entry', 'male', 'english', 'bpl_north_america',
                        'loc_campus1', 'loc_campus2', 'loc_campus3', 'dist_from_cut', 'probation_year1', 'probation_ever',
                        'left_school', 'year2_dist_from_cut', 'suspended_ever', 'gradin4', 'gradin5', 'gradin6']]

    table1 = pd.DataFrame()
    table1['Mean'] = variables.mean()
    table1['Standard Deviation'] = variables.std()
    table1 = table1.astype(float).round(2)
    table1['Description'] = ["High School Grade Percentile", "Credits attempted first year", "Age at entry",
                             "Male", "English is first language", "Born in North America",
                             "At Campus 1", "At Campus 2", "At Campus 3",
                             "Distance from cutoff in first year", "On probation after first year", " Ever on acad. probation",
                             "Left Uni after 1st evaluation", "Distance from cutoff at next evaluation", "Ever suspended",
                             "Graduated by year  4", "Graduated by year  5", "Graduated by year  6"]

    table1.loc[0:9, 'Type'] = "Characteristics"
    table1.loc[9:, 'Type'] = "Outcomes"
    
    return table1
