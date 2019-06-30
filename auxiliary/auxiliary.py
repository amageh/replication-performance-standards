"""This module contains auxiliary functions which we use in the main notebook."""
import json

import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels as sm


# ************* RDD TABLES ***************************************************************

def pvalue_5percent_red(val):
    """
    Formatting Function: Takes a scalar and returns a string with the css property `'color: red'` 
    for values below 0.05, black otherwise.
    """
    color = 'red' if val < 0.05 else 'black'
    return 'color: %s' % color




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
    table = pd.DataFrame({ 'GPA below cutoff (1)': [], 'P-Value (1)':[], 'Std.err (1)':[], 
                       'Intercept (0)':[], 'P-Value (0)':[], 'Std.err (0)':[], 
                       'Observations':[]})
    
    table['outcomes'] = outcomes
    table = table.set_index('outcomes')

    for outcome in outcomes:
        data = data.dropna(subset=[outcome])
        model = sm.regression.linear_model.OLS(data[outcome], data[regressors], hasconst=True)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data['clustervar']})
        outputs = [result.params['gpalscutoff'], result.pvalues['gpalscutoff'], result.bse['gpalscutoff'], 
                   result.params['const'], result.pvalues['const'], result.bse['const'], 
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
    table = pd.DataFrame({ 'GPA below cutoff (1)': [], 'P-Value (1)':[], 'Std.err (1)':[], 
                       'Intercept (0)':[], 'P-Value (0)':[], 'Std.err (0)':[], 
                       'Observations':[]})
    
    table['groups'] = keys
    table = table.set_index('groups')
    
    for key in keys:
        data = dictionary[key]
        data = data.dropna(subset=[outcome])
        model = sm.regression.linear_model.OLS(data[outcome], data[regressors], hasconst=True)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data['clustervar']})
        outputs = [result.params['gpalscutoff'], result.pvalues['gpalscutoff'], result.bse['gpalscutoff'], 
                   result.params['const'], result.pvalues['const'], result.bse['const'], 
                   len(data[outcome])]   
        table.loc[key] = outputs

    table = table.round(3)
    return table



def create_table_6(groups_dict, groups_labels, outcome, regressors):
    table = table_template = pd.DataFrame({ 'GPA below cutoff (1)': [], 'P-Value (1)':[], 'Std.err (1)':[], 
                       'Intercept (0)':[], 'P-Value (0)':[], 'Std.err (0)':[], 
                       'Observations':[]})
    table['groups'] = groups_labels
    table = table.set_index('groups')
        
    for group in groups_labels:
        data = groups_dict[group]
        data = data.dropna(subset=[outcome])
        
        model = sm.regression.linear_model.OLS(data[outcome], data[regressors], hasconst=True)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': data['clustervar']})
        outputs = [result.params['gpalscutoff'], result.pvalues['gpalscutoff'], result.bse['gpalscutoff'], 
                   result.params['const'], result.pvalues['const'], result.bse['const'], 
                   len(data[outcome])]   
        table.loc[group] = outputs
    table = table.round(3)
    return table

# ************* RDD PLOTS ****************************************************************

def create_predictions(data, outcome, regressors):
    
    steps = np.arange(-1.2,1.25,0.05)
    predictions_df = pd.DataFrame([])
    # Ensure there are no missings in the outcome variable
    data = data.dropna(subset=[outcome])   
    # Loop through bins or 'steps'.
    for step in steps:  
        #df =  df.dropna(subset=['year2_dist_from_cut'])
        df = data[(data.dist_from_cut >= (step - 0.6)) & (data.dist_from_cut <= (step + 0.6))]
        # Run regression for with all values in the range specified above.
        model = sm.regression.linear_model.OLS(df[outcome], df[regressors], hasconst=True)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': df['clustervar']})

            # Fill in row for each step in the prediction datframe. 
        predictions_df.loc[step,'dist_from_cut'] = step
        if step < 0: 
            predictions_df.loc[step,'gpalscutoff'] = 1 
        else:
            predictions_df.loc[step,'gpalscutoff'] = 0 

        predictions_df.loc[step,'gpaXgpalscutoff']= (predictions_df.loc[step,'dist_from_cut'])*predictions_df.loc[step,'gpalscutoff']
        predictions_df.loc[step,'gpaXgpagrcutoff']= (predictions_df.loc[step,'dist_from_cut'])*(1-predictions_df.loc[step,'gpalscutoff'])
        predictions_df.loc[step,'const'] = 1

        # Make prediction for each step based on regression of each step and save value in the prediction dataframe.
        predictions_df.loc[step,'prediction'] = result.predict(exog=[[
                                                    predictions_df.loc[step,'const'], 
                                                    predictions_df.loc[step,'gpalscutoff'],
                                                    predictions_df.loc[step,'gpaXgpalscutoff'], 
                                                    predictions_df.loc[step,'gpaXgpagrcutoff']
                                                ]])

    predictions_df.round(4)
            
    return predictions_df


def create_fig3_predictions(groups_dict, regressors):
    
    predictions_groups_dict = {}
    # Loop through groups:
    for group in groups_dict:    

        steps = np.arange(-1.2,1.25,0.05)
        predictions_df = pd.DataFrame([])
        
        # Loop through bins or 'steps'.
        for step in steps:  
            # Select dataframe from the dictionary.
            df = groups_dict[group][(groups_dict[group].dist_from_cut >= (step - 0.6)) & (groups_dict[group].dist_from_cut <= (step+0.6))]
            # Run regression for with all values in the range specified above.
            model = sm.regression.linear_model.OLS(df['left_school'], df[regressors], hasconst=True)
            result = model.fit(cov_type='cluster', cov_kwds={'groups': df['clustervar']})
            
            # Fill in row for each step in the prediction datframe. 
            predictions_df.loc[step,'dist_from_cut'] = step
            if step < 0: 
                predictions_df.loc[step,'gpalscutoff'] = 1 
            else:
                predictions_df.loc[step,'gpalscutoff'] = 0 

            predictions_df.loc[step,'gpaXgpalscutoff']= (predictions_df.loc[step,'dist_from_cut'])*predictions_df.loc[step,'gpalscutoff']
            predictions_df.loc[step,'gpaXgpagrcutoff']= (predictions_df.loc[step,'dist_from_cut'])*(
                                                         1-predictions_df.loc[step,'gpalscutoff']
                                                         )
            predictions_df.loc[step,'const'] = 1
            
            # Make prediction for each step based on regression of each step and save value in the prediction dataframe.
            predictions_df.loc[step,'prediction'] = result.predict(exog=[[
                                                        predictions_df.loc[step,'const'], 
                                                        predictions_df.loc[step,'gpalscutoff'],
                                                        predictions_df.loc[step,'gpaXgpalscutoff'], 
                                                        predictions_df.loc[step,'gpaXgpagrcutoff']
                                                    ]])

            predictions_df = predictions_df.round(4)
        # Save the predictions for all groups in a dictionary. 
        predictions_groups_dict[group] = predictions_df
    
    return predictions_groups_dict

# ************ PLOTS FORMATTING, FRAMES, ETC.********************************************************************

def plot_figure3(inputs_dict, outputs_dict, keys):
    """ Plot results from RD anlaysis for the six subgroups of students in the paper.
    Args:
        inputs_dict(dict): Dicionary containing all dataframes for each subgroup, used for plotting the bins (dots).
        outputs_dict(dict): Dictionary containing the results from RD analysis for each subgroup, used for plotting the lines.
        keys(list): List of keys of the dictionaries, both dictionarie must have the same keys.
    
    Returns:
        plot: Figure 3 from the paper (figure consists of 6 subplots, one for each subgroup of students)
    """
    # Frame for entire figure.
    plt.pyplot.figure(figsize=(10, 13), dpi= 70, facecolor='w', edgecolor='k')
    plt.pyplot.subplots_adjust(wspace=0.4, hspace=0.4)    
    
    # Remove dataframe 'All' because I only want to plot the results for the subgroups of students.
    keys = keys.copy()
    keys.remove('All')
    
    # Create plots for all subgroups.
    for idx, key in enumerate(keys):
        # Define position of subplot.
        plot = plt.pyplot.subplot(3,2,idx+1)
        # Create frame for subplot
        plot = plt.pyplot.xlim(-1.5,1.5,0.1)
        plot = plt.pyplot.ylim(0,0.22,0.1)
        plot = plt.pyplot.axvline(x=0, color='r')
        plot = plt.pyplot.xlabel('First year GPA minus probation cutoff')
        plot = plt.pyplot.ylabel('Left university voluntarily')        
        # Plot subplot.
        plot = plt.pyplot.plot(inputs_dict[key].left_school.groupby(inputs_dict[key]['bins']).mean(), 'o')
        plot = plot_RDD_curve(df = outputs_dict[key], running_variable="dist_from_cut", outcome="prediction", cutoff=0)
        plot = plt.pyplot.title(key)
 
    return plot


def plot_RDD_curve(df, running_variable, outcome, cutoff):
    """ Function to plot RDD curves. Function splits dataset into treated and untreated group based on running variable
        and plots outcome (group below cutoff is treated, group above cutoff is untreated).
        Args:
            df(DataFrame): Dataframe containing the data to be plotted.
            running_variable(column): DataFrame column name of the running variable.
            outome(column): DataFrame column name of the outcome variable.
            cutoff(numeric): Value of cutoff.
        Returns:
            plot
    """    
    df_treat = df[df[running_variable] < cutoff]
    df_untreat = df[df[running_variable] >= cutoff]
    plt.pyplot.plot(df_treat[outcome])
    plt.pyplot.plot(df_untreat[outcome])
    
    plot = plt.pyplot.savefig(fname='plot')

    return plot



def plot_RDD_curve_colored(df, running_variable, outcome, cutoff, color):
    """ Function to plot RDD curves. Function splits dataset into treated and untreated group based on running variable
        and plots outcome (group below cutoff is treated, group above cutoff is untreated).
        Args:
            df(DataFrame): Dataframe containing the data to be plotted.
            running_variable(column): DataFrame column name of the running variable.
            outome(column): DataFrame column name of the outcome variable.
            cutoff(numeric): Value of cutoff.
        Returns:
            plot
    """    
    df_treat = df[df[running_variable] < cutoff]
    df_untreat = df[df[running_variable] >= cutoff]
    plt.pyplot.plot(df_treat[outcome], color = color, label='_nolegend_')
    plt.pyplot.plot(df_untreat[outcome], color = color, label='_nolegend_')
    
    plot = plt.pyplot.savefig(fname='plot')

    return plot