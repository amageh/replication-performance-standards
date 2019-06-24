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


def create_table_6(groups_dict, groups_labels, outcome):
    table = table_template.copy()
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

def create_predictions(data, outcome):
    
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


def create_fig3_predictions(groups_dict):
    
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

# ************ PLOTS *********************************************************************

def fig3_subplots_frame():
    """ 
    Creates frame to be used for all subplots of figure 3.
    """
    plt.pyplot.xlim(-1.5,1.5,0.1)
    plt.pyplot.ylim(0,0.22,0.1)
    plt.pyplot.axvline(x=0, color='r')
    plt.pyplot.xlabel('First year GPA minus probation cutoff')
    plt.pyplot.ylabel('Left university voluntarily')
    plot = plt.pyplot.savefig(fname='plot')
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