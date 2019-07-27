"""This module contains auxiliary functions for RD predictions used in the main notebook."""
import json

import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels as sm


def create_predictions(data, outcome, regressors, bandwidth):

    steps = np.arange(-1.2, 1.25, 0.05)
    predictions_df = pd.DataFrame([])
    # Ensure there are no missings in the outcome variable
    data = data.dropna(subset=[outcome])
    # Loop through bins or 'steps'.
    for step in steps:
        df = data[(data.dist_from_cut >= (step - bandwidth)) &
                  (data.dist_from_cut <= (step + bandwidth))]
        # Run regression for with all values in the range specified above.
        model = sm.regression.linear_model.OLS(df[outcome], df[regressors], hasconst=True)
        result = model.fit(cov_type='cluster', cov_kwds={'groups': df['clustervar']})

        # Fill in row for each step in the prediction datframe.
        predictions_df.loc[step, 'dist_from_cut'] = step
        if step < 0:
            predictions_df.loc[step, 'gpalscutoff'] = 1
        else:
            predictions_df.loc[step, 'gpalscutoff'] = 0

        predictions_df.loc[step, 'gpaXgpalscutoff'] = (predictions_df.loc[step, 'dist_from_cut']) * predictions_df.loc[step, 'gpalscutoff']
        predictions_df.loc[step, 'gpaXgpagrcutoff'] = (predictions_df.loc[step, 'dist_from_cut']) * (1 - predictions_df.loc[step, 'gpalscutoff'])
        predictions_df.loc[step, 'const'] = 1

        # Make prediction for each step based on regression of each step and
        # save value in the prediction dataframe.
        predictions_df.loc[step, 'prediction'] = result.predict(exog=[[
                                                                        predictions_df.loc[step, 'const'],
                                                                        predictions_df.loc[step, 'gpalscutoff'],
                                                                        predictions_df.loc[step, 'gpaXgpalscutoff'],
                                                                        predictions_df.loc[step, 'gpaXgpagrcutoff']
                                                                    ]])

    predictions_df.round(4)

    return predictions_df



def create_bin_frequency_predictions(data, steps, bandwidth):

    #steps = np.arange(-1.2, 1.25, 0.05)
    predictions_df = pd.DataFrame([])
    # Ensure there are no missings in the outcome variable
    #data = data.dropna(subset=[outcome])
    # Loop through bins or 'steps'.
    for step in steps:
        df = data[(data.bins >= (step - bandwidth)) &
                  (data.bins <= (step + bandwidth))]
        # Run regression for with all values in the range specified above.
        model = sm.regression.linear_model.OLS(df['freq'], df[['const','bins']], hasconst=True)
        result = model.fit()

        # Fill in row for each step in the prediction datframe.
        predictions_df.loc[step, 'bins'] = step
        predictions_df.loc[step, 'const'] = 1
        predictions_df.loc[step, 'prediction'] = result.predict(exog=[[
                                                                        predictions_df.loc[step, 'const'],
                                                                        predictions_df.loc[step, 'bins'],
                                                                    ]])

    predictions_df.round(4)

    return predictions_df



def create_fig3_predictions(groups_dict, regressors, bandwidth):

    predictions_groups_dict = {}
    # Loop through groups:
    for group in groups_dict:

        steps = np.arange(-1.2, 1.25, 0.05)
        predictions_df = pd.DataFrame([])

        # Loop through bins or 'steps'.
        for step in steps:
            # Select dataframe from the dictionary.
            df = groups_dict[group][(groups_dict[group].dist_from_cut >= (step - bandwidth)) & 
                                    (groups_dict[group].dist_from_cut <= (step + bandwidth))]
            # Run regression for with all values in the range specified above.
            model = sm.regression.linear_model.OLS(df['left_school'], df[regressors], hasconst=True)
            result = model.fit(cov_type='cluster', cov_kwds={'groups': df['clustervar']})

            # Fill in row for each step in the prediction datframe.
            predictions_df.loc[step, 'dist_from_cut'] = step
            if step < 0:
                predictions_df.loc[step, 'gpalscutoff'] = 1
            else:
                predictions_df.loc[step, 'gpalscutoff'] = 0

            predictions_df.loc[step, 'gpaXgpalscutoff'] = ( predictions_df.loc[step, 'dist_from_cut']) * predictions_df.loc[step, 'gpalscutoff']
            predictions_df.loc[step, 'gpaXgpagrcutoff'] = (predictions_df.loc[step, 'dist_from_cut']) * (1 - predictions_df.loc[step, 'gpalscutoff'])
            predictions_df.loc[step, 'const'] = 1

            # Make prediction for each step based on regression of each step
            # and save value in the prediction dataframe.
            predictions_df.loc[step, 'prediction'] = result.predict(exog=[[
                                                                            predictions_df.loc[step, 'const'],
                                                                            predictions_df.loc[step, 'gpalscutoff'],
                                                                            predictions_df.loc[step, 'gpaXgpalscutoff'],
                                                                            predictions_df.loc[step, 'gpaXgpagrcutoff']
                                                                        ]])

            predictions_df = predictions_df.round(4)
        # Save the predictions for all groups in a dictionary.
        predictions_groups_dict[group] = predictions_df

    return predictions_groups_dict



def bootstrap_predictions(n, data, outcome, regressors, bandwidth):
    bootstrap_pred = pd.DataFrame({})
    for i in range(0,n):
        bootstrap = data.sample(n=len(data), replace=True)
        pred = create_predictions(data=bootstrap, outcome = outcome, regressors=regressors, bandwidth=bandwidth)
        bootstrap_pred['pred_'+ str(i)]= pred.prediction
        i=+1
    return bootstrap_pred 


def get_confidence_interval(data, lbound, ubound, index_var):
    confidence_interval = pd.DataFrame({})
    for i in data.index:
        confidence_interval.loc[i,"lower_bound"] = np.percentile(data.loc[i,:], lbound)
        confidence_interval.loc[i,"upper_bound"] = np.percentile(data.loc[i,:], ubound)
    
    confidence_interval[index_var] = confidence_interval.index        
    return confidence_interval