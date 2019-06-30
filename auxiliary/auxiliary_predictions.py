"""This module contains auxiliary functions for RD predictions used in the main notebook."""
import json

import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels as sm


def create_predictions(data, outcome, regressors):

    steps = np.arange(-1.2, 1.25, 0.05)
    predictions_df = pd.DataFrame([])
    # Ensure there are no missings in the outcome variable
    data = data.dropna(subset=[outcome])
    # Loop through bins or 'steps'.
    for step in steps:
        #df =  df.dropna(subset=['year2_dist_from_cut'])
        df = data[(data.dist_from_cut >= (step - 0.6)) &
                  (data.dist_from_cut <= (step + 0.6))]
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


def create_fig3_predictions(groups_dict, regressors):

    predictions_groups_dict = {}
    # Loop through groups:
    for group in groups_dict:

        steps = np.arange(-1.2, 1.25, 0.05)
        predictions_df = pd.DataFrame([])

        # Loop through bins or 'steps'.
        for step in steps:
            # Select dataframe from the dictionary.
            df = groups_dict[group][(groups_dict[group].dist_from_cut >= (step - 0.6)) & 
                                    (groups_dict[group].dist_from_cut <= (step + 0.6))]
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
