"""This module contains auxiliary functions for plotting which are used in the main notebook."""

import matplotlib as plt
import pandas as pd
import numpy as np
import statsmodels as sm


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
    plt.pyplot.figure(figsize=(10, 13), dpi=70, facecolor='w', edgecolor='k')
    plt.pyplot.subplots_adjust(wspace=0.4, hspace=0.4)

    # Remove dataframe 'All' because I only want to plot the results for the
    # subgroups of students.
    keys = keys.copy()
    keys.remove('All')

    # Create plots for all subgroups.
    for idx, key in enumerate(keys):
        # Define position of subplot.
        plot = plt.pyplot.subplot(3, 2, idx + 1)
        # Create frame for subplot
        plot = plt.pyplot.xlim(-1.5, 1.5, 0.1)
        plot = plt.pyplot.ylim(0, 0.22, 0.1)
        plot = plt.pyplot.axvline(x=0, color='r')
        plot = plt.pyplot.xlabel('First year GPA minus probation cutoff')
        plot = plt.pyplot.ylabel('Left university voluntarily')
        # Plot subplot.
        plot = plt.pyplot.plot(inputs_dict[key].left_school.groupby(inputs_dict[key]['bins']).mean(), 'o')
        plot = plot_RDD_curve(
                                df=outputs_dict[key], 
                                running_variable="dist_from_cut", 
                                outcome="prediction", 
                                cutoff=0
                                )
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
    plt.pyplot.plot(
                    df_treat[outcome], 
                    color=color, 
                    label='_nolegend_'
                    )
    plt.pyplot.plot(
                    df_untreat[outcome], 
                    color=color, 
                    label='_nolegend_')

    plot = plt.pyplot.savefig(fname='plot')

    return plot
