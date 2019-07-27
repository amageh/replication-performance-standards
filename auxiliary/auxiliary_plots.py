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
        keys(list): List of keys of the dictionaries, both dictionaries must have the same keys.

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
        # Create frame for subplot.
        plot = plt.pyplot.xlim(-1.5, 1.5, 0.1)
        plot = plt.pyplot.ylim(0, 0.22, 0.1)
        plot = plt.pyplot.axvline(x=0, color='r')
        plot = plt.pyplot.xlabel('First year GPA minus probation cutoff')
        plot = plt.pyplot.ylabel('Left university voluntarily')
        # Calculate bin means.
        bin_means = inputs_dict[key].left_school.groupby(inputs_dict[key]['dist_from_cut_med10']).mean()
        bin_means = pd.Series.to_frame(bin_means)
        # Plot subplot.
        plot = plt.pyplot.plot(list(bin_means.index),list(bin_means.left_school), 'o')
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

    return 


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

    return 


def plot_RDD_curve_CI(df, running_variable, outcome, cutoff, lbound, ubound, CI_color, linecolor):
    """ Function to plot RDD curves with confidence intervals. Function splits dataset into treated and 
        untreated group based on running variable and plots outcome (group below cutoff is treated, group above 
        cutoff is untreated).

        Args:
            df(DataFrame): Dataframe containing the data to be plotted.
            running_variable(column): DataFrame column name of the running variable.
            outome(column): DataFrame column name of the outcome variable.
            cutoff(numeric): Value of cutoff.
            lbound(column): Lower bound of confidence interval.
            ubound(column): Upper bound of confidence interval.


        Returns:
            plot
    """
    df_treat = df[df[running_variable] < cutoff]
    df_untreat = df[df[running_variable] >= cutoff]
    # Plot confidence Intervals
    plt.pyplot.plot(df_treat[lbound], color=CI_color, alpha=0.7)
    plt.pyplot.plot(df_treat[ubound], color=CI_color, alpha=0.7)
    plt.pyplot.plot(df_untreat[lbound], color=CI_color, alpha=0.7)
    plt.pyplot.plot(df_untreat[ubound], color=CI_color, alpha=0.7)
    plt.pyplot.fill_between(df_treat[running_variable], 
                            y1=df_treat[lbound], 
                            y2=df_treat[ubound], 
                            facecolor=CI_color, 
                            alpha=0.7
                           )
    plt.pyplot.fill_between(df_untreat[running_variable], 
                            y1=df_untreat[lbound], 
                            y2=df_untreat[ubound], 
                            facecolor=CI_color, 
                            alpha=0.7
                           )
    
    # Plot estimated lines
    plt.pyplot.plot(df_untreat[outcome],
                    color=linecolor, 
                    label='_nolegend_'
                   )
    plt.pyplot.plot(df_treat[outcome],
                    color=linecolor, 
                    label='_nolegend_')

    return 


def plot_bin_frequency_RDD(bin_frequency, bins, predictions):
    """
    Args:
    bin_frequency(pd.DataFrame): Dataframe containing the frequency of each bin.
    bins(list): List of bins.
    predictions(pd.DataFrame): Predicted frequency of each bin.
    
    Returns: plot
    """
    plt.pyplot.xlim(-1.5,1.5,0.1)
    plt.pyplot.ylim(0,2100.5,50)
    plt.pyplot.axvline(x=0, color='r')
    plt.pyplot.xlabel('First year GPA minus probation cutoff')
    plt.pyplot.ylabel('Frequency count')
    plt.pyplot.plot(bin_frequency.bins,bin_frequency.freq, 'o')
    plot_RDD_curve(df = predictions, running_variable="bins", outcome="prediction", cutoff=0)
    plt.pyplot.title("Figure 1. Distribution of Student Grades Relative to their Cutoff")
    
    return 


