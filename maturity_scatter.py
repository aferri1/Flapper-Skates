#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 14:03:26 2025

@author: trekz1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import curve_fit

# Read the data
archDf = pd.read_csv('/Users/trekz1/Documents/MDM3/PhaseA/data/csv/archival_cleaned.csv')
IDDf = pd.read_csv('/Users/trekz1/Documents/MDM3/PhaseA/data/csv/skateids-archival.csv')

# Convert timestamp column to datetime
archDf['timestamp'] = pd.to_datetime(archDf['timestamp'])

# Filter data by maturity
matureDf = archDf[archDf['individual_id'].isin(IDDf[IDDf['maturity'] == 'mature']['individual_id'])]
immatureDf = archDf[archDf['individual_id'].isin(IDDf[IDDf['maturity'] == 'immature']['individual_id'])]

# Generate date columns
for df in [matureDf, immatureDf]:
    df['date'] = df['timestamp'].dt.date

# Find start dates for each category
start_dates = {
    "mature": matureDf['timestamp'].min(),
    "immature": immatureDf['timestamp'].min()
}

# Group by individual and date to find average daily depth and temp
def average_daily(df, category):
    return df.groupby(['individual_id', 'date'])['depth', 'temp'].mean().reset_index()

avgDailyDepthMature = average_daily(matureDf, 'mature')
avgDailyDepthImmature = average_daily(immatureDf, 'immature')

# Add days since start for daily data
for df, start in zip([avgDailyDepthMature, avgDailyDepthImmature],
                     [start_dates['mature'], start_dates['immature']]):
    df['daysSS'] = (pd.to_datetime(df['date']) - start).dt.days

# Scatter plot of average daily depth behavior from start for mature skates
plt.figure(figsize=(10, 5))
plt.scatter(avgDailyDepthMature['daysSS'], avgDailyDepthMature['depth'], color='blue', label='Mature Skates')
plt.title('Average Daily Depth Behavior Over Time (Mature Skates)')
plt.xlabel('Days Since Start')
plt.ylabel('Depth')
plt.legend()
plt.show()

# Scatter plot of average daily depth behavior from start for immature skates
plt.figure(figsize=(10, 5))
plt.scatter(avgDailyDepthImmature['daysSS'], avgDailyDepthImmature['depth'], color='red', label='Immature Skates')
plt.title('Average Daily Depth Behavior Over Time (Immature Skates)')
plt.xlabel('Days Since Start')
plt.ylabel('Depth')
plt.legend()
plt.show()

# Explanation and further analysis
'''
I noticed that when plotting the average daily depth behavior over a year, the immature skates seemed to follow a less consistent
 pattern than the mature skates. To explore this, I performed a covariance analysis based on the parameters of a sinusoidal model 
 fitted to the daily depth data for both mature and immature skates. The covariance matrices of these fitted parameters provided 
 insight into whether immature skates exhibited less predictable behavior than mature skates.
'''

# Sinusoidal model for curve fitting
def sinusoidal_model(x, A, B, C, D):
    return A * np.sin(B * (x - C)) + D

# Fit the sinusoidal model to the data for both mature and immature skates
def fit_sin_model(df, start_date, category):
    df_filtered = df[(df['daysSS'] >= 1) & (df['daysSS'] <= 365)]
    initial_guess = [
        (df_filtered['depth'].max() - df_filtered['depth'].min()) / 2,  # amplitude
        2 * np.pi / df_filtered['daysSS'].max(),  # frequency
        0,  # phase shift
        df_filtered['depth'].mean()  # vertical shift
    ]
    params, covmat = curve_fit(sinusoidal_model, df_filtered['daysSS'], df_filtered['depth'], p0=initial_guess)
    return params, covmat, df_filtered

params_mature, covmat_mature, df_mature = fit_sin_model(avgDailyDepthMature, start_dates['mature'], 'mature')
params_immature, covmat_immature, df_immature = fit_sin_model(avgDailyDepthImmature, start_dates['immature'], 'immature')

# Confidence intervals calculation based on covariance
def calculate_confidence_intervals(params, covariance, n, alpha=0.05):
    tval = chi2.ppf(1.0 - alpha/2., n - len(params))
    ci = [(param - np.sqrt(covariance[i, i]) * tval, param + np.sqrt(covariance[i, i]) * tval) for i, param in enumerate(params)]
    return ci

n_mature = len(df_mature)
n_immature = len(df_immature)
conf_int_mature = calculate_confidence_intervals(params_mature, covmat_mature, n_mature)
conf_int_immature = calculate_confidence_intervals(params_immature, covmat_immature, n_immature)

# Conclusion
'''
After performing covariance analysis, it was clear that the mature skates displayed a more predictable pattern in their 
resting depths, evidenced by smaller confidence intervals around the parameters of their sinusoidal fit. In contrast, the
 immature skates exhibited larger variability, suggesting less consistency in their depth behavior over the course of the year.
'''

# Plotting results for both mature and immature skates
def plot_model_fit(df, params, category):
    x_fit = np.linspace(df['daysSS'].min(), df['daysSS'].max(), 500)
    y_fit = sinusoidal_model(x_fit, *params)
    plt.figure(figsize=(10, 5))
    plt.scatter(df['daysSS'], df['depth'], color='blue')
    plt.plot(x_fit, y_fit, color='red', label=f'Fitted {category.capitalize()} Model')
    plt.xlabel('Days Since Start')
    plt.ylabel('Depth')
    plt.legend()
    plt.title(f'{category.capitalize()} Skate Resting Depth Over Time')
    plt.show()

plot_model_fit(df_mature, params_mature, 'mature')
plot_model_fit(df_immature, params_immature, 'immature')
