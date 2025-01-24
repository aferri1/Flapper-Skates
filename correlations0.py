#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:40:52 2023

@author: trekz1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import scipy.stats as stats
from scipy.optimize import curve_fit
from scipy.stats import chi2



archDf = pd.read_csv('/Users/trekz1/Documents/MDM3/PhaseA/data/csv/archival_cleaned.csv')
IDDf = pd.read_csv('/Users/trekz1/Documents/MDM3/PhaseA/data/csv/skateids-archival.csv')

archDf['timestamp'] = pd.to_datetime(archDf['timestamp'])



# create male and female dfs along with mature & immature:
maleDf = archDf[archDf['individual_id'].isin(IDDf[IDDf['sex'] == 'male']['individual_id'])] 
femaleDf = archDf[archDf['individual_id'].isin(IDDf[IDDf['sex'] == 'female']['individual_id'])]

matureDf = archDf[archDf['individual_id'].isin(IDDf[IDDf['maturity'] == 'mature']['individual_id'])] 
immatureDf = archDf[archDf['individual_id'].isin(IDDf[IDDf['maturity'] == 'immature']['individual_id'])] 

# generate date columns for all 4 of these df
maleDf['date'] = maleDf['timestamp'].dt.date
femaleDf['date'] = femaleDf['timestamp'].dt.date

matureDf['date'] = matureDf['timestamp'].dt.date
immatureDf['date'] = immatureDf['timestamp'].dt.date

# find startdates for each category
startdateMale = maleDf['timestamp'].min()
startdateFemale = femaleDf['timestamp'].min()

startdateMature = matureDf['timestamp'].min()
startdateImmature = immatureDf['timestamp'].min()

# # find days since start for each category
# maleDf['daysSS'] = (maleDf['timestamp'] - startdateMale).dt.days
# femaleDf['daysSS'] = (femaleDf['timestamp'] - startdateFemale).dt.days

# matureDf['daysSS'] = (matureDf['timestamp'] - startdateMature).dt.days
# immatureDf['daysSS'] = (immatureDf['timestamp'] - startdateImmature).dt.days




# find average monthly depth for each individual fish in each ofthese 4 cateogries
avgMonthlyDepthMale = maleDf.groupby(['individual_id','month'])['depth'].mean().reset_index()
avgMonthlyDepthFemale = femaleDf.groupby(['individual_id','month'])['depth'].mean().reset_index()

avgMonthlyDepthMature = matureDf.groupby(['individual_id','month'])['depth'].mean().reset_index()
avgMonthlyDepthImmature = immatureDf.groupby(['individual_id','month'])['depth'].mean().reset_index()

# calculate correlation between avg monthly depth and month for each individual category
corrMonthlyDepthMale = pearsonr(avgMonthlyDepthMale['month'],avgMonthlyDepthMale['depth'])
corrMonthlyDepthFemale = pearsonr(avgMonthlyDepthFemale['month'],avgMonthlyDepthFemale['depth'])

corrMonthlyDepthMature = pearsonr(avgMonthlyDepthMature['month'],avgMonthlyDepthMature['depth'])
corrMonthlyDepthImmature = pearsonr(avgMonthlyDepthImmature['month'],avgMonthlyDepthImmature['depth'])



# find daily average depth for each individual fish in each of these 4 categories
avgDailyDepthMale = maleDf.groupby(['individual_id','date'])['depth'].mean().reset_index()
avgDailyDepthFemale = femaleDf.groupby(['individual_id','date'])['depth'].mean().reset_index()

avgDailyDepthMature = matureDf.groupby(['individual_id','date'])['depth'].mean().reset_index()
avgDailyDepthImmature = immatureDf.groupby(['individual_id','date'])['depth'].mean().reset_index()

# convert dates to datetime
avgDailyDepthMale['date'] = pd.to_datetime(avgDailyDepthMale['date'])
avgDailyDepthFemale['date'] = pd.to_datetime(avgDailyDepthFemale['date'])

avgDailyDepthMature['date'] = pd.to_datetime(avgDailyDepthMature['date'])
avgDailyDepthImmature['date'] = pd.to_datetime(avgDailyDepthImmature['date'])

# add days since start column for to each daily df
avgDailyDepthMale['daysSS'] = (avgDailyDepthMale['date']-startdateMale).dt.days
avgDailyDepthFemale['daysSS'] = (avgDailyDepthFemale['date']-startdateFemale).dt.days

avgDailyDepthMature['daysSS'] = (avgDailyDepthMature['date']-startdateMature).dt.days
avgDailyDepthImmature['daysSS'] = (avgDailyDepthImmature['date']-startdateImmature).dt.days


# calculate correlation between avg daily depth and day for each individual category
corrDailyDepthMale = pearsonr(avgDailyDepthMale['daysSS'],avgDailyDepthMale['depth'])
corrDailyDepthFemale = pearsonr(avgDailyDepthFemale['daysSS'],avgDailyDepthFemale['depth'])

corrDailyDepthMature = pearsonr(avgDailyDepthMature['daysSS'],avgDailyDepthMature['depth'])
corrDailyDepthImmature = pearsonr(avgDailyDepthImmature['daysSS'],avgDailyDepthImmature['depth'])



# find average monthly temp for each individual fish in each ofthese 4 cateogries
avgMonthlyTempMale = maleDf.groupby(['individual_id','month'])['temp'].mean().reset_index()
avgMonthlyTempFemale = femaleDf.groupby(['individual_id','month'])['temp'].mean().reset_index()

avgMonthlyTempMature = matureDf.groupby(['individual_id','month'])['temp'].mean().reset_index()
avgMonthlyTempImmature = immatureDf.groupby(['individual_id','month'])['temp'].mean().reset_index()

# calculate correlation between avg monthly temp and month for each individual category
corrMonthlyTempMale = pearsonr(avgMonthlyTempMale['month'],avgMonthlyTempMale['temp'])
corrMonthlyTempFemale = pearsonr(avgMonthlyTempFemale['month'],avgMonthlyTempFemale['temp'])

corrMonthlyTempMature = pearsonr(avgMonthlyTempMature['month'],avgMonthlyTempMature['temp'])
corrMonthlyTempImmature = pearsonr(avgMonthlyTempImmature['month'],avgMonthlyTempImmature['temp'])



# find average daily temp for each individual fish in each of the 4 categories
avgDailyTempMale = maleDf.groupby(['individual_id','date'])['temp'].mean().reset_index()
avgDailyTempFemale = femaleDf.groupby(['individual_id','date'])['temp'].mean().reset_index()

avgDailyTempMature = matureDf.groupby(['individual_id','date'])['temp'].mean().reset_index()
avgDailyTempImmature = immatureDf.groupby(['individual_id','date'])['temp'].mean().reset_index()

# convert dates to datetime
avgDailyTempMale['date'] = pd.to_datetime(avgDailyTempMale['date'])
avgDailyTempFemale['date'] = pd.to_datetime(avgDailyTempFemale['date'])

avgDailyTempMature['date'] = pd.to_datetime(avgDailyTempMature['date'])
avgDailyTempImmature['date'] = pd.to_datetime(avgDailyTempImmature['date'])

# add days since start column for to each daily df
avgDailyTempMale['daysSS'] = (avgDailyTempMale['date']-startdateMale).dt.days
avgDailyTempFemale['daysSS'] = (avgDailyTempFemale['date']-startdateFemale).dt.days

avgDailyTempMature['daysSS'] = (avgDailyTempMature['date']-startdateMature).dt.days
avgDailyTempImmature['daysSS'] = (avgDailyTempImmature['date']-startdateImmature).dt.days

# calculate correlation between avg daily temp and day for each individual category
corrDailyTempMale = pearsonr(avgDailyTempMale['daysSS'],avgDailyTempMale['temp'])
corrDailyTempFemale = pearsonr(avgDailyTempFemale['daysSS'],avgDailyTempFemale['temp'])

corrDailyTempMature = pearsonr(avgDailyTempMature['daysSS'],avgDailyTempMature['temp'])
corrDailyTempImmature = pearsonr(avgDailyTempImmature['daysSS'],avgDailyTempImmature['temp'])

# plt.scatter(avgMonthlyTempFemale['month'],avgMonthlyTempFemale['Temp'])
# plt.xlabel('Month of year')
# plt.ylabel('Average temp')


# plt.scatter(avgDailyDepthMature['daysSS'],avgDailyDepthMature['depth'])
# plt.xlabel('Days since start')
# plt.ylabel('Average depth')
# plt.title('Scatter graph of average daily depth for mature skates')
# plt.show()

# plt.scatter(avgDailyDepthImmature['daysSS'],avgDailyDepthImmature['depth'])
# plt.xlabel('Days since start')
# plt.ylabel('Average depth')
# plt.title('Scatter graph of average daily depth for immature skates')
# plt.show()

# plt.scatter(avgDailyDepthMale['daysSS'],avgDailyDepthMale['depth'])
# plt.xlabel('Days since start')
# plt.ylabel('Average depth')
# plt.title('Scatter graph of average daily depth for male skates')
# plt.show()

# plt.scatter(avgDailyDepthFemale['daysSS'],avgDailyDepthFemale['depth'])
# plt.xlabel('Days since start')
# plt.ylabel('Average depth')
# plt.title('Scatter graph of average daily depth for female skates')
# plt.show()



# plt.scatter(avgMonthlyDepthMature['month'],avgMonthlyDepthMature['depth'])
# plt.xlabel('Month')
# plt.ylabel('Average depth')
# plt.title('Scatter graph of average monthly depth for mature skates')
# plt.show()

# plt.scatter(avgMonthlyDepthImmature['month'],avgMonthlyDepthImmature['depth'])
# plt.xlabel('Month')
# plt.ylabel('Average depth')
# plt.title('Scatter graph of average monthly depth for immature skates')
# plt.show()

# plt.scatter(avgMonthlyDepthMale['month'],avgMonthlyDepthMale['depth'])
# plt.xlabel('Month')
# plt.ylabel('Average depth')
# plt.title('Scatter graph of average monthly depth for male skates')
# plt.show()

# plt.scatter(avgMonthlyDepthFemale['month'],avgMonthlyDepthFemale['depth'])
# plt.xlabel('Month')
# plt.ylabel('Average depth')
# plt.title('Scatter graph of average monthly depth for female skates')
# plt.show()


# plt.scatter(avgDailyTempMature['daysSS'],avgDailyTempMature['temp'])
# plt.xlabel('Days since start')
# plt.ylabel('Average temperature')
# plt.title('Scatter graph of average daily temperature for mature skates')
# plt.show()

# plt.scatter(avgDailyTempImmature['daysSS'],avgDailyTempImmature['temp'])
# plt.xlabel('Days since start')
# plt.ylabel('Average temperature')
# plt.title('Scatter graph of average daily temperature for immature skates')
# plt.show()

# plt.scatter(avgDailyTempMale['daysSS'],avgDailyTempMale['temp'])
# plt.xlabel('Days since start')
# plt.ylabel('Average temperature')
# plt.title('Scatter graph of average daily temperature for male skates')
# plt.show()

# plt.scatter(avgDailyTempFemale['daysSS'],avgDailyTempFemale['temp'])
# plt.xlabel('Days since start')
# plt.ylabel('Average temperature')
# plt.title('Scatter graph of average daily temperature for female skates')
# plt.show()



# plt.scatter(avgMonthlyTempMature['month'],avgMonthlyTempMature['temp'])
# plt.xlabel('Month')
# plt.ylabel('Average temp')
# plt.title('Scatter graph of average monthly temp for mature skates')
# plt.show()

# plt.scatter(avgMonthlyTempImmature['month'],avgMonthlyTempImmature['temp'])
# plt.xlabel('Month')
# plt.ylabel('Average temp')
# plt.title('Scatter graph of average monthly temp for immature skates')
# plt.show()

# plt.scatter(avgMonthlyTempMale['month'],avgMonthlyTempMale['temp'])
# plt.xlabel('Month')
# plt.ylabel('Average temp')
# plt.title('Scatter graph of average monthly temp for male skates')
# plt.show()

# plt.scatter(avgMonthlyTempFemale['month'],avgMonthlyTempFemale['temp'])
# plt.xlabel('Month')
# plt.ylabel('Average temp')
# plt.title('Scatter graph of average monthly temp for female skates')
# plt.show()






# do over a year for mature and immature and calc covariance matrix, diag values of covariance matrix will show how many of each 
# confidence interval

avgDailyDepthMature2 = avgDailyDepthMature[(avgDailyDepthMature['daysSS']>=1) & (avgDailyDepthMature['daysSS']<=365)]
avgDailyDepthImmature2 = avgDailyDepthImmature[(avgDailyDepthImmature['daysSS']>=1) & (avgDailyDepthImmature['daysSS']<=365)]

def sinusoidal_model(x, A, B, C, D):
    return A * np.sin(B * (x - C)) + D  # sinusoidal model

# inital guesses for mature and immature:
    
amplitude_guess_mature = (avgDailyDepthMature2['depth'].max() - avgDailyDepthMature2['depth'].min()) / 2
frequency_guess = 2 * np.pi / avgDailyDepthMature2['daysSS'].max()
phase_shift_guess = 0  # without more information, starting with zero
vertical_shift_guess_mature = avgDailyDepthMature2['depth'].mean()

amplitude_guess_immature = (avgDailyDepthImmature2['depth'].max() - avgDailyDepthImmature2['depth'].min()) / 2
vertical_shift_guess_immature = avgDailyDepthImmature2['depth'].mean()

initial_guess_mature = [amplitude_guess_mature,frequency_guess,phase_shift_guess,vertical_shift_guess_mature]

initial_guess_immature = [amplitude_guess_immature,frequency_guess,phase_shift_guess,vertical_shift_guess_immature]


# fit model to mature and immature datasets:
params_mature, covmat_mature = curve_fit(
sinusoidal_model,
avgDailyDepthMature2['daysSS'].values,
avgDailyDepthMature2['depth'].values,
p0=initial_guess_mature
)    

params_immature, covmat_immature = curve_fit(
sinusoidal_model,
avgDailyDepthImmature2['daysSS'].values,
avgDailyDepthImmature2['depth'].values,
p0=initial_guess_immature
)   

x_fit = np.linspace(avgDailyDepthMature2['daysSS'].min(), avgDailyDepthMature2['daysSS'].max(), 500)

y_fit_mature = sinusoidal_model(x_fit, *params_mature)
y_fit_immature = sinusoidal_model(x_fit, *params_immature)
plt.rcParams.update({'font.size': 20})
#  plot the mature scatter graph & fitted curve:
plt.figure(figsize=(10, 5))
plt.scatter(avgDailyDepthMature2['daysSS'],avgDailyDepthMature2['depth'],color='blue')
plt.plot(x_fit,y_fit_mature,color='red',label = 'Fitted Mature Model')
plt.xlabel('Days Since March 4th 2016')
plt.ylabel('Depth')
plt.legend()
plt.show()


plt.figure(figsize=(10, 5))
plt.scatter(avgDailyDepthImmature2['daysSS'],avgDailyDepthImmature2['depth'],color='blue')
plt.plot(x_fit,y_fit_immature,color='red',label = 'Fitted Immature Model')
plt.xlabel('Days Since March 4th 2016')
plt.ylabel('Depth')
plt.ylim(0,250)
plt.legend()
plt.show()

# print parameters, covmat and confidence intervals of both mature and immature:

# print("Mature Fish Model Parameters:", params_mature)
# print("Covariance Matrix of Mature Fish:\n", covmat_mature)

# confidence_interval_mature = 1.96 * np.sqrt(np.diag(covmat_mature))  # 1.96 represents 95% confidence level
# print("Confidence Intervals for Mature Fish Parameters:", confidence_interval_mature)


# print("Immature Fish Model Parameters:", params_immature)
# print("Covariance Matrix of Immature Fish:\n", covmat_immature)

# confidence_interval_immature = 1.96 * np.sqrt(np.diag(covmat_immature))
# print("Confidence Intervals for Immature Fish Parameters:", confidence_interval_immature)

# Function to calculate confidence intervals
def calculate_confidence_intervals(params, covariance, n, alpha):
    ci = []
    p = len(params)
    dof = max(0, n - p)  # degrees of freedom
    tval = chi2.ppf(1.0 - alpha/2., dof)
    for i, param in enumerate(params):
        sigma = np.sqrt(covariance[i, i])
        delta = sigma * np.sqrt(tval)
        ci.append((param - delta, param + delta))
    return ci

n1 = len(avgDailyDepthMature2['depth'])
n2 = len(avgDailyDepthImmature2['depth'])
alpha = 0.05

conf_int_mature = calculate_confidence_intervals(params_mature, covmat_mature, n1, alpha)
conf_int_immature = calculate_confidence_intervals(params_immature, covmat_immature, n2, alpha)






# fit models with different set of parameters but same model onto mature and immature graphs
#  compare covariance matrices of the parameters for each graph
# the one the higher covariance matrix is a worse fit? double check that with liam







