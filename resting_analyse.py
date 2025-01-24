5#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:37:07 2023

@author: trekz1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data
ids = pd.read_csv('/Users/trekz1/Documents/MDM3/PhaseA/data/csv/skateids-archival.csv')
data = pd.read_csv('/Users/trekz1/Documents/MDM3/PhaseA/data/csv/data_with_diving_and_resting.csv')

# Subset data by sex and maturity
male_data = data[data['individual_id'].isin(ids[ids['sex'] == 'male']['individual_id'])]
female_data = data[data['individual_id'].isin(ids[ids['sex'] == 'female']['individual_id'])]

mature_data = data[data['individual_id'].isin(ids[ids['maturity'] == 'mature']['individual_id'])]
immature_data = data[data['individual_id'].isin(ids[ids['maturity'] == 'immature']['individual_id'])]

def analyse_resting_times(data, ids):
    """
    Analyse resting times, including most common hours and variance.
    """
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['date'] = data['timestamp'].dt.date

    individuals = [25, 62, 65]  # Example subset of individuals
    all_variances = []

    for individual in individuals:
        indiv_resting = data[(data['resting'] == 1) & (data['individual_id'] == individual)]

        # Plot histogram of resting hours
        plt.hist(indiv_resting['hour'], bins=np.arange(0, 25, 1), edgecolor='k', alpha=0.7, density=True)
        plt.xlabel('Hour of the Day')
        plt.ylabel('Frequency')
        plt.title(f'Resting Hour Distribution - Individual {individual}')
        plt.xticks(range(0, 24, 2))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()

        # Calculate top 3 hours and their percentages
        hourly_counts = indiv_resting['hour'].value_counts().reset_index()
        hourly_counts.columns = ['hour', 'count']
        hourly_counts = hourly_counts.sort_values(by='count', ascending=False)
        total_resting_events = len(indiv_resting)
        top_3_hours = hourly_counts.head(3)
        top_3_hours['percentage'] = (top_3_hours['count'] / total_resting_events) * 100

        # Calculate variance of resting hours
        variance = indiv_resting['hour'].var()
        all_variances.append((individual, variance))

    # Create variance DataFrame
    variance_df = pd.DataFrame(all_variances, columns=['individual_id', 'variance'])
    variance_df = variance_df.sort_values(by='variance', ascending=False)

    # Merge with metadata
    fish_info = ids[['individual_id', 'sex', 'maturity']]
    variance_df = variance_df.merge(fish_info, on='individual_id', how='left')

    return variance_df

def monthly_avg_resting_time(data):
    """
    Calculate average resting hours per month.
    """
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['month'] = data['timestamp'].dt.month

    avg_resting_monthly = []

    for (month, individual), group in data.groupby(['month', 'individual_id']):
        max_timestamp = group['timestamp'].max().replace(hour=23, minute=59, second=59)
        group = group[group['timestamp'] <= max_timestamp]

        total_resting_time = group['resting'].sum() * 2  # Total resting time in minutes
        total_time = len(group) * 2  # Total recorded time in minutes

        if total_time > 0:
            proportion_resting = total_resting_time / total_time
            hours_resting = 24 * proportion_resting

            avg_resting_monthly.append({'month': month, 'individual_id': individual, 'average_hours_resting': hours_resting})

    avg_resting_monthly_df = pd.DataFrame(avg_resting_monthly)
    monthly_avg = avg_resting_monthly_df.groupby('month')['average_hours_resting'].mean()
    yearly_avg = monthly_avg.mean()

    return monthly_avg, yearly_avg

def analyse_depth_patterns(data):
    """
    Analyse resting depths by creating depth ranges.
    """
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data['depth_range'] = pd.cut(data['depth'], bins=range(0, 290, 10), labels=False)
    return data

# Calculate averages for males and females
monthly_male, yearly_male = monthly_avg_resting_time(male_data)
monthly_female, yearly_female = monthly_avg_resting_time(female_data)

# Plot results
plt.figure(figsize=(10, 6))
plt.plot(monthly_male.index, monthly_male.values, marker='o', label='Male Monthly Mean')
plt.plot(monthly_female.index, monthly_female.values, marker='s', label='Female Monthly Mean')
plt.axhline(y=yearly_male, color='blue', linestyle='--', label=f'Male Yearly Mean: {yearly_male:.2f} hrs')
plt.axhline(y=yearly_female, color='orange', linestyle='--', label=f'Female Yearly Mean: {yearly_female:.2f} hrs')

# Add labels, legend, and grid
plt.xlabel('Month', fontsize=12)
plt.ylabel('Average Resting Time (Hours)', fontsize=12)
plt.title('Monthly and Yearly Average Resting Time by Gender', fontsize=14)
plt.xticks(range(1, 13), fontsize=10)
plt.legend(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()