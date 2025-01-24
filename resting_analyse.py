5#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 21:37:07 2023

@author: trekz1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ids = pd.read_csv('/Users/trekz1/Documents/MDM3/PhaseA/data/csv/skateids-archival.csv')
T = pd.read_csv('/Users/trekz1/Documents/MDM3/PhaseA/data/csv/data_with_diving_and_resting.csv')

maleDf = T[T['individual_id'].isin(ids[ids['sex'] == 'male']['individual_id'])] 
femaleDf = T[T['individual_id'].isin(ids[ids['sex'] == 'female']['individual_id'])]

matureDf = T[T['individual_id'].isin(ids[ids['maturity'] == 'mature']['individual_id'])] 
immatureDf = T[T['individual_id'].isin(ids[ids['maturity'] == 'immature']['individual_id'])] 

current_skate = T

# indiv = current_skate['individual_id'].unique()

# for skate in indiv:

#     resting_only = current_skate[current_skate["resting"] == 1] #extract resting data
#     resting_only["start"] = resting_only["streak_id_resting"].shift() != resting_only["streak_id_resting"]

#     depth_averages_list = resting_only.groupby("streak_id_resting")["depth"].mean() # find mean depth of resting

#     # print(len(depth_averages_list))
#     # print(len(resting_only[resting_only["start"] == 1]))
#     # print(resting_only[["streak_id_resting", "start", "resting"]])

#     resting_only.loc[resting_only[resting_only["start"] == 1].index, "average_depth"] = list(depth_averages_list) # add this average depth to first entry of each rest
#     resting_only = resting_only[resting_only["start"] == 1] # extract the starts
#     resting_only['timestamp']=pd.to_datetime(resting_only['timestamp'])
#     # resting_only["timestamp"] = pd.to_datetime(resting_only["timestamp"]) # set timestamp as index
#     # resting_only.set_index('timestamp', inplace=True)
#     # resting_only = resting_only.sort_index() # sort by date not skate
#     resting_only['hour'] = resting_only['timestamp'].dt.hour
#     resting_only = resting_only.sort_values(by='hour')
    

def times_analysis(T,ids):
    current_skate = T
    resting_only = current_skate[current_skate['resting']==1] #extract resting data
    
    resting_only['timestamp']=pd.to_datetime(resting_only['timestamp'])
    resting_only['hour'] = resting_only['timestamp'].dt.hour # generate hour column based on timestamp
    resting_only = resting_only.sort_values(by='hour')
    resting_only['date'] = resting_only['timestamp'].dt.date # generate date column
    
   #  indiv = current_skate['individual_id'].unique() # loop through all or change to just do for one
    indiv = [25,62,65]
    
    all_variances = []
    
    for skate in indiv:
        indiv_resting = resting_only[resting_only['individual_id']==skate]
        plt.rcParams['font.size'] = 20
        plt.hist(indiv_resting['hour'],bins=np.arange(0, 25, 1),edgecolor='k',alpha=0.7,density=True) # creating a distribution graph for times spent resting
        plt.xlabel('Hour of the Day')
        plt.ylabel('Frequency')
        plt.xticks(range(0,26,2),fontsize = 14)  # label x-axis with hours (1 to 24)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
        hourly_counts = indiv_resting['hour'].value_counts().reset_index() # count number of entries at each hour
        hourly_counts.columns = ['hour', 'count']
        hourly_counts = hourly_counts.sort_values(by='count', ascending=False) # sort by largest first
        
        total_resting_events = len(indiv_resting) # total number of rsting entries for that fish
        top_3_hours = hourly_counts.head(3) # find top 3 hours
        top_3_hours['percentage'] = (top_3_hours['count']/total_resting_events) * 100 # percentage of resting events spent there
        variance = indiv_resting['hour'].var()
        all_variances.append((skate, variance))       
        unique_hours = indiv_resting['hour'].unique() 
        for hour in unique_hours:  # iterate through all the unique hours that this fish was resting for
            indiv_hours = indiv_resting[indiv_resting['hour']==hour] # filter out for each hour
            total_days_resting = len(indiv_resting['date'].unique()) # total number of days spent resting at this this hour for this skate
            total_resting_eventss = len(indiv_hours['date'].unique())
            # print(f'Fish {skate} spent was resting at hour {hour} for {total_resting_eventss} out of a possible {total_days_resting} days.')
            
    variance_df = pd.DataFrame(all_variances, columns=['individual_id', 'variance'])
    variance_df = variance_df.sort_values(by='variance', ascending=False)  # creating dataframes with each fish's variance
    
    fish_info_subset = ids[['individual_id', 'sex', 'maturity']]
    variance_df = variance_df.merge(fish_info_subset, on='individual_id', how='left') # merging dfs to include sex and maturity in variance df
    
    return total_days_resting, total_resting_eventss, top_3_hours, indiv_hours, resting_only, variance_df

total_days_resting, total_resting_eventss, top_3_hours, indiv_hours,resting_only, variance_df = times_analysis(T,ids)        

                
def avg_time_resting_monthly(T):
    T['timestamp'] = pd.to_datetime(T['timestamp'])    
    T['month'] = T['timestamp'].dt.month

    avg_resting_monthly = pd.DataFrame()  # initialise empty df to store results
    
    grouped = T.groupby(['month','individual_id']) # group by month % id
    
    for (month,individual), group in grouped:  # loop thru all individuals and months
        
        max_timestamp = group['timestamp'].max().replace(hour=23, minute=59, second=59) # consider only full days by removing aynthing past final midnight point
        group = group[group['timestamp'] <= max_timestamp]
        
        total_resting_time = group['resting'].sum() * 2  # Total resting minutes in the month for this individual
        total_time = len(group) * 2  
        
        if total_time > 0:            
            proportion_resting = total_resting_time / total_time # Calculate the proportion of the day spent resting
            hours_resting = 24 * proportion_resting  # Convert to hours
            # Append the result to the DataFrame:
            new_row = pd.DataFrame([{
                 'month': month,
                 'individual_id': individual,
                 'average_hours_resting': hours_resting
                 }])
            avg_resting_monthly = pd.concat([avg_resting_monthly, new_row], ignore_index=True)
      
    monthly_avg = avg_resting_monthly.groupby('month')['average_hours_resting'].mean()   # find average across all individuals
    yearly_avg = monthly_avg.mean()
  
    return monthly_avg,yearly_avg    

# plt.figure(figsize=(10, 5)) 
# monthly_avg.plot(kind='line', marker='o', label='Monthly average')
# plt.axhline(y=yearly_avg, color='r', linestyle='--', label='Yearly Mean')
# plt.title('Monthly and Yearly Average Resting Hours per Day')
# plt.xlabel('Month')
# plt.ylabel('Average Hours Resting Per Day')
# plt.xticks(range(1, 13))  # Set x-ticks to be the months
# plt.legend()
# plt.grid(True)
# plt.show()
        
# plt.figure(figsize=(10, 5))

# plt.plot(monthly_male.index, monthly_male.values, marker='o', label='Male Monthly Mean')
# plt.plot(monthly_female.index, monthly_female.values, marker='s', label='Female Monthly Mean')
# plt.axhline(y=yearly_male, color='blue', linestyle='--', label='Male Yearly Mean')
# plt.axhline(y=yearly_female, color='orange', linestyle='--', label='Female Yearly Mean')

# plt.xlabel('Month')
# plt.ylabel('Mean Hours Resting Per Day')
# plt.xticks(range(1, 13))
# plt.legend()
# plt.grid(True)
# plt.show()


# plt.figure(figsize=(10, 5))

# plt.plot(monthly_mature.index, monthly_mature.values, marker='o', label='Mature Monthly Mean')
# plt.plot(monthly_immature.index, monthly_immature.values, marker='s', label='Immature Monthly Mean')
# plt.axhline(y=yearly_mature, color='blue', linestyle='--', label='Mature Yearly Mean')
# plt.axhline(y=yearly_immature, color='orange', linestyle='--', label='Immature Yearly Mean')

# plt.xlabel('Month')
# plt.ylabel('Mean Hours Resting Per Day')
# plt.xticks(range(1, 13))
# plt.legend()
# plt.grid(True)
# plt.show()


        
def depths_analysis(T):
    current_skate = T
    resting_only = current_skate[current_skate['resting']==1] #extract resting data
    
    resting_only['timestamp']=pd.to_datetime(resting_only['timestamp'])
    resting_only['hour'] = resting_only['timestamp'].dt.hour # generate hour column based on timestamp
    resting_only = resting_only.sort_values(by='hour')
    resting_only['date'] = resting_only['timestamp'].dt.date # generate date column
    
    depth_bins = list(range(0,290,10)) # list of depth ranges that are resting
    resting_only['depth_range'] = pd.cut(resting_only['depth'],bins=depth_bins,labels=False)
    return
        

def avg_depth_resting_monthly(T):
    T['timestamp'] = pd.to_datetime(T['timestamp'])    
    T['month'] = T['timestamp'].dt.month

    avg_resting_monthly = T.groupby('month')['depth'].mean().reset_index()
    
    yearly_avg = avg_resting_monthly['depth'].mean()
    
    return(avg_resting_monthly,yearly_avg)

# avg_resting_monthly,yearly_avg = avg_depth_resting_monthly(T)
# monthly_male,yearly_male = avg_depth_resting_monthly(maleDf)
# monthly_female, yearly_female = avg_depth_resting_monthly(femaleDf)
# monthly_mature, yearly_mature = avg_depth_resting_monthly(matureDf)
# monthly_immature, yearly_immature = avg_depth_resting_monthly(immatureDf)



# plt.figure(figsize=(10, 5))

# avg_resting_monthly.plot(x='month', y='depth', kind='line', marker='o', label='Monthly Mean')
# plt.axhline(y=yearly_avg, color='r', linestyle='--', label='Yearly Mean')

# plt.xlabel('Month')
# plt.ylabel('Average Resting Depth')
# plt.xticks(range(1, 13))  # Set x-ticks to be the months
# plt.legend()
# plt.grid(True)
# plt.show()


# plt.figure(figsize=(10, 5))

# plt.plot(monthly_male['month'], monthly_male['depth'], marker='o', label='Male Monthly Mean')
# plt.plot(monthly_female['month'], monthly_female['depth'], marker='s', label='Female Monthly Mean')
# plt.axhline(y=yearly_male, color='blue', linestyle='--', label='Male Yearly Mean')
# plt.axhline(y=yearly_female, color='orange', linestyle='--', label='Female Yearly Mean')

# plt.xlabel('Month')
# plt.ylabel('Average Resting Depth')
# plt.xticks(range(1, 13))
# plt.legend()
# plt.grid(True)
# plt.show()



# plt.figure(figsize=(10, 5))

# plt.plot(monthly_mature['month'], monthly_mature['depth'], marker='o', label='Mature Monthly Mean')
# plt.plot(monthly_immature['month'], monthly_immature['depth'], marker='s', label='Immature Monthly Mean')
# plt.axhline(y=yearly_male, color='blue', linestyle='--', label='Mature Yearly Mean')
# plt.axhline(y=yearly_female, color='orange', linestyle='--', label='Immature Yearly Mean')

# plt.xlabel('Month')
# plt.ylabel('Average Resting Depth')
# plt.xticks(range(1, 13))
# plt.legend()
# plt.grid(True)
# plt.show()





# get gpt4 and take another look at heatmap

# analyse the 3 most commmon resting hours for each fish & % of time spent there
# consider how frequently fish were resting at a certain time? e.g 25 days out of a possible 30 they were resting between 1am - 2am
# add to graph what sex and maturity they are
# average time resting for male vs female  
# average time resting for immature vs mature 
# average rest lengths for male vs female
# average reste lengths for mature vs immature



# consider how to analyse common resting depths overall as well.


# compare most common rest ranges for different seasons/ months
# calculate distribution of depth ranges and variance for each fish as well

# put resting data into max's script












# resting_only = resting_only[1000:1250] # reduce the amount of data

# heatmap_data = resting_only.pivot(index='timestamp', columns='average_depth', values='average_depth') # generating heatmap  

# # sns.heatmap(heatmap_data, cmap='viridis')

# # plt.title('Fish Average Depths Heatmap')
# # plt.xlabel('Average Depth')
# # plt.ylabel('Timestamp')

# # # Display the heatmap
# # plt.show()



