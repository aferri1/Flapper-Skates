# Flapper-Skates
This repository contains two python scripts designed to analyse the behavioural patterns of flapper skates, an endangered species of fish. We were tasked with finding patterns in the depths the skates reside at for tracking and conservation purposes, and were presented with a raw data file containing the skate ID, depth below sea level, date-time, gender, maturity and other factors for multiple skates over a year or more. The depth was recorded in two minute intervals. For privacy reasons the original data cannot be uploaded, however a folder with results from the script is uploaded for transparency.
Files:
- 'resting_analyse.py' - this script examines the resting behaviour of skates, such as finding the most common times of day that individual skates are resting, and the average amount of time spent resting each month. It works as follows:
  - Subsets the data by sex and maturity
  - Plots distributions of resting hours for certain individual skates, calculating top resting hours & their percentages
  - Computes the variance of resting hours and merges these results with metadata such as sex and maturity to see if there's a relationship between these factors and the variability in resting behavior.
  - Calculates average resting time by month and year, and plots these results by sex, to investigate if there is a significant difference in resting patterns between male and female skates.
- 'maturity_scatter.py' - this script investigates the behaviour of mature and immature skates. It focuses on the average daily depth over a year for both categories, and analyses whether there is a statistically significant difference between their behaviours. The script does the following:
  - Filters data by maturity
  - Finds average daily depths for each individual skate and plots a scatter graph of these over a year by maturity, by setting a start date and calculating 'Days Since Start' for the x-axis 
  - Fits a sinusoidal model to both scatter graphs and uses a covariance matrix to investigate whether immature skates have a less consistent pattern than mature skates
  - Plots the scatter graphs with fitted sinusoids for both categories, for visualisation purposes.

 Finally, the folder 'Results' contains outputs from these scripts, which includes:
 - 'avg_resting_times_sex.png' - graph showing monthly and yearly average resting depths for both male and female skates
 - 'Resting Hours Distributions' - folder containing histograms of three different skates' average resting hours. In the project report, the similarity in the shape of the histograms was used to suggest some skates show a pattern in their times resting, which could be useful for conservationists
 - 'Scatter Graphs with Sinusoids' - folder showing the outputs from maturity_scatter.py
 - 'Report Insert.png' - an insert from the project report which shows the confidence intervals calculated from the parameters of the sinusoids fitted to each scatter graph. The larger intervals for the immature category suggests a less certain fit for the sinusoid, implying that immature skates follow a less certain pattern in their daily depth behaviour than mature skates, which could be useful information for conservationists to know
