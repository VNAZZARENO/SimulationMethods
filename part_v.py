# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 18:47:54 2023

@author: vince
"""

import pandas as pd

git_url = "https://raw.githubusercontent.com/VNAZZARENO/SimulationMethods/main/data_simulation_methods.csv"
data = pd.read_csv(git_url, header=None, delimiter=',')

print(data)


# Q1 #

import csv
import math
import matplotlib.pyplot as plt
from astropy.modeling import models,fitting

import numpy as np
import matplotlib.pyplot as plt

def calculate_mean(tab):
    # calculate the mean
    mean = np.mean(np.log(tab))
    return mean

def calculate_volatility(tab, mean):
    # calculate the volatility
    vol = np.mean(np.log(tab) - mean)
    return vol

def calculate_log_return(tab):
    # calculate the log return
    # (P1 - P0) / P0 * 100
    return np.diff(np.log(tab)) / np.log(tab[:-1]) * 100

def create_box_plot(tab):
    # create a box plot
    figure = plt.boxplot(tab, patch_artist=True)
    return figure

# MAIN
asset_1 = data[0]
asset_2 = data[1]

# calculate the mean
mean_1 = calculate_mean(asset_1)
mean_2 = calculate_mean(asset_2)

print("Mean asset 1: ", mean_1)
print("Mean asset 2: ", mean_2)

# calculate the volatility
vol_1 = calculate_volatility(asset_1, mean_1)
print("***\nVolatility asset 1: ", vol_1)

vol_2 = calculate_volatility(asset_2, mean_2)
print("Volatility asset 2: ", vol_2)

# calculate the log return
return_1 = calculate_log_return(asset_1)
return_2 = calculate_log_return(asset_2)

# calculate the drift
drift_1 = np.mean(return_1)
drift_2 = np.mean(return_2)

print("***\nDrift asset 1: ", drift_1)
print("Drift asset 2: ", drift_2)

# create a box plot
boxplot_1 = create_box_plot(asset_1)
boxplot_2 = create_box_plot(asset_2)

plt.title('Box Plots of asset 1 and asset 2:')
plt.show()



# Q2 #


def create_histogram(tab):
    # create a histogram
    num_bins = 30
    plt.grid(True)
    figure = plt.hist(tab, num_bins, facecolor='blue', alpha=0.5, label='histogram')
    return figure

# create histograms
histogram_1 = create_histogram(return_1)
plt.title('Hist of log returns of A1:')
plt.show()

histogram_2 = create_histogram(return_2)
plt.title('Hist of log returns of A2:')
plt.show()



def Best_Fit_line(tab):
    # Plot a best fit line
    bin_heights, bin_borders = np.histogram(tab, bins='auto')
    bin_widths = np.diff(bin_borders)
    bin_centers = bin_borders[:-1] + bin_widths/2

    t_init = models.Gaussian1D()
    fit_t = fitting.LevMarLSQFitter()
    t = fit_t(t_init, bin_centers, bin_heights)

    x_interval_for_fit = np.linspace(bin_borders[0], bin_borders[-1], 10000)

    plt.figure()
    plt.bar(bin_centers, bin_heights, width=bin_widths, label='histogram')
    plt.plot(x_interval_for_fit, t(x_interval_for_fit), label='fit', c='red')
    plt.legend()

# Best fit line for Asset_1
Best_Fit_line(return_1)
plt.title('Hist of log returns of A1:')
plt.show()

# Best fit line for Asset_2
Best_Fit_line(return_2)
plt.title('Hist of log returns of A2:')
plt.show()


# Q3 #

# create a scatter plot
plt.title('Scatter Plot of the log-returns of the 2 assets')
plt.scatter(return_1, return_2, c='orange')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.xlabel('Log-return of Asset 1')
plt.ylabel('Log-return of Asset 2')
plt.show()


# Q4 #

import scipy.stats as stats
import seaborn as sns
from scipy.stats import lognorm

# Transform data to unit square (copula scale) using a kernel estimator of the cumulative distribution function
s1 = np.std(data[0])
s2 = np.std(data[1])

u = lognorm.cdf(data[0], s=s1)
v = lognorm.cdf(data[1], s=s2)

# Kendall-tau
tau, p_value = stats.kendalltau(data[0], data[1])
print("Kendall's tau:", tau, "with a p-value of", p_value)

# Find optimal theta
theta_opt = 2 * tau / (1 - tau)

# Clayton copula
def clayton_copula(u, v, theta):
    return pow(pow(u, -theta) + pow(v, -theta) - 1, -1/theta)

def clayton_generator(t, theta):
    return (1/theta * (pow(t, -theta) - 1))

clayton = clayton_copula(u, v, theta_opt)

# Create a scatter plot of the Clayton copula
sns.scatterplot(data=clayton, x=u, y=v)
plt.title('Clayton Copula')
plt.xlabel('u')
plt.ylabel('v')
plt.show()




# Q5 #

def gen_box_muller(n):
    """Box-Muller to generate Standard Normal variables"""
    # Generate U1 & U2 ~U(0,1)
    u1 = np.random.uniform(0, 1, size=n)
    theta = 2 * np.pi * u1
    u2 = np.random.uniform(0, 1, size=n)
    
    # Normal random variables
    R = -2 * np.log(1 - u2)
    
    return np.sqrt(R) * np.cos(theta)

def plot_trajectories(tab_lines, tab_labels):
    """Plot the trajectories"""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,8))
    fig.suptitle("Price Trajectory Simulations")
    
    for i in range(len(tab_lines)):
        ax.plot(tab_lines[i], label=tab_labels[i])
    
    ax.grid(True)
    ax.legend(loc='best')
    ax.set_xlabel('Days')
    ax.set_ylabel('Price')
    plt.show()

# Next 1000 days of data
nb_of_days = 1000
S0 = data.tail(1) 

# Generation of normal values ~N(m, vol)
random_values = gen_box_muller(nb_of_days)

# Following a standard normal distribution
normal_values_1 = random_values * (np.std(data[0])) ** 0.5
normal_values_2 = random_values * (np.std(data[1])) ** 0.5

walk_1 = [S0.iloc[0,0]] # First trajectory
walk_2 = [S0.iloc[0,1]] # Second trajectory

for i in range(nb_of_days):
    walk_1.append(walk_1[i] + normal_values_1[i])
    walk_2.append(walk_2[i] + normal_values_2[i])

list_trajectories = [walk_1, walk_2]
list_label_trajectories = [f"Walk-1:\nS0={S0.iloc[0,0]}\nMean={mean_1}\nVolatility={vol_1}",
                          f"Walk-2:\nS0={S0.iloc[0,1]}\nMean={mean_2}\nVolatility={vol_2}"]

# Plot trajectories
plot_trajectories(list_trajectories, list_label_trajectories)


#MAIN:
asset_1 = walk_1
asset_2 = walk_2

#Mean
simulate_mean_1 = calculate_mean(asset_1)
simulate_mean_2 = calculate_mean(asset_2)
print("Mean asset1 : ", simulate_mean_1)
print("Mean asset2 : ", simulate_mean_2)

#Volatility
simulate_vol_1 = calculate_volatility(asset_1, simulate_mean_1)
print("***\nVolatility asset1 : ", simulate_vol_1)
simulate_vol_2 = calculate_volatility(asset_2, simulate_mean_2)
print("Volatility asset2 : ", simulate_vol_2)

#log-return
return_1 = []
return_2 = []
return_1 = calculate_log_return(asset_1)
return_2 = calculate_log_return(asset_2)

#Drift
drift_1 = 0
drift_2 = 0
for i in range(len(return_1)):
    drift_1 = drift_1 + return_1[i]
    drift_2 = drift_2 + return_2[i]

drift_1 = drift_1 / len(return_1)
drift_2 = drift_2 / len(return_2)
print("***\nDrift asset1 : ", drift_1)
print("Drift asset2 : ", drift_2)

#Box plot
plt.subplot(121)
BP_1 = create_box_plot(asset_1)
plt.subplot(122)
BP_2 = create_box_plot(asset_2)
plt.title('Box Plots of asset_1 and asset_2:')
plt.show()


list_trajectories = [
    walk_1,
    walk_2,
    data[0],
    data[1]
]

list_label_trajectories = [
    f"Walk-1 :\nS0 = {S0.iloc[0,0]}\nm={mean_1}\nvol={vol_1}",
    f"Walk-2 :\nS0={S0.iloc[0,1]}\nm={mean_2}\nvol={vol_2}",
    f"Asset-1 :\nS0={data.iloc[0,0]}\nm={mean_1}\nvol={vol_1}",
    f"Asset-2 :\nS0={data.iloc[0,1]}\nm={mean_2}\nvol={vol_2}"
]

plot_trajectories(list_trajectories, list_label_trajectories)

