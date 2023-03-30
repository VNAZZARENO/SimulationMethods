# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:15:22 2023

@author: vince
"""

# PART IV #
# Q1 #
aj = [9, 9, 2, 1]
b = 10

def bary_expansion(base_list, base):
    # convert the base_list to a number k in base 10
    k = 0
    for i in range(len(base_list)):
        k += base_list[i] * (base ** i)
    
    # find k+1 in base 10
    k_plus_1 = k + 1
    
    # convert k+1 to base b
    k_final = []
    while k_plus_1 != 0:
        k_final.append(k_plus_1 % base) 
        k_plus_1 = k_plus_1 // base
    
    return k_final


print(bary_expansion(aj, b))


# Q2 #

def van_der_corput(k, base):
    # convert k to the given base
    base_list = []
    while k != 0:
        base_list.append(k % base)
        k = k // base
    
    # calculate the value using Horner's method
    value = 0
    for i in range(len(base_list)-1, -1, -1):
        value = (value + base_list[i]) / base
    
    return value


k = 11
base = 10
for i in range(1, k + 1):
    element_value = van_der_corput(i, base)
    print(f"Element {i} in base {base} is equal to: {element_value:.5f} in the Van der Corput sequence")



# Q3 #

import numpy as np
import seaborn as sns

n = 100000
s0 = 100
K = 110
T = 1
r = 0.05
sigma = 0.2
mu = 9.25
constant = 500

# generate a Van der Corput sequence of size n
vdc = np.array([van_der_corput(i, 10) for i in range(1, n+1)])

# generate prices according to the given dynamics
sigma_sqrt_T = sigma * np.sqrt(T)
S_T = s0 * np.exp((r - sigma**2 / 2) * T + sigma_sqrt_T * vdc)
S_T_mu = s0 * np.exp((mu - sigma**2 / 2) * T + sigma_sqrt_T * vdc)

# plot the generated prices
sns.histplot(data=S_T, color="green")
sns.histplot(data=S_T_mu, color="red")

# calculate the Radon-Nikodym derivative
rn_der = np.exp(-((np.log(S_T_mu / s0) - (r - sigma**2 / 2) * T)**2 / (2 * sigma**2 * T))
               + ((np.log(S_T_mu / s0) - (mu - sigma**2 / 2) * T)**2 / (2 * sigma**2 * T))
               + constant)

# calculate the option prices using Monte Carlo method
payoff = np.maximum(S_T_mu - K, 0)
call_price = np.mean(np.exp(-r * T) * payoff * rn_der)

digital_payoff = np.where(payoff > 0, 1, payoff)
digital_price = np.mean(np.exp(-r * T) * digital_payoff * rn_der)

# calculate the ratio of prices
price_ratio = call_price / digital_price

# print the results
print(f"The ratio of prices is: {price_ratio:.5f}")

