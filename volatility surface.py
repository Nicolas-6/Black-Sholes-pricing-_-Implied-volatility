# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 11:04:26 2022

@author: 06nic
"""
#credit : https://www.codearmo.com/blog/implied-volatility-european-call-python


import numpy as np
from scipy.stats import norm

N_prime = norm.pdf
N = norm.cdf

def black_scholes_call(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: call price
    '''

    ###standard black-scholes formula
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call = S * N(d1) -  N(d2)* K * np.exp(-r * T)
    return call

N_prime = norm.pdf


def vega(S, K, T, r, sigma):
    '''

    :param S: Asset price
    :param K: Strike price
    :param T: Time to Maturity
    :param r: risk-free rate (treasury bills)
    :param sigma: volatility
    :return: partial derivative w.r.t volatility
    '''

    ### calculating d1 from black scholes
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / sigma * np.sqrt(T)

    
    vega = S  * np.sqrt(T) * N_prime(d1)
    return vega

def implied_volatility_call(C, S, K, T, r, tol=0.0001,
                            max_iterations=100):
    '''

    :param C: Observed call price
    :param S: Asset price
    :param K: Strike Price
    :param T: Time to Maturity
    :param r: riskfree rate
    :param tol: error tolerance in result
    :param max_iterations: max iterations to update vol
    :return: implied volatility in percent
    '''


    ### assigning initial volatility estimate for input in Newton_rap procedure
    sigma = 0.3

    for i in range(max_iterations):

        ### calculate difference between blackscholes price and market price with
        ### iteratively updated volality estimate
        diff = black_scholes_call(S, K, T, r, sigma) - C

        ###break if difference is less than specified tolerance level
        if abs(diff) < tol:
            print(f'found on {i}th iteration')
            print(f'difference is equal to {diff}')
            break

        ### use newton rapshon to update the estimate
        sigma = sigma - diff / vega(S, K, T, r, sigma)

    return sigma


n = 10
T =  [t/n for t in range (5+1)]
K = [ k for k in range(90,115,2)]
Vol_surf = np.ones(shape=(len(T), len(K)))
observed_price = 18
S = 100
r = 0.05


i = 0
for t in T:
    j = 0
    for k in K:
        
        int_sigma =  0.3 
        Vol_surf[i,j] =implied_volatility_call(observed_price, S, k, t, r, tol=0.0001,max_iterations=100)
        print(Vol_surf[i,j])
        j = j + 1
    i = i + 1  
        

print(Vol_surf)

import matplotlib.pyplot as plt

fig = plt.figure(figsize = (12,10))
ax = plt.axes(projection='3d')

x = T
y = K
Y,X = np.meshgrid(y, x)
Z = Vol_surf
surf = ax.plot_surface(X, Y, Z, cmap = plt.cm.cividis)
ax.set_title('Volatility Surface')

# Set axes label
ax.set_xlabel('Time to maturity', labelpad=20)
ax.set_ylabel('Strike / Moneyness', labelpad=20)
ax.set_zlabel('Volatility', labelpad=20)