# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 16:38:40 2023

@author: Nicolas Awlime
"""

#Option Valuation

#libraries
import numpy as np
import math
from scipy.stats import norm 
N = norm.cdf
import pandas as pd
import matplotlib.pyplot as plt

def BSM_option_valution( So, K, r, sigma, T, option=1): 
    """

    Parameters
    ----------
    So : TYPE = Numerical value
        DESCRIPTION = Current .
    K : TYPE
        DESCRIPTION.
    r : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    T : TYPE
        DESCRIPTION.
    option : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    Return a numerical value which corresponds to the value of call/put option

    """
    So = So #current price of the underlying
    K = K # strike
    r = r # risk free rate
    sigma = sigma # underling volatility
    T = T # Time to maturity
    d1 = (math.log(So/K) +( r-sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    
    
    if option==1:
        
        #we are valuing call
        c = N(d1)*So - math.exp(-r*T)*N(d2)*K
        return c
        
    else:
        #we are valuing put
        p =-( N(-d1)*So - math.exp(-r*T)*N(-d2)*K)
        return p


def main() :
    #visualize option delta
    So = np.linspace(50,150,10)
    So = pd.Series(So)
    K = 100
    r = 0.05
    sigma = 0.1
    T = 5 # units = years
    option = 1
    
    #Intitialize option prices
    opt = np.zeros(len(So))
    for i in range(len(opt)):
        opt[i]=BSM_option_valution(So[i],K,r,sigma,T,option)
        
    opt = pd.Series(opt)
    
    opt.index = So
    
    opt.index.name="Stock Price"
    opt.name = "Call" if option == 1 else  "Put"
    
    opt.plot()
    
    print("this figure shows the sensivity of our option price to underlying price change")
    
    # let's explore how delta behave when Implied Volatility (or sigma )changes 
    sigma_var= [0.02,0.04,0.06,0.08,0.1,0.12,0.14,0.16,0.18,0.2]
    opt = np.zeros((len(So),len(sigma_var)))
    
    for j in range(len(sigma_var)):
        for i in range(len(So)):
            opt[i,j]=BSM_option_valution(So[i],K,r,sigma_var[j],T,option)
            
    opt = pd.DataFrame(opt)
    opt.columns = [f"IV of {i*100}%" for i in sigma_var]
    opt.index = So
    opt.index.name="Stock Price"
    plt.ylabel("Option Value")
    opt.plot()
    
    print("We can visualize how moneyness of our option and IV are related")
    

if "__init___"=="name":
    main()
