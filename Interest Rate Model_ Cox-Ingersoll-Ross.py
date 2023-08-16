# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 10:26:40 2023

@author: Nicolas Awlime
"""
from scipy.stats import norm 
import math 
import random
import numpy as np
import pandas as pd

# interest rates models

# Cox-Ingersoll-Ross (CIR) Model

def CIR_model(r:float,b:float,sigma:float,dt:float,a:float=0.8)-> float: 
    #this function return the change in interest rate according to cox_ingersoll
    #-ross model
    #the idea : is that interest rate should revert to long term level (b)
    # so absolute change in interest rate is difference of short term rate (r)
    #from short rate. And we adjust the movement toward the long term rate by a 
    # factor (a)
    # also we assume that the movement toward the long term rate can be modify by 
    # randomness term which depends on current rate
    
    drift = a*(b-r)*dt
    random_term = sigma*math.sqrt(r*dt)*norm.ppf(random.random())
    
    #dr = drift + random_term
    dr = drift + random_term
    return dr


def simulation(n:int,inv_dt:int,r:float,b:float,a:float=0.8):
    """
    

    Parameters
    ----------
    n : int
        DESCRIPTION.
    inv_dt : int
        which is the inverse term of dt so correspond to the number of period 
        we want in one year.

    Returns
    -------
    None.

    """
    simu = np.zeros((inv_dt,n))
    
    for j in range(n):
        for i in range(inv_dt):
            if i ==0:
                simu[i,j]=r
            else:
                simu[i,j]= simu[i-1,j] + CIR_model(simu[i-1,j],b,1/inv_dt,a)
    
    simu = pd.DataFrame(simu) 
    simu.index.name = "Time"
    simu.plot()
    return    print("Simulation is done")       

def main():
    simulation(5,20,0.05,0.05)
    
if __init__ =="__name__":
    main()
