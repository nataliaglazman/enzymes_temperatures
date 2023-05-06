#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:41:54 2023

@author: nataliaglazman
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from IPython.display import set_matplotlib_formats
from matplotlib.pyplot import figure
from sklearn.utils import resample as bootstrap
import csv
set_matplotlib_formats('svg')



#Import data
file_name = '/Users/nataliaglazman/Library/Mobile Documents/com~apple~CloudDocs/Desktop/FYP/Database.csv'

data = pd.read_csv(file_name).dropna(axis =1)

data = data.loc[:,['originalid','originaltraitvalue','interactor1',
            'interactor1wholeparttype','interactor1temp',
            'interactor1wholeparttype',
            'interactor1kingdom', 'interactor1phylum', 'interactor1class',
            'interactor1order', 'interactor1family', 'interactor1genus',
            'interactor1species']]

data['lnA'] = data['originaltraitvalue'].apply(lambda x: np.log(x) if x != 0 else x)
data['tempkelvin'] = data['interactor1temp'] + 273.15
id_list = data.originalid.unique()
organism_list = data.interactor1.unique()

#Define set parameters and model function

h = 6.62607015e-34
kB = 1.380649e-23
R = 8.314


#Define and fit model

Topt_dictionary = {}
Topt_estim_dictionary = {}
Tinf_dictionary = {}
parameter_list = []
RSS = []
        

def fit_model(data, plot_data, bootstrapping):
    x0 = -100,1.3,-100
    figure(figsize=(12, 10), dpi=3000)
    
    
    for index, i in enumerate(id_list[0:4]):
        
        
        data_set = data.loc[data['originalid'] == i]
        T0 = data_set['tempkelvin'].loc[data_set['originaltraitvalue'].idxmax()] - 4
        tmin = data_set['tempkelvin'].min() 
        tmax = data_set['tempkelvin'].max()
        dt = 0.5
        t=np.arange(tmin, tmax, dt)
        
        
        #Defining the model
        
        def model_log(T, deltaH, deltaC, deltaS):
            A = np.log((kB*T)/(h)) - ((deltaH + deltaC)*(T - T0))/(R * T) + ((deltaS + deltaC * np.log(T/T0))/R)
            return np.exp(A)
        
        
        
        #Fit model to data and get estimates for Topt and Tinf
        
        
        pfit, pcov = curve_fit(model_log,data_set['tempkelvin'], data_set['originaltraitvalue'],  p0 = x0, method = 'trf')
        
        sigma_ab = np.sqrt(np.diagonal(pcov))
        
        Topt = (pfit[0] - (pfit[1]*T0))/(-pfit[1]-R)
        Topt_estim = T0-(pfit[0]/pfit[1])
        Topt_dictionary[i] = Topt
        Topt_estim_dictionary[i] = Topt_estim
        Tinf = (pfit[0] - (pfit[1]*T0))/(-pfit[1]+np.sqrt(-pfit[1]*R))
        Tinf_dictionary[i] = Tinf
        Topt = Topt-273.15
        Tinf = Tinf-273.15
        Topt_estim = Topt_estim - 273.15
    
    
        
        
        #Plot model and data if needed
        
        
        if plot_data == True:
    
            y = model_log(t, *pfit)

            y_true = data_set['originaltraitvalue']
            t_true = data_set['tempkelvin']
            bound_upper = model_log(t_true, *(pfit + sigma_ab))
            bound_lower = model_log(t_true, *(pfit - sigma_ab))
            
            t = t-273.15
        
            plt.subplot(2, 2, index+1)
            plt.plot(t, y, label = 'model', linewidth = 2.5, color = 'limegreen')
            plt.plot(data_set['interactor1temp'], data_set['originaltraitvalue'], 'o', label='data', markersize = 5.5, color = 'r')
            plt.fill_between(t_true-273, bound_lower, bound_upper,
                     color = 'black', alpha = 0.15, edgecolor = 'black')
        
            
            if bootstrapping == True:

                nboot = 100
                bspreds = np.zeros((nboot, y_true.size))
        
                for b in range(nboot):
                    xb,yb = bootstrap(t_true,y_true)
                    p0, cov = curve_fit(model_log, xb, yb)
                    bspreds[b] = model_log(t_true,*p0)
                
                plt.plot(t_true, bspreds.T, color = 'C0', alpha = 0.05)
                
                
            plt.axvline(x = Topt, color = 'b', linestyle = 'dashed')
            plt.axvline(x = Tinf, color = 'g', linestyle = 'dashed')
            plt.legend()
            plt.title(i[0:3], fontsize = 15, fontweight = 'bold')
            plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
            plt.ylabel('Relative activity (%)', fontsize = 13, fontweight = 'bold')
        
        
    if plot_data == True:
        
        plt.rcParams['figure.dpi'] = 2000
        plt.rcParams['savefig.dpi'] = 2000
        plt.suptitle('Hobbs et al. model fit with bootstrapping', fontsize = 25)
        plt.tight_layout()
        plt.show()



        
fit_model(data[0:32], plot_data = True, bootstrapping = False)








# with open('Topt.csv', 'w') as f: 
#      w = csv.DictWriter(f, Topt_dictionary.keys())
#      w.writeheader()
#      w.writerow(Topt_dictionary)
    
    
    

