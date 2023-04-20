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
set_matplotlib_formats('svg')
import csv


#Import data
file_name = '/Users/nataliaglazman/Desktop/FYP/VecTraits-Natalia-final_3.csv'

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
        

def fit_model(data):
    x0 = 25000,-100,6000,-100
    figure(figsize=(12, 8), dpi=3000)


    
    for index, i in enumerate(id_list[0:4]):
        
        
        data_set = data.loc[data['originalid'] == i]
        T0 = data_set['tempkelvin'].loc[data_set['originaltraitvalue'].idxmax()] - 4
        tmin = data_set['tempkelvin'].min() 
        tmax = data_set['tempkelvin'].max()
        dt = 1
        t=np.arange(tmin, tmax, dt)
        
        
        def model_exp(T, p, deltaH, deltaC, deltaS):
            A = (np.sqrt(((kB*T)/(h*p))) * np.exp(((-((deltaH - deltaC)*(T - T0))/(R * T)) + ((deltaS - deltaC * np.log(T/T0))/R)))/2)                                              
            return A
        
        def residuals(q):
            p, deltaH, deltaC, deltaS = q
            return data_set['originaltraitvalue'] - model_exp(data_set['tempkelvin'], p, deltaH, deltaC, deltaS)
        
        pfit = least_squares(residuals, x0, method = 'lm')
        
        
        Topt = (pfit.x[1] - (pfit.x[2]*T0))/(-pfit.x[2]-R)
        Topt_estim = T0-(pfit.x[1]/pfit.x[2])
        Topt_dictionary[i] = Topt
        Topt_estim_dictionary[i] = Topt_estim
        Tinf = (pfit.x[2] - (pfit.x[3]*T0))/(-pfit.x[3]+np.sqrt(-pfit.x[3]*R))
        Tinf_dictionary[i] = Tinf
        
        
        parameter_list.append([pfit.x[0], pfit.x[1], pfit.x[2], pfit.x[3]])
        
        # x = data_set['tempkelvin']
        # y = data_set['originaltraitvalue']
        # y_pred = model_exp(x, pfit.x[0], pfit.x[1], pfit.x[2], pfit.x[3])
        RSS.append(sum(pfit.fun**2))
        
        
#Plot data and model

        plt.subplot(2, 2, index+1)
        plt.plot(t, model_exp(t, pfit.x[0], pfit.x[1], pfit.x[2], pfit.x[3]), label = 'model')
        plt.plot(data_set['tempkelvin'], data_set['originaltraitvalue'], 'x', label='data')
        plt.axvline(x = Topt, color = 'b', label = 'Topt', linestyle = 'dashed')
        plt.text(1,1,'Topt',rotation=90, ha='right', va='center')
        plt.legend()
        plt.title(i[0:3], fontsize = 15)
        plt.xlabel('Temperature (K)', fontsize = 13)
        plt.ylabel('Relative activity (%)', fontsize = 13)
        
        
    plt.rcParams['figure.dpi'] = 2000
    plt.rcParams['savefig.dpi'] = 2000
    plt.tight_layout()
    plt.show()
    
    
    
        
fit_model(data[0:31])

# for i,m in Topt_dictionary.items():
#     Topt_dictionary[i] = m-273.15
    
# with open('Topt.csv', 'w') as f: 
#     w = csv.DictWriter(f, Topt_dictionary.keys())
#     w.writeheader()
#     w.writerow(Topt_dictionary)
    


