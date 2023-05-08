#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:41:54 2023

@author: nataliaglazman
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import set_matplotlib_formats
from sklearn.utils import resample as bootstrap
import csv
set_matplotlib_formats('svg')



#Import data
data_file_name = '/Users/nataliaglazman/Library/Mobile Documents/com~apple~CloudDocs/Desktop/FYP/Database.csv'

data = pd.read_csv(data_file_name)

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

optimal_growth_temp_name = '/Users/nataliaglazman/Library/Mobile Documents/com~apple~CloudDocs/Desktop/FYP/optimal_growth_temp.csv'

optimal_growth_temp = pd.read_csv(optimal_growth_temp_name)
optimal_growth_temp = optimal_growth_temp.loc[:,['organism', 'optmumgrowthtemp']]


#Define set parameters

h = 6.62607015e-34
kB = 1.380649e-23
R = 8.314


#Define and fit model

Topt_dictionary = {}
Topt_estim_dictionary = {}
organism_enzyme_dict = {}
Tinf_dictionary = {}
parameter_list = []
RSS = []
        

def fit_model(data, plot_data, bootstrapping):
    x0 = -100,1.3,-100
    figure(figsize=(12, 10), dpi=3000)
    
    
    for index, i in enumerate(id_list):
        
        
        data_set = data.loc[data['originalid'] == i]
        T0 = data_set['tempkelvin'].loc[data_set['originaltraitvalue'].idxmax()] - 4
        tmin = data_set['tempkelvin'].min() 
        tmax = data_set['tempkelvin'].max()
        organisme = data_set['interactor1'].iloc[0]
        organism_enzyme_dict[i] = optimal_growth_temp.optmumgrowthtemp.loc[optimal_growth_temp.organism == organisme].array[0]
        
        
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
        Tinf = (pfit[0] - (pfit[1]*T0))/(-pfit[1]+np.sqrt(-pfit[1]*R))
        
        
        #Converting kelving to celsius
        
        Topt = Topt-273.15
        Tinf = Tinf-273.15
        Topt_estim = Topt_estim - 273.15
        
        
        #Adding parameters to dictionary
        
        Tinf_dictionary[i] = Tinf
        Topt_dictionary[i] = Topt
        Topt_estim_dictionary[i] = Topt_estim
    
    
    
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
        plt.suptitle('Hobbs et al. model fit', fontsize = 25)
        plt.tight_layout()
        plt.show()



        
fit_model(data, plot_data = False, bootstrapping = False)

Topt_data = pd.DataFrame.from_dict(Topt_dictionary, orient = 'index')
Tinf_data = pd.DataFrame.from_dict(Tinf_dictionary, orient = 'index')
optgrowth_data = pd.DataFrame.from_dict(organism_enzyme_dict, orient = 'index')



#Plotting relationship between topt and optimal growth temp

figure(figsize=(8, 7), dpi=3000)
plt.scatter(Topt_data[0:18], Topt_data.index[0:18], label = 'Topt')
plt.scatter(Tinf_data[0:18], Tinf_data.index[0:18], label = 'Tinf')
plt.axvline(x = 30, linestyle = 'dashed')
plt.legend()
plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
plt.ylabel('Enzyme ID', fontsize = 13, fontweight = 'bold')
plt.title('Topt and Tinf comparison to optimal\n' + r'growth temperature of Bacillus subtilis', fontsize = 15, fontweight = 'bold')
plt.show()


    
    
    


