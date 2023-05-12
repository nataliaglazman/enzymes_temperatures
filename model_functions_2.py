#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 15:41:54 2023

@author: nataliaglazman
"""

import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from IPython.display import set_matplotlib_formats
from sklearn.utils import resample as bootstrap
from scipy.stats.distributions import  t
import lmfit
from lmfit import Parameters
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
    
    
    for index, i in enumerate(id_list[4:8]):
        
        
        data_set = data.loc[data['originalid'] == i]
        T0 = data_set['tempkelvin'].loc[data_set['originaltraitvalue'].idxmax()] - 4
        tmin = data_set['tempkelvin'].min() 
        tmax = data_set['tempkelvin'].max()
        organisme = data_set['interactor1'].iloc[0]
        organism_enzyme_dict[i] = optimal_growth_temp.optmumgrowthtemp.loc[optimal_growth_temp.organism == organisme].array[0]
        
        
        dt = 0.5
        temp=np.arange(tmin, tmax, dt)
        
        
        
        #Defining the model1
        
        def model_exp(T, deltaH, deltaC, deltaS):
            A = ((kB*T)/h) * np.exp(-((deltaH+(deltaC*(T-T0)))/(R*T)) + ((deltaS+(deltaC*np.log(T/T0)))/R))
            return A
        
        
        def error_prop(T, diag_cov):
            dH = -1/(R*T)
            dC = (T-T0)/(R*T) + np.log(T/T0)/R
            dS = 1/R
            derivatives = np.array([dH, dC, dS])
            variance = np.dot(diag_cov, derivatives**2) 
            return np.sqrt(variance)
        
        #Fit model to data and get estimates for Topt and Tinf
        
        pfit, pcov = curve_fit(model_exp,data_set['tempkelvin'], data_set['originaltraitvalue'],  p0 = x0, method = 'trf')
        sigma_ab = np.diagonal(pcov)
        

        y = model_exp(temp, *pfit)
        
        yerr = error_prop(temp, sigma_ab)
        print(yerr)
        
        
        
        
        
        alpha = 0.05 # 95% confidence interval
        N = len(y)
        P = len(pfit)
        dof = max(0,N-P)
        # dof is the degrees of freedom
        
        tval = t.ppf((1 - alpha / 2), dof)

        
        
        # pars = Parameters()
        # pars.add('deltaH', value=-300, min=-100000, max=100)
        # pars.add('deltaC', value = -100, min=-100000, max=100)
        # pars.add('deltaS', value = 1, min=-100, max=100)
        # model = lmfit.Model(model_exp)

        # result = model.fit(data_set['originaltraitvalue'], pars, T=data_set['tempkelvin'])
        # print(result.fit_report())

        # # now calculate explicit 1-, 2, and 3-sigma uncertainties:
        # ci = result.conf_interval(sigmas=[1,2,3])
        # lmfit.printfuncs.report_ci(ci)

        
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
    

            y_true = data_set['originaltraitvalue']
            t_true = data_set['tempkelvin']
            #bound_upper = model_exp(t_true, *(pfit + sigma_ab*tval))
            #bound_lower = model_exp(t_true, *(pfit - sigma_ab*tval))
            
            temp = temp-273.15
        
            plt.subplot(2, 2, index+1)
            plt.plot(temp, y, label = 'model', linewidth = 2.5, color = 'limegreen')
            plt.plot(data_set['interactor1temp'], data_set['originaltraitvalue'], 'o', label='data', markersize = 5.5, color = 'r')
            #plt.fill_between(t_true-273, bound_lower, bound_upper,
            #         color = 'black', alpha = 0.15, edgecolor = 'black')
            plt.fill_between(temp, y+yerr, y-yerr, color = 'black', alpha = 0.1, edgecolor = 'black')
        
            
            if bootstrapping == True:

                nboot = 100
                bspreds = np.zeros((nboot, y_true.size))
        
                for b in range(nboot):
                    xb,yb = bootstrap(t_true,y_true)
                    p0, cov = curve_fit(model_exp, xb, yb)
                    bspreds[b] = model_exp(t_true,*p0)
                
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



        
fit_model(data[32:68], plot_data = True, bootstrapping = False)

Topt_data = pd.DataFrame.from_dict(Topt_dictionary, orient = 'index')
Tinf_data = pd.DataFrame.from_dict(Tinf_dictionary, orient = 'index')
optgrowth_data = pd.DataFrame.from_dict(organism_enzyme_dict, orient = 'index')



#Plotting relationship between topt and optimal growth temp

def plot_comparison():
    figure(figsize=(8, 7), dpi=3000)
    plt.scatter(Topt_data[45:50], Topt_data.index[45:50], label = 'Topt')
    plt.scatter(Tinf_data[45:50], Tinf_data.index[45:50], label = 'Tinf')
    plt.axvline(x = 15, linestyle = 'dashed')
    plt.legend()
    plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
    plt.ylabel('Enzyme ID', fontsize = 13, fontweight = 'bold')
    plt.title('Topt and Tinf comparison to optimal\n' + r'growth temperature of Psychrobacter sp. ANT206', fontsize = 15, fontweight = 'bold')
    plt.show()
    
#plot_comparison()

    
    
    
    
    


