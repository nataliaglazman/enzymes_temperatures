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
deltaH_list = []
deltaC_list = []
deltaS_list = []
deltaH_error = []
deltaC_error = []
deltaS_error = []
        

def fit_model(data, plot_data, bootstrapping):
    x0 = 10, -100,1.3,-100
    #x1 = 1,1,1,1
    figure(figsize=(12, 10), dpi=3000)
    
    
    for index, i in enumerate(id_list):
        
        
        data_set = data.loc[data['originalid'] == i]
        T0 = data_set['tempkelvin'].loc[data_set['originaltraitvalue'].idxmax()] - 4
        Tmax = data_set['tempkelvin'].loc[data_set['originaltraitvalue'].idxmax()]
        tmin = data_set['tempkelvin'].min() 
        tmax = data_set['tempkelvin'].max()
        
        #Convert relative activity from a percentage to a decimal
        y_true = data_set['originaltraitvalue']/100

        
        #Create a fake temperature parameter for model plots
        dt = 0.5
        temp=np.arange(tmin, tmax, dt)
        
        
        t_true = data_set['tempkelvin'].array
        
        
        
        #Defining the model
        
        def model_Hobbs(T,p, deltaH, deltaC, deltaS):
            A = ((kB*T)/(p*h)) * np.exp(-((deltaH+(deltaC*(T-T0)))/(R*T)) + ((deltaS+(deltaC*np.log(T/T0)))/R))
            return A
        
        def model_EEAR(T, A0, Eb, EDH, EDC):
            A = A0*np.exp(-Eb-(EDH*(1-T/T0)+(EDC*(T-T0-(T*np.log(T/T0))))))/kB*T
            return A
        
        
        
        #Error propagation and error calculation functions
    
        def propagate_errors(T, diag_cov):
            dH = -1/(R*T)
            dC = (T-T0)/(R*T) + np.log(T/T0)/R
            dS = 1/R
            derivatives = np.array([dH, dC, dS])
            variance = np.dot(diag_cov, derivatives**2) 
            return np.sqrt(variance)
        
        
        def calculate_sigma():
            diff = []
            for i in range(len(t_true)-1):
                           
               diff.append(t_true[i+1] - t_true[i])
               
            mean = sum(diff)/len(diff)
            sigma = (mean/Tmax) * data_set['originaltraitvalue']
            
            return sigma
        
        
        sigma = calculate_sigma()
        
        
        
        #Model fitting using curve fit

        pfit, pcov = curve_fit(model_Hobbs, data_set['tempkelvin'], y_true, p0 = x0, method = 'lm')
        
        y_model = model_Hobbs(temp, *pfit)
        
        
        sigma_parameters = np.diagonal(pcov)

        #yerr = propagate_errors(temp, sigma_parameters)
        
        
        deltaH_list.append(pfit[1])
        deltaC_list.append(pfit[2])
        deltaS_list.append(pfit[3])
        
        sqrt_sigma_parameters = np.sqrt(sigma_parameters)
        
        deltaH_error.append(sqrt_sigma_parameters[1])
        deltaC_error.append(sqrt_sigma_parameters[2])
        deltaS_error.append(sqrt_sigma_parameters[3])

        

        
        ###TEST OF AN ALTERNATIVE MODEL
        #pfit_2, pcov_2 = curve_fit(model_EEAR, data_set['tempkelvin'], y_true,  p0 = x1, method = 'lm')

        
        #y_2 = model_EEAR(temp, *pfit_2)
        
        
    

#Using Lmfit to fit the models 
        
        pars = Parameters()
        pars.add('deltaH', value=-1000, min=-1000000, max=1000000)
        pars.add('deltaC', value= -10000, min=-10000, max=1)
        pars.add('deltaS', value= -100, min=-10000, max=10000)
        
    
    
        # model = lmfit.Model(model_Hobbs)
        

        # result = model.fit(y_true, T=data_set['tempkelvin'], params = pars, weights = 1/sigma)
        # print(result.fit_report())

        # #now calculate explicit 1-, 2, and 3-sigma uncertainties:
        # try:
        #     ci = result.conf_interval(sigmas=[1,2,3])
        #     lmfit.printfuncs.report_ci(ci)
        # except:
        #     print('error')
        
        
        

#Estimation of Topt and Tinf parameters
        
        Topt = (pfit[0] - (pfit[1]*T0))/(-pfit[1]-R)
        Topt_estim = T0-(pfit[0]/pfit[1])
        Tinf = (pfit[0] - (pfit[1]*T0))/(-pfit[1]+np.sqrt(-pfit[1]*R))
        
        
        #Converting kelvin to celsius
        
        Topt = Topt-273.15
        Tinf = Tinf-273.15
        Topt_estim = Topt_estim - 273.15
        
        
        #Adding parameters to dictionary
        
        Tinf_dictionary[i] = Tinf
        Topt_dictionary[i] = Topt
        Topt_estim_dictionary[i] = Topt_estim
        
        
        
    
        #Plot model and data if needed
        
        y_model_plot = y_model * 100
        
        if plot_data == True:
            
            if index in [0,1,2,3]:
    
                bound_upper = model_Hobbs(t_true, *(pfit + sigma_parameters))
                bound_lower = model_Hobbs(t_true, *(pfit - sigma_parameters))
            
                temp = temp-273.15
        
                plt.subplot(2, 2, index+1)
                plt.plot(temp, y_model_plot, label = 'model', linewidth = 2.5, color = 'limegreen')
                plt.plot(data_set['interactor1temp'], y_true*100, 'o', label='data', markersize = 5.5, color = 'r')
                #plt.fill_between(t_true-273.15, bound_lower, bound_upper,
                #        color = 'black', alpha = 0.15, edgecolor = 'black')
                #plt.fill_between(temp, y_model_plot+yerr, y_model_plot-yerr, color = 'black', alpha = 0.1, edgecolor = 'black')
        
            
                if bootstrapping == True:

                    nboot = 100
                    bspreds = np.zeros((nboot, y_true.size))
        
                    for b in range(nboot):
                        xb,yb = bootstrap(t_true,y_true)
                        p0, cov = curve_fit(model_Hobbs, xb, yb)
                        bspreds[b] = model_Hobbs(t_true,*p0)
                
                    plt.plot(t_true, bspreds.T, color = 'C0', alpha = 0.05)
                
                
                #plt.axvline(x = Topt, color = 'b', linestyle = 'dashed')
                #plt.axvline(x = Tinf, color = 'g', linestyle = 'dashed')
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



        
fit_model(data, plot_data = True, bootstrapping = False)

Topt_data = pd.DataFrame.from_dict(Topt_dictionary, orient = 'index')
Tinf_data = pd.DataFrame.from_dict(Tinf_dictionary, orient = 'index')
optgrowth_data = pd.DataFrame.from_dict(organism_enzyme_dict, orient = 'index')





#Plotting relationship between Topt and optimal growth temp for each organism

def plot_comparison():
    figure(figsize=(8, 7), dpi=3000)
    plt.scatter(Topt_data[0:18], Topt_data.index[0:18], label = 'Topt')
    plt.scatter(Tinf_data[0:18], Tinf_data.index[0:18], label = 'Tinf')
    plt.axvline(x = 30, linestyle = 'dashed')
    plt.legend()
    plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
    plt.ylabel('Enzyme ID', fontsize = 13, fontweight = 'bold')
    plt.title('Topt and Tinf comparison to optimal\n' + r'growth temperature of B. subtilis', fontsize = 15, fontweight = 'bold')
    plt.show()
    
#plot_comparison()


#Create dataframe for optimal parameter values and their condifence intervals
# parameters = pd.DataFrame(deltaH_list, index=id_list, columns = ['deltaH'])
# parameters['deltaC'] = deltaC_list
# parameters['deltaS'] = deltaS_list
# parameters['deltaHposerror'] = deltaH_error
# parameters['deltaCposerror'] = deltaC_error
# parameters['deltaSposerror'] = deltaS_error
    
    
    
    


