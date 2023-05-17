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
set_matplotlib_formats('svg')



#Import data
data_file_name = 'Database_v2.csv'

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
k = 8.62e-05
R = 8.314


#Define and fit model


organism_enzyme_dict = {}
Tinf_list = []
Topt_error_list =[]
Topt_list = []
Tinf_error_list = []
parameter_list = []
deltaH_list = []
deltaC_list = []
deltaS_list = []
deltaH_error = []
deltaC_error = []
deltaS_error = []
        

def fit_model(data, plot_data, bootstrapping):
    x0 = -100,1.3,-100
    x1 = 1400, 0.58, 2.2, 0.085
    figure(figsize=(12, 10), dpi=3000)
    
    
    for index, i in enumerate(id_list):
        
        
        data_set = data.loc[data['originalid'] == i]
        T0 = data_set['tempkelvin'].loc[data_set['originaltraitvalue'].idxmax()] + 4
        Tmax = data_set['tempkelvin'].loc[data_set['originaltraitvalue'].idxmax()]
        tmin = data_set['tempkelvin'].min() 
        tmax = data_set['tempkelvin'].max()
        
        #Convert relative activity from a percentage to a decimal
        y_true = data_set['originaltraitvalue']/100

        
        #Create a fake temperature parameter for model plots
        dt = 0.5
        temp=np.arange(tmin, tmax, dt)
        
        
        t_true = data_set['tempkelvin'].array
        
        
        tm = 340
        
        #Defining the model
        
        def model_Hobbs(T, deltaH, deltaC, deltaS):
            A = ((kB*T)/(h)) * np.exp(-((deltaH+(deltaC*(T-T0)))/(R*T)) + ((deltaS+(deltaC*np.log(T/T0)))/R))
            return A
        
        def model_EAAR(T, A0, eb, ef, ehc):  
            A = A0*np.exp(-((eb-(ef*(1-(T/tm)))+(ehc*(T-tm-(T*np.log(T/tm)))))/(k*T)))
            return A
        
        
        #Error propagation and error calculation functions
    
        def propagate_errors(T, diag_cov, p):
            dH = -1/(R*T)
            dC = (T-T0)/(R*T) + np.log(T/T0)/R
            dS = 1/R
            #dp = -1/p
            derivatives = np.array([dH, dC, dS])
            variance = np.dot(diag_cov, derivatives**2) 
            return np.sqrt(variance)
        
        
        def propagate_errors_2(T, diag_cov, A0):
            dA0 = 1/A0
            dEb = -1/(kB*T)
            dEDH = (1-(T/T0))/(kB*T)
            dEDC = (T-T0-(T*np.log(T/T0)))/(kB*T)
            derivatives = np.array([dEb, dEDH, dEDC, dA0])
            variance = np.dot(diag_cov, derivatives**2)
            return np.sqrt(variance)
            
        
        
        
        # def calculate_sigma():
        #     diff = []
        #     for i in range(len(t_true)-1):
                           
        #         diff.append(t_true[i+1] - t_true[i])
               
        #     mean = sum(diff)/len(diff)
        #     sigma = (mean/Tmax) * data_set['originaltraitvalue']
            
        #     return sigma
        
        
        # sigma = calculate_sigma()
        
        
        
        #Model fitting using curve fit

        pfit, pcov = curve_fit(model_Hobbs, data_set['tempkelvin'], y_true, p0 = x0, method = 'lm')
        
        y_model = model_Hobbs(temp, *pfit)
        
        sigma_parameters = np.diagonal(pcov)

        yerr = propagate_errors(temp, sigma_parameters, pfit[0])
        
        
        
        deltaH_list.append(pfit[0])
        deltaC_list.append(pfit[1])
        deltaS_list.append(pfit[2])
        
        sqrt_sigma_parameters = np.sqrt(sigma_parameters)
        
        deltaH_error.append(sqrt_sigma_parameters[0])
        deltaC_error.append(sqrt_sigma_parameters[1])
        deltaS_error.append(sqrt_sigma_parameters[2])
        


        

        
        ###TEST OF AN ALTERNATIVE MODEL
        pfit_2, pcov_2 = curve_fit(model_EAAR, data_set['tempkelvin'], y_true, p0 = x1, maxfev = 10000)
        # bounds=([-np.exp(40), -100,-6,-6], [np.exp(40),100,6,6]
        
        y_2 = model_EAAR(temp, *pfit_2)
        sigma_2 = np.sqrt(np.diag(pcov_2))
        
        # yerr_up = model_EAAR(temp, *(pfit_2+sigma_2))
        
        # yerr_down = model_EAAR(temp, *(pfit_2-sigma_2))
        
        # yerr2 = propagate_errors_2(temp, np.diag(pcov_2), pfit_2[0])
        
        #print(yerr2)
        
        

#Estimation of Topt and Tinf parameters
        
        Topt = (pfit[0] - (pfit[1]*T0))/(-pfit[1]-R)
        Topt_estim = T0-(pfit[0]/pfit[1])
        Tinf = (pfit[0] - (pfit[1]*T0))/(-pfit[1]+np.sqrt(-pfit[1]*R))
        
        #Error propagation
        Topt_error = (pfit[0]**2 * ((1/pfit[1])**2)) + (pfit[1]**2 * (((R*T0)-pfit[0])/(pfit[1]+R)**2)**2)
        Topt_error = np.sqrt(Topt_error)
        
        Tinf_error = (pfit[0]**2 * (1/(-pfit[1] + np.sqrt(-pfit[1]*R)))**2) + ((pfit[1]**2) * (((2*pfit[0]*np.sqrt(-pfit[1]*R))+ (R*T0*pfit[1])+(R*pfit[0]))/((2*np.sqrt(-R*pfit[1])) * (np.sqrt(-R*pfit[1])-pfit[1])**2))**2)
        Tinf_error = np.sqrt(Tinf_error)
        
        
        
        #Converting kelvin to celsius
        
        Topt = Topt-273.15
        Tinf = Tinf-273.15
        Topt_estim = Topt_estim - 273.15
        
        
        #Adding parameters to dictionary
        
        Topt_error_list.append(Topt_error)
        Topt_list.append(Topt)
        Tinf_list.append(Tinf)
        Tinf_error_list.append(Tinf_error)
        
        print(i + ': ' + str(Tinf) + ' , '+ str(Tinf_error))
        
    
        #Plot model and data if needed
        
        y_model_plot = y_model * 100
        
        if plot_data == True:
            
            if index in [0,1,2,3]:
    
            
                temp = temp-273.15
        
                plt.subplot(2, 2, index+1)
                plt.plot(temp, y_model_plot, label = 'model', linewidth = 2.5, color = 'limegreen')
                plt.plot(data_set['interactor1temp'], y_true*100, 'o', label='data', markersize = 5.5, color = 'r')
                #plt.errorbar(data_set['interactor1temp'], y_true*100,sigma)
                #plt.fill_between(temp, yerr_up*100, yerr_down*100,
                #       color = 'black', alpha = 0.15, edgecolor = 'black')
                #plt.fill_between(temp, (y_2+yerr2)*100, (y_2-yerr2)*100, color = 'black', alpha = 0.1, edgecolor = 'black')

                if bootstrapping == True:

                    nboot = 100
                    bspreds = np.zeros((nboot, y_true.size))
        
                    for b in range(nboot):
                        xb,yb = bootstrap(t_true,y_true)
                        p0, cov = curve_fit(model_Hobbs, xb, yb)
                        bspreds[b] = model_Hobbs(t_true,*p0)
                
                    plt.plot(t_true-273.15, bspreds.T*100, color = 'C0', alpha = 0.05)
                
                
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


#optgrowth_data = pd.DataFrame.from_dict(organism_enzyme_dict, orient = 'index')




#Plotting relationship between Topt and optimal growth temp for each organism

def plot_comparison():
    figure(figsize=(8, 7), dpi=3000)
    plt.scatter(Topt_list[0:18], id_list[0:18], label = 'Topt')
    plt.scatter(Tinf_list[0:18], id_list[0:18], label = 'Tinf')
    plt.errorbar(Topt_list[0:18], id_list[0:18], xerr=Topt_error_list[0:18], fmt="o")
    plt.errorbar(Tinf_list[0:18], id_list[0:18], xerr=Tinf_error_list[0:18], fmt="o")
    plt.axvline(x = 30, linestyle = 'dashed')
    plt.legend()
    plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
    plt.ylabel('Enzyme ID', fontsize = 13, fontweight = 'bold')
    plt.title('Topt and Tinf comparison to optimal\n' + r'growth temperature of B. subtilis', fontsize = 15, fontweight = 'bold')
    plt.show()
    
#plot_comparison()


#Create dataframe for optimal parameter values and their condifence intervals
parameters = pd.DataFrame(deltaH_list, index=id_list, columns = ['deltaH'])
parameters['deltaC'] = deltaC_list
parameters['deltaS'] = deltaS_list
parameters['deltaHposerror'] = deltaH_error
parameters['deltaCposerror'] = deltaC_error
parameters['deltaSposerror'] = deltaS_error


    


