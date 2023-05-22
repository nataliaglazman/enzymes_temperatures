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
import lmfit 
from lmfit import Model
from statsmodels.graphics.gofplots import qqplot
from scipy.stats import shapiro
from scipy.stats import chisquare
from scipy.stats.distributions import t

set_matplotlib_formats('svg')
np.set_printoptions(suppress=True)

#Import data
data_file_name = 'Database_v2.csv'

data = pd.read_csv(data_file_name)

data = data.loc[:,['originalid', 'originaltraitname','originaltraitvalue','interactor1','interactor1temp',
            'interactor1species']]

data['log_activity'] = data['originaltraitvalue'].apply(lambda x: np.log(x) if x != 0 else x)
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
Topt_list = []
Tinf_confint_list = []
Topt_confint_list =[]
deltaH_list = []
deltaC_list = []
deltaS_list = []
deltaH_error = []
deltaC_error = []
deltaS_error = []
enzymes_with_nonnormal_errors = []

        

        
        
#Error propagation and error calculation functions
def propagate_errors(T,T0, diag_cov):
    dH = -1/(R*T)
    dC = (T-T0)/(R*T) + np.log(T/T0)/R
    dS = 1/R
    #dp = -1/p
    derivatives = np.array([dH, dC, dS])
    variance = np.dot(diag_cov, derivatives**2) 
    return np.sqrt(variance) 

def propagate_errors_2(T, T0, diag_cov, A0):
    dA0 = 1/A0
    dEb = -1/(kB*T)
    dEDH = (1-(T/T0))/(kB*T)
    dEDC = (T-T0-(T*np.log(T/T0)))/(kB*T)
    derivatives = np.array([dEb, dEDH, dEDC, dA0])
    variance = np.dot(diag_cov, derivatives**2)
    return np.sqrt(variance)
        
        
def test_error_normality(i, y_true, y_predicted):
    residuals = y_true-y_predicted
    stat, p = shapiro(residuals)
    alpha = 0.05
    if p <= alpha:
        enzymes_with_nonnormal_errors.append(i)
        


def fit_model(data, plot_data, bootstrapping):
    x0 = -100,1.3,-100
    x1 = 14000, 0.58, 0.8, -0.085
    figure(figsize=(12, 10), dpi=3000)
    
    
    for index, i in enumerate(id_list):
        
        def model_Hobbs(T, deltaH, deltaC, deltaS):
            A = ((kB*T)/(h)) * np.exp(-((deltaH+(deltaC*(T-T0)))/(R*T)) + ((deltaS+(deltaC*np.log(T/T0)))/R))
            return A
                
        def model_EAAR(T, A0, eb, ef, ehc):  
            A = A0*np.exp(-((eb-(ef*(1-(T/T0)))+(ehc*(T-T0-(T*np.log(T/T0)))))/(k*T)))
            return A
        
                
        data_set = data.loc[data['originalid'] == i]
        T0 = data_set['tempkelvin'].loc[data_set['originaltraitvalue'].idxmax()] + 4
        tmin = data_set['tempkelvin'].min() 
        tmax = data_set['tempkelvin'].max()
                
                
        #Normalize non-relative data 
        if data_set['originaltraitname'].array[0] != 'relative activity':
            y_true = (data_set['originaltraitvalue'] - data_set['originaltraitvalue'].min()) / (data_set['originaltraitvalue'].max() - data_set['originaltraitvalue'].min())
            y_true = y_true *100
        else:
            y_true = data_set['originaltraitvalue']
                    
            
                
        #Create a fake temperature parameter for model plots
        dt = 0.5
        temp=np.arange(tmin, tmax, dt)
        t_true = data_set['tempkelvin'].array
            
        
        
        #Model fitting using curve fit

        pfit, pcov = curve_fit(model_Hobbs, t_true,  y_true, p0 = x0, method = 'lm')
        
        y_predicted = model_Hobbs(t_true, *pfit)
        y_model = model_Hobbs(temp, *pfit)
        
        
        #Error estimation and propagation
        test_error_normality(i, y_true, y_predicted)
        
        sigma_parameters = np.diagonal(pcov)
        yerr = propagate_errors(temp, T0, sigma_parameters)
        # yerr_up = y_model + yerr
        # yerr_down = y_model - yerr
        y_true = data_set['originaltraitvalue'].array
        chi_sqrt = 0
        for i in range(len(y_true)):
            chi_sqrt = chi_sqrt +  ((y_true[i] - y_predicted[i])**2)/y_predicted[i]
        #chi_Hobbs = np.sqrt((y_true-y_predicted)**2/y_predicted)
        #chi_Hobbs = chisquare(y_true, y_predicted)
        
        
        print(chi_sqrt)
        
        ###TEST OF EAAR MODEL
        pfit_2, pcov_2 = curve_fit(model_EAAR, t_true, y_true, p0 = x1)
        
        y_2 = model_EAAR(temp, *pfit_2)
        y_predicted_2 = model_EAAR(t_true, *pfit_2)
        sigma_2 = np.sqrt(np.diag(pcov_2))
        #chi_EAAR = np.sqrt((y_true-y_predicted_2)/y_predicted_2)
        
        # yerr_up = model_EAAR(temp, *(pfit_2+sigma_2))
        
        # yerr_down = model_EAAR(temp, *(pfit_2-sigma_2))
        
        yerr2 = propagate_errors_2(temp, T0, np.diag(pcov_2), pfit_2[0])
        yerr_up = y_2 + yerr2
        yerr_down = y_2 - yerr2
        
        

        #Estimation of Topt and Tinf parameters
        
        Topt = (pfit[0] - (pfit[1]*T0))/(-pfit[1]-R)
        Topt_estim = T0-(pfit[0]/pfit[1])
        Tinf = (pfit[0] - (pfit[1]*T0))/(-pfit[1]+np.sqrt(-pfit[1]*R))
        
        
        Topt = Topt-273.15
        Tinf = Tinf-273.15
        Topt_estim = Topt_estim - 273.15
        
        #Error propagation 
        alpha = 0.05
        N = len(y_true)
        P = len(pfit)
        dof = max(0,N-P)
        tval = t.ppf((1-alpha/2), dof)
        
        
        Topt_error = (pfit[0]**2 * ((1/pfit[1])**2)) + (pfit[1]**2 * (((R*T0)-pfit[0])/(pfit[1]+R)**2)**2)
        Topt_error = np.sqrt(np.sqrt(Topt_error))
        Topt_confint = Topt_error * tval
        
        Tinf_error = (pfit[0]**2 * (1/(-pfit[1] + np.sqrt(-pfit[1]*R)))**2) + ((pfit[1]**2) * (((2*pfit[0]*np.sqrt(-pfit[1]*R))+ (R*T0*pfit[1])+(R*pfit[0]))/((2*np.sqrt(-R*pfit[1])) * (np.sqrt(-R*pfit[1])-pfit[1])**2))**2)
        Tinf_error = np.sqrt(np.sqrt(Tinf_error))
        Tinf_confint = Tinf_error * tval
    
        
        #Adding parameters to dictionary
        
        
        Topt_list.append(Topt)
        Tinf_list.append(Tinf)
        Tinf_confint_list.append(Tinf_confint)
        Topt_confint_list.append(Topt_confint)
        
        #print(i + ': ' + str(pcov) + ' , ')
    


        #Plot model and data if needed
        
        
        if plot_data == True:
            
            y = y_predicted
            
            temp = temp-273.15
            index_list = list(range(4, len(id_list), 5))
            index_list_2 = list(range(1, len(id_list), 5))
            index_list_3 = list(range(2, len(id_list), 5))
            index_list_4 = list(range(3, len(id_list), 5))
            
            if index == 0:
                
                plt.subplot(2, 2, 1)
                plt.plot(temp, y, label = 'model', linewidth = 2.5, color = 'limegreen')
                plt.plot(data_set['interactor1temp'], y_true, 'o', label='data', markersize = 5.5, color = 'r')
                plt.fill_between(temp, yerr_up, yerr_down,
                       color = 'black', alpha = 0.15, edgecolor = 'black')
                
                plt.title(i[0:3], fontsize = 15, fontweight = 'bold')
                plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
                plt.ylabel('Relative activity (%)', fontsize = 13, fontweight = 'bold')
                plt.axvline(x = Topt, color = 'b', linestyle = 'dashed', label = 'Topt')
                plt.axvline(x = Tinf, color = 'g', linestyle = 'dashed', label = 'Tinf')
                plt.legend()
            
            if index in index_list:
                

                plt.rcParams['figure.dpi'] = 2000
                plt.rcParams['savefig.dpi'] = 2000
                plt.suptitle('Hobbs et al. model fit', fontsize = 25)
                plt.tight_layout()
                plt.show()
                
                figure(figsize=(12, 10), dpi=4000)
                plt.subplot(2, 2, 1)
                plt.plot(temp, y, label = 'model', linewidth = 2.5, color = 'limegreen')
                plt.plot(data_set['interactor1temp'], y_true, 'o', label='data', markersize = 5.5, color = 'r')
                plt.fill_between(temp, yerr_up, yerr_down,
                       color = 'black', alpha = 0.15, edgecolor = 'black')
                plt.title(i[0:3], fontsize = 15, fontweight = 'bold')
                plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
                plt.ylabel('Relative activity (%)', fontsize = 13, fontweight = 'bold')
                plt.axvline(x = Topt, color = 'b', linestyle = 'dashed', label = 'Topt')
                plt.axvline(x = Tinf, color = 'g', linestyle = 'dashed', label = 'Tinf')
                plt.legend()
                
            elif index in index_list_2:
                
                plt.subplot(2,2,2)
                plt.plot(temp, y, label = 'model', linewidth = 2.5, color = 'limegreen')
                plt.plot(data_set['interactor1temp'], y_true, 'o', label='data', markersize = 5.5, color = 'r')
                plt.fill_between(temp, yerr_up, yerr_down,
                       color = 'black', alpha = 0.15, edgecolor = 'black')
                plt.title(i[0:3], fontsize = 15, fontweight = 'bold')
                plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
                plt.ylabel('Relative activity (%)', fontsize = 13, fontweight = 'bold')
                plt.axvline(x = Topt, color = 'b', linestyle = 'dashed', label = 'Topt')
                plt.axvline(x = Tinf, color = 'g', linestyle = 'dashed', label = 'Tinf')
                plt.legend()
                
            elif index in index_list_3:
                plt.subplot(2,2,3)
                plt.plot(temp, y, label = 'model', linewidth = 2.5, color = 'limegreen')
                plt.plot(data_set['interactor1temp'], y_true, 'o', label='data', markersize = 5.5, color = 'r')
                plt.fill_between(temp, yerr_up, yerr_down,
                       color = 'black', alpha = 0.15, edgecolor = 'black')
                plt.title(i[0:3], fontsize = 15, fontweight = 'bold')
                plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
                plt.ylabel('Relative activity (%)', fontsize = 13, fontweight = 'bold')
                plt.axvline(x = Topt, color = 'b', linestyle = 'dashed', label = 'Topt')
                plt.axvline(x = Tinf, color = 'g', linestyle = 'dashed', label = 'Tinf')
                plt.legend()
                
            elif index in index_list_4:
                plt.subplot(2,2,4)
                plt.plot(temp, y, label = 'model', linewidth = 2.5, color = 'limegreen')
                plt.plot(data_set['interactor1temp'], y_true, 'o', label='data', markersize = 5.5, color = 'r')
                plt.fill_between(temp, yerr_up, yerr_down,
                       color = 'black', alpha = 0.15, edgecolor = 'black')
                plt.title(i[0:3], fontsize = 15, fontweight = 'bold')
                plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
                plt.ylabel('Relative activity (%)', fontsize = 13, fontweight = 'bold')
                plt.axvline(x = Topt, color = 'b', linestyle = 'dashed', label = 'Topt')
                plt.axvline(x = Tinf, color = 'g', linestyle = 'dashed', label = 'Tinf')
                plt.legend()
            
                
                # if bootstrapping == True:

                #     nboot = 100
                #     bspreds = np.zeros((nboot, y_true.size))
        
                #     for b in range(nboot):
                #         xb,yb = bootstrap(t_true,y_true)
                #         p0, cov = curve_fit(model_Hobbs, xb, yb)
                #         bspreds[b] = model_Hobbs(t_true,*p0)
                
                #     plt.plot(t_true-273.15, bspreds.T*100, color = 'C0', alpha = 0.05)
                
                



        
fit_model(data, plot_data = False, bootstrapping = False)


#Plotting relationship between Topt and optimal growth temp for each organism

def plot_comparison():
    figure(figsize=(8, 7), dpi=3000)
    plt.scatter(Topt_list[0:18], id_list[0:18], label = 'Topt')
    plt.scatter(Tinf_list[0:18], id_list[0:18], label = 'Tinf')
    plt.errorbar(Topt_list[0:18], id_list[0:18], xerr=Topt_confint_list[0:18], fmt="o",capsize = 3)
    plt.errorbar(Tinf_list[0:18], id_list[0:18], xerr=Tinf_confint_list[0:18], fmt="o", capsize = 3)
    plt.axvline(x = 30, linestyle = 'dashed', label = 'Optimal growth temperature')
    plt.legend()
    plt.xlabel('Temperature (°C)', fontsize = 13, fontweight = 'bold')
    plt.ylabel('Enzyme ID', fontsize = 13, fontweight = 'bold')
    plt.title('Topt and Tinf comparison to optimal\n' + r'growth temperature of B. subtilis', fontsize = 15, fontweight = 'bold')
    plt.show()
    
#plot_comparison()

    


