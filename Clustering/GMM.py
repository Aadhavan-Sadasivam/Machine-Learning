# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 13:39:48 2018

@author: aadha
"""

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from pca import PCA



def e_step(row, center, prior, cov, k):
    
    numerator = []
    denom = 0
    for i in range(k):
        num = multivariate_normal.pdf(row, mean = center[i], cov = cov) * prior[i]
        denom += num
        numerator.append(num)
    result = []
    
    for num in numerator:
        result.append(num/denom)
        
    return result

def mvn(row, center, cov, prior):
    return multivariate_normal.pdf(row, mean = center, cov = cov) * prior

def gmm(data, k):
    
    k = 2
    max_itr = 100
    tolerance = 1e-3
    
    
    center = data.sample(n=k, random_state = 10).values
    
    covariance = data.cov().values
    
    prior = [0.5, 0.5]
   
    like = []
    
    for itr in range(100):
        
        #Estep
        E = data.apply(e_step, axis = 1, args = (center,prior,covariance,k ))
        E = E.values.tolist()
        EFrame = pd.DataFrame(E)
        
        #mstep
        for j in range(k):
            val = []
            #
            #calculate sum pf all instances * expectation for computing center
            for i in range(len(data)):
                val.append(data.iloc[i,:] * EFrame.iloc[i,j])
                
            numerator = val[0]
            #calculate sum of expectation for computing prior
            for i in range(1, len(val)):
                numerator += val[i]
                denominator = sum(EFrame.iloc[:,j])
                
            #
            # update center and prior
            center[j] = numerator/denominator
            prior[j] = denominator/data.shape[1]
            
        
        likelihood = 0
        for j in range(k):
            temp = data.apply(mvn, axis = 1, args=(center[j], covariance, prior[j]))
            likelihood += np.log(sum(temp))
            
        
        like.append(likelihood)
        
        
        if len(like) > 1 and abs(like[-2] - like[-1]) < tolerance:
            print("converged at iteration {}..".format(itr))
            break
        if itr == max_itr-1:
            print("Maximum number of iterations reached, algorithm failed to converge..")
    
    """ 
    x = np.arange(len(like))
    plt.plot(x,like, color = 'red')
    plt.xlabel('Iterations')
    plt.ylabel('Log likelihood')
    plt.title('Iterations vs Likelihood')
    plt.show()
    """
    
    FinalCluster = (EFrame > 0.5) * 1
   
    plt.cla()
    plt.scatter(data.iloc[:, 0].values, data.iloc[:, 1].values, c = FinalCluster[0].values, s=40, cmap='viridis');
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('GMM')
    plt.show()    

if __name__ == "__main__":
    
    data = pd.read_csv('audioData.csv', header = None)
    #Without PCA
    gmm(data,2)
    #with PCA
    PCA_data = PCA(data, 2)
    gmm(PCA_data, 2)