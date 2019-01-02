# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 00:25:24 2018

@author: aadha
"""

import pandas as pd
import numpy as np

def PCA(data, n_components):
    
    
    covariance = data.cov().values
    values, vectors = np.linalg.eig(covariance)
    
    TopComponents = vectors[:,:n_components]
    
    new_data = np.matmul(data.values, TopComponents)
    newFrame = pd.DataFrame(new_data)
    
    return newFrame


data = pd.DataFrame([[1,1,2],[-1,-1,-2],[3,3,6]])
covariance = data.cov().values
values, vectors = np.linalg.eig(covariance)

data = np.matmul(data.values,vectors[:,0])