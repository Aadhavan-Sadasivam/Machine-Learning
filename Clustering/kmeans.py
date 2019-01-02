# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 16:22:03 2018

@author: Aadhavan Sadasivam
"""

import pandas as pd
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
from pca import PCA


def em(data, center,k):
    
    updatedCenter = pd.DataFrame([[0]*center.shape[1]]*k)
    
    #calculate Eucledian distance
    dist = distance.cdist(data.iloc[:,:-1].values, center.values, metric='euclidean')
    
    #cluster assignment
    clusterAssignment = []
    for item in dist:   
        clusterAssignment.append(np.argmin(item))
    data['cluster'] = clusterAssignment
    cluster = []
    for i in range(k):
        cluster.append(data.loc[data['cluster'] == i])
        
    #update center
    for i in range(k):
        updatedCenter.iloc[i,:] = cluster[i].iloc[:,:-1].mean(axis = 0)
    return updatedCenter

def calculate_loss(data, center, k):
    dist = distance.cdist(data.iloc[:,:-1].values, center.values, metric='euclidean')
    
    loss = 0
    for item in dist:   
        loss += (item[np.argmin(item)])**2
    return loss

def kmeans(data, k):
    
    data['cluster'] = 1
    center = data.sample(n=k, random_state = 10)
    center = center.iloc[:,:-1]
    
    i = 0
    while(True):
        newCenter = em(data, center, k)
        if newCenter.equals(center):
            break
        else:
            center = newCenter.iloc[:,:]
            #print(k,str(i+1))
        i+=1
            
    return calculate_loss(data, center, k)
        
        
        
if __name__ == "__main__":
    
    data = pd.read_csv('audioData.csv', header = None)
    loss = []
    for k in range(2,11): 
        loss.append(kmeans(data,k))

    x = np.arange(len(loss))
    xticks = np.arange(2,11)
    plt.plot(x, loss, color = 'red')
    plt.xlabel('K')
    plt.ylabel('Loss')
    plt.title('Kmeans - K Vs Loss')
    plt.xticks(x,xticks)
    plt.show()
    
    #
    #With PCA
   
    PCA_data = PCA(data, 2)
    loss = []
    for k in range(2,11): 
        loss.append(kmeans(PCA_data,k))

    plt.cla()
    x = np.arange(len(loss))
    xticks = np.arange(2,11)
    plt.plot(x, loss, color = 'red')
    plt.xlabel('K')
    plt.ylabel('Loss')
    plt.title('Kmeans - K Vs Loss (PCA)')
    plt.xticks(x,xticks)
    plt.show()