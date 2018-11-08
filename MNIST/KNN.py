# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 12:38:28 2018

@author: Aadhavan Sadasivam
"""
from load_dataset import read
import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def loadData(data): 
    #
    #Read data
    Label, Img1 = read(data)
    #
    #normalise data
    Img1 = Img1/255   
    #
    #flattendata
    Img = np.asarray([img.flatten() for img in Img1])
    
    #create a frame
    Frame = pd.DataFrame(Img)
    Frame['label'] = Label   
    
    return Frame


def getDecision(Kneighbors):
    count = dict()
    for item in Kneighbors:
        if item in count.keys():
            count[item] += 1
        else:
            count[item] = 1
    maxlabel = max(count.values())
    return (list(count.keys())[list(count.values()).index(maxlabel)])

def accuracy(cm, total):
    correct = 0
    for i in range(len(cm)):
        correct += cm[i][i]
    return correct/total
     
if __name__ == "__main__":
    
    #
    #normalise and load data into a dataframe
    TrainFrame = loadData("training")
    TestFrame = loadData("testing")
    
    #Test Images
    #Train Images
    TrainImg = TrainFrame.iloc[:,:-1].values
    TestImg = TestFrame.iloc[:,:-1].values
    
    #
    #TestLabel
    TestLabel =  TestFrame.iloc[:,-1].values
    kcount = [1,3,5,10,30,50,70,80,90,100]
    #
    #computer eucledian distance
    dist = distance.cdist(TestImg[:100], TrainImg, metric='euclidean')
    
    #load eucledian distance into a dataframe
    DistanceFrame = pd.DataFrame(dist)
    #final label of a test image for all K values
    klabel = []
    for i in range(len(DistanceFrame)):
        #
        #list to store the final label of a test image
        templabel = []
        #
        #sort the eucledian distance of a Test Image
        euclid = DistanceFrame.iloc[i,:].sort_values()
        #
        #Get first 100 neighbors, since we are considering a maximum of 100 neighbors
        neighbors100 = euclid[:100].index.tolist()
        #
        #Get the label of those 100 neighbors
        labels = TrainFrame.iloc[neighbors100,-1].values.tolist()
        #
        #Get the label with maximum count among the K neighbors
        #Arbitrary decision in case of tie
        for k in kcount:
            Kneighbors = labels[:k]
            templabel.append(getDecision(Kneighbors))
        klabel.append(templabel)
    
    ResultFrame = pd.DataFrame(klabel, columns = kcount)
    #list to store Accuracy
    Accuracy = []
    for i in range(len(kcount)):
        label = ResultFrame.iloc[:,i].values.tolist()
        cm = confusion_matrix(TestLabel[:100], label)
        cmFrame = pd.DataFrame(cm)
        cmFrame.to_csv('KNNaccuracy/K_'+str(kcount[i])+'.csv',index=None, header=False)
        Accuracy.append(accuracy(cm, len(TestImg)))
    
    x = np.arange(len(kcount))
    plt.plot(x, Accuracy, color='red')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.title('KNN - Number of Neighbors VS Accuracy')
    plt.xticks(x, kcount)
    plt.savefig('KNN_accuracy.jpg')