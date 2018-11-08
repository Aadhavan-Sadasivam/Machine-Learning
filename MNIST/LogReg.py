# -*- coding: utf-8 -*-
"""
Created on Tue Sep 25 01:49:29 2018

@author: Aadhavan Sadasivam
"""

from load_dataset import read
import numpy as np
import pandas as pd
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

def numer(data, w):
    weight = w[:-1]
    bias = w[-1]
    multiplication = np.matmul(data, weight) + bias
    return np.exp(multiplication)

def denom(data,w):
    summation = pd.Series([1]*len(data))
    for weight in w:
        w1 = weight[:-1]
        bias = weight[-1]
        multiplication = np.matmul(data, w1) + bias
        exponent = pd.Series(np.exp(multiplication))
        summation  = summation + exponent
    return summation

def test(TestImg, w):
    Tpredict = []
    for i in range(len(TestImg)):
        denominator = 1
        probability = []
        numerator = []
        for weight in w:
            #calculate the numerator for each label
            term = np.exp(np.dot(weight[:-1], TestImg[i])+weight[-1])
            numerator.append(term)
            denominator += term
        numerator.append(1)
        probability = pd.Series(numerator)
        #
        #numerator/denominator
        probability = probability.div(denominator)
        #
        #returns the label with maximum probability 
        Tpredict.append(probability.values.argmax())
    return Tpredict

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
    
    #get Training features
    TrainImg = TrainFrame.iloc[:,:-1].values
    #get training label
    TrainLabel = TrainFrame.iloc[:,-1]
    
    #get testing features
    TestImg = TestFrame.iloc[:,:-1].values
    #get testing data
    TestLabel = TestFrame.iloc[:,-1].values.tolist()
    
    #
    #constants and features
    LearningRate = 0.015
    maxIter = 100
    #
    #dimension of a weight vector = len(features)+1
    numFeatures = TestImg.shape[1]+1 
    
    #
    #initialise the weight vectors to zero
    w = np.zeros(9,numFeatures)
    
    #
    #To store accuracy for each iteration
    Accuracy = []
    
    #
    #Training
    for iteration in range(maxIter):
        #
        #I reduce the learning rate for each iteration and restart it after 20 iterations
        LearningRate = 0.015 - ((iteration % 20) * 0.00014985)
        #For each weight vector
        for j in range(9):
            #
            #P(Y = j) 
            Label = (TrainLabel == j) * 1
            
            #
            #train bias
            
            #calculate p(Y = j | X,W)
            denominator = denom(TrainImg.iloc[:,:-1].values,w)
            numerator = numer(TrainImg, w[j])
            term1 = numerator/denominator
            
            #calculate P(Y = j) - p(Y = j | X,W)
            term2 = Label - term1
            
            #wj0 = wj0 + (learningrate * sum((P(Y = j) - p(Y = j | X,W))))
            #w0 is the last element in weight vector
            w[j][-1] += LearningRate * sum(term2)
            
            #
            #train each feature
            for i in range(numFeatures-1):
                #calculate p(Y = j | X,W)
                denominator = denom(TrainImg.iloc[:,:-1].values,w)
                numerator = numer(TrainImg, w[j])
                term1 = numerator/denominator
                
                #calculate P(Y = j) - p(Y = j | X,W)
                term2 = Label - term1
                
                #calculate (P(Y = j) - p(Y = j | X,W))*xi
                term3 = term2 * TrainImg.iloc[:,i]
                
                #wji = wji + (learningrate * sum((P(Y = j) - p(Y = j | X,W))*xi))
                w[j][i] += LearningRate * sum(term3)
                
            print(iteration,j)
        
        Tpredict  = test(TestImg, w)   
        cm = confusion_matrix(TestLabel, Tpredict)
        Accuracy.append(accuracy(cm, len(Tpredict)))
        cmFrame = pd.DataFrame(cm)
        cmFrame.to_csv('LogRegAccuracy/iteration'+str(iteration)+'.csv',index=False, header=False)
        print(cm)
        
        print("Accuracy", accuracy(cm, len(Tpredict)), "Learning Rate", LearningRate)

    x = np.arange(len(Accuracy))
    plt.plot(x, Accuracy, color='red')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.title('Logistic Regeression - Number of Iteration VS Accuracy')
    plt.savefig('LogReg_accuracy.jpg')        