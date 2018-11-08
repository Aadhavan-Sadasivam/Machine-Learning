# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 03:23:20 2018

@author: Aadhavan Sadasivam
"""

import os
import string
import pandas as pd
import numpy as np
from collections import Counter
from operator import itemgetter 
from sklearn.metrics import confusion_matrix

def get_files(directory):
    
    files = []

    #
    #Read files from negative review folder
    for file in os.listdir(directory+"/neg"):
        name = file
        """
        filename = directory+"/neg/"+file
        file = open(filename,'r', encoding='utf-8')
        content = file.read()
        content = content.translate(str.maketrans('','',string.punctuation))
        """
        files.append([name, 0])
        
    #
    #Read files from negative review folder
    for file in os.listdir(directory+"/pos"):
        name = file
        """
        filename = directory+"/pos/"+file
        file = open(filename,'r', encoding='utf-8')
        content = file.read()
        content = content.translate(str.maketrans('','',string.punctuation))
        """
        files.append([name, 1])
    
    Frame = pd.DataFrame(files,columns=['FileName','label'])
    
    return Frame

def getVocabulary():
    return [line.rstrip('\n') for line in open('vocabulary.txt', 'r', encoding='utf-8')]
        
def sampleData(frame, TrainFraction):
    
    PosFrame = frame.loc[FilesFrame['label']==1] 
    NegFrame = frame.loc[FilesFrame['label']==0] 
    
    #
    #sample positive and negative files in the same propotion as actual dataset
    PosTrainFraction = TrainFraction * (len(PosFrame)/len(frame))
    NegTrainFraction = TrainFraction * (len(NegFrame)/len(frame))
    
    PosTrainFrame = PosFrame.sample(n = int(PosTrainFraction * len(FilesFrame)))
    NegTrainFrame = NegFrame.sample(n = int(NegTrainFraction * len(FilesFrame)))
    
    PosTestFrame = PosFrame.drop(PosTrainFrame.index.tolist()).reset_index(drop = True)
    NegTestFrame = NegFrame.drop(NegTrainFrame.index.tolist()).reset_index(drop = True)
    
    PosTrainFrame = PosTrainFrame.reset_index(drop = True)
    NegTrainFrame = NegTrainFrame.reset_index(drop = True)
    
    return PosTrainFrame, NegTrainFrame, PosTestFrame, NegTestFrame



if __name__ =="__main__":

    #get files
    FilesFrame = get_files("MovieData")
    #load vocabulary obtained from preprocessing
    vocabulary = getVocabulary()
    
    #
    #get stop words
    #I have preprocessed the files and stored all stop words in a text file to save computation
    FileContent = [line.rstrip('\n') for line in open('stop_words.txt', 'r', encoding='utf-8')]
    
    stop_words = []
    file_list = []
    
    for item in FileContent:
        split_words = item.split(",")
        stop_words.append(split_words[1:])
        file_list.append(split_words[0])
        
    for i in range(3): 
        #split train and test data
        TrainFraction = 0.8
        PosTrainFrame, NegTrainFrame, PosTestFrame, NegTestFrame = sampleData(FilesFrame, TrainFraction)
        
        TrainFrame = pd.concat([PosTrainFrame, NegTrainFrame], axis = 0)
        TestFrame = pd.concat([PosTestFrame, NegTestFrame],axis = 0)
        
        Probability_label = [len(PosTrainFrame) / len(TrainFrame), len(NegTrainFrame) / len(TrainFrame)]
        
        
        #Get pos files and words and count
        PosTrainFiles = PosTrainFrame['FileName'].values.tolist()
        FileListFrame = pd.DataFrame(file_list,columns=['FileName'])
        IndexFrame = FileListFrame.loc[FileListFrame['FileName'].isin(PosTrainFiles)]
        PosTrainIndex = IndexFrame.index.tolist()
        #Has stop words from all positive training samples
        PosTrainWords = itemgetter(*PosTrainIndex)(stop_words)
                
        #count words in each positive files
        PosTrainWordCount = []
        for item in PosTrainWords:
            PosTrainWordCount.append(Counter(item))
                    
                    
        #Get neg files and words and count
        NegTrainFiles = NegTrainFrame['FileName'].values.tolist()
        IndexFrame = FileListFrame.loc[FileListFrame['FileName'].isin(NegTrainFiles)]
        NegTrainIndex = IndexFrame.index.tolist()
             
        #Has stop words from all negative training samples
        NegTrainWords = itemgetter(*NegTrainIndex)(stop_words)  
                            
        #count words in all positive files
        NegTrainWordCount = []
        for item in NegTrainWords:
            NegTrainWordCount.append(Counter(item))
                                
        NegativeVocabulary = [y for x in NegTrainWords for y in x]
        PositiveVocabulary = [y for x in PosTrainWords for y in x]
                                
        TotalVocabulary = NegativeVocabulary + PositiveVocabulary

        TrainVocabulary = Counter(TotalVocabulary)
                                
        #count occourence of words from files 
        word_count = []
        for word in TrainVocabulary.keys():
            PosCount = 0
            NegCount = 0
            for file in PosTrainWordCount:
                if(word in file.keys()):
                    PosCount += 1
            for file in NegTrainWordCount:
                if(word in file.keys()):
                    NegCount += 1
            word_count.append([word,PosCount,NegCount])
                                                    
                                                    
        SeenLikelihood = []
        #Calcualte the likehood for seen words
        for item in word_count:
            yes = (item[1] +1)/ ( len(PosTrainFiles) + len(vocabulary) ) 
            no = (item[2] + 1) / ( len(NegTrainFiles) + len(vocabulary) ) 
            SeenLikelihood.append([item[0], yes, no])
            
        SeenLikelihoodFrame = pd.DataFrame(SeenLikelihood, columns = ['word', 'yes', 'no'])        
        #print(sum(SeenLikelihoodFrame['yes']))
    
        Unseen = vocabulary - TrainVocabulary.keys()
        UnseenLikehood = []
        for word in Unseen:
            yes = (1)/ ( len(PosTrainFiles) + len(vocabulary) ) 
            no = (1) / ( len(NegTrainFiles) + len(vocabulary) ) 
            UnseenLikehood.append([word, yes, no])
            
        UnseenLikelihoodFrame = pd.DataFrame(UnseenLikehood, columns = ['word', 'yes', 'no'])        
    
    
        LikelihoodFrame = pd.concat([SeenLikelihoodFrame,UnseenLikelihoodFrame], axis = 0)
        
        
        #classifying
        Test_files = TestFrame['FileName'].values.tolist()
        TestFilesFrame = pd.DataFrame(Test_files,columns=['FileName'])
        IndexFrame = TestFilesFrame.loc[TestFilesFrame['FileName'].isin(Test_files)]
        TestIndex = IndexFrame.index.tolist() 
        #Has stop words from all test samples
        TestWords = itemgetter(*TestIndex)(stop_words)
        cmm = []
        PredLabel = []
        file = TestWords[0]
        for file in TestWords:
            P_Y = 1
            P_N = 1
            row = LikelihoodFrame.loc[LikelihoodFrame['word'].isin(file)]
            P_Y = row['yes'].product()
            P_N = row['no'].product()
            if(P_Y > P_N):
                PredLabel.append(1)
            else:
                PredLabel.append(0)
        TestLabel = TestFrame['label'].values.tolist()
        
        cm = confusion_matrix(TestLabel, PredLabel)
        print(cm)
        cmm.append(cm)
        