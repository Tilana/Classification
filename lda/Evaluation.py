from __future__ import division
from sklearn import metrics
import pandas as pd
import numpy as np

class Evaluation:

    def __init__(self, target, prediction):
        self.target= target
        self.prediction= prediction
        self.checkLength()
        self.n = len(self.target)

    def accuracy(self):
        self.accuracy = np.float64(self.n_TP + self.n_TN)/self.n
                                                                    
    def recall(self):
        self.recall = np.float64(self.n_TP)/(self.n_TP + self.n_FN)
                                                                    
    def precision(self):
        self.precision = np.float64(self.n_TP)/(self.n_TP + self.n_FP)

    def confusionMatrix(self):
        matrix = np.array([[self.n_TN, self.n_FP], [self.n_FN, self.n_TP]], dtype=int)
        self.confusionMatrix = pd.DataFrame(matrix)
        self.confusionMatrix = self.confusionMatrix.rename(index={0:'Target False', 1:'Target True'}, columns={0:'Predicted False', 1:'Predicted True'})

    def checkLength(self):
        if len(self.target) != len(self.prediction):
            print 'WARNING: Evaluation - length of target and prediction list is unequal'

    def createTags(self):
        self.tags = []
        for predValue, targetValue in zip(self.prediction, self.target):
            if predValue == True:
                if predValue == targetValue:
                    self.tags.append('TP')
                else:
                    self.tags.append('FP')
            else:
                if predValue == targetValue:
                    self.tags.append('TN')
                else:
                    self.tags.append('FN')

    def setTag(self, tag):
        indices = [ind for ind,value in enumerate(self.tags) if value==tag]
        setattr(self, tag, indices)

    def setTagLength(self, tag):
        setattr(self, 'n_'+tag, len(getattr(self, tag)))


    def setAllTags(self):
        self.createTags()
        categories = ['TP', 'FP', 'TN', 'FN']
        for tag in categories:
            self.setTag(tag)
            self.setTagLength(tag)

    
