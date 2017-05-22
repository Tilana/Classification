from __future__ import division
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
import pandas as pd
import numpy as np

class Evaluation:

    def __init__(self, target, prediction, average='binary'):
        self.target= target
        self.prediction= prediction
        self.checkLength()
        self.n = len(self.target)
        self.average = average

    def accuracy(self):
        self.accuracy = accuracy_score(self.target, self.prediction)
                                                                    
    def recall(self):
        self.recall = recall_score(self.target, self.prediction, average=self.average)
                                                                    
    def precision(self):
        self.precision = precision_score(self.target, self.prediction, average=self.average)


    def confusionMatrix(self, labels=None):
        matrix = confusion_matrix(self.target, self.prediction)
        self.confusionMatrix = pd.DataFrame(matrix)
        if labels:
            self.confusionMatrix.columns = labels
            self.confusionMatrix.index = labels
        self.normalizeMatrix()

    def normalizeMatrix(self):
        self.normConfusionMatrix = self.confusionMatrix.div(self.confusionMatrix.sum(axis=1), axis=0)

    def classificationReport(self, labels=None):
        self.report = classification_report(self.target.tolist(), self.prediction, target_names=labels)


    def checkLength(self):
        if len(self.target) != len(self.prediction):
            print 'WARNING: Evaluation - length of target and prediction list is unequal'


    def createTags(self):
        self.tags = []
        if self.average =='binary':
            self.binaryTags()
        else:
            self.multiTags()


    def multiTags(self):
        for predValue, targetValue in zip(self.prediction, self.target):
           if predValue == targetValue:
               self.tags.append('T')
           else:
               self.tags.append('F')


    def binaryTags(self):
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
        categories = ['T','F','TP', 'FP', 'TN', 'FN']
        for tag in categories:
            self.setTag(tag)
            self.setTagLength(tag)
    
    def toSeries(self, fields=['accuracy', 'precision', 'recall']):
       return pd.Series([self.__dict__[field] for field in fields])
