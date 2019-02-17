from __future__ import division
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report, multilabel_confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np


class Evaluation:

    def __init__(self, target, prediction, average='binary', classes=None):
        #import pdb
        #pdb.set_trace()
        self.target = target
        self.prediction = prediction
        self.checkLength()
        self.n = len(self.target)
        self.average = average
        if isinstance(self.target[0], list):

            print('MULTI-LABEL CLASSIFICATION')
            binarizer = MultiLabelBinarizer(classes)
            self.org_target = target
            self.org_prediction = prediction
            self.target = binarizer.fit_transform(target)
            self.prediction = binarizer.fit_transform(prediction)
            if average == 'binary':
                self.average = 'micro'
        else:
            if average == 'binary' and len(set(target)) > 2:
                self.average = 'micro'
            #binarizer = MultiLabelBinarizer()
            #self.target = binarizer.fit_transform(self.target)
            #self.prediction = binarizer.fit_transform(self.prediction)

    def computeMeasures(self):
        self.accuracy()
        self.recall()
        self.precision()

    def accuracy(self):
        self.accuracy = accuracy_score(self.target, self.prediction)

    def recall(self):
        self.recall = recall_score(self.target, self.prediction, average=self.average)

    def precision(self):
        self.precision = precision_score(self.target, self.prediction, average=self.average)

    def multilabelConfusionMatrix(self):
        return multilabel_confusion_matrix(self.target, self.prediction)

    def confusionMatrix(self, labels=None):
        matrix = confusion_matrix(self.target, self.prediction)
        self.confusionMatrix = pd.DataFrame(matrix)
        if self.average != 'binary':
            diagonal = np.diagonal(matrix)
            row_sum = np.sum(matrix, axis=1)
            col_sum = np.sum(matrix, axis=0)
            row_false = row_sum - diagonal
            col_false = col_sum - diagonal
            cols = pd.DataFrame([row_false.tolist(), row_sum.tolist(), diagonal.tolist()], index=['false', 'total', 'correct']).transpose()
            rows = pd.DataFrame([col_sum.tolist(), diagonal.tolist(), col_false.tolist()], index=['total', 'correct', 'false'])
            self.confusionMatrix = pd.concat([self.confusionMatrix, cols], axis=1)
            self.confusionMatrix = pd.concat([self.confusionMatrix, rows], axis=0, sort=False)
        if labels:
            mapping = dict(zip(range(len(labels)), labels))
            self.confusionMatrix.rename(columns=mapping, index=mapping, inplace=True)
        self.normalizeMatrix()

    def normalizeMatrix(self):
        self.normConfusionMatrix = self.confusionMatrix.div(self.confusionMatrix.sum(axis=1), axis=0)

    def classificationReport(self, labels=None):
        self.report = classification_report(self.target, self.prediction, target_names=labels)

    def checkLength(self):
        if len(self.target) != len(self.prediction):
            print('WARNING: Evaluation - length of target and prediction list is unequal')

    def createTags(self):
        self.tags = []
        if self.average == 'binary':
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
            if predValue:
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
        indices = [ind for ind, value in enumerate(self.tags) if value == tag]
        setattr(self, tag, indices)

    def setTagLength(self, tag):
        setattr(self, 'n_'+tag, len(getattr(self, tag)))

    def setAllTags(self):
        self.createTags()
        categories = ['T', 'F', 'TP', 'FP', 'TN', 'FN']
        for tag in categories:
            self.setTag(tag)
            self.setTagLength(tag)

    def toSeries(self, fields=['accuracy', 'precision', 'recall']):
        return pd.Series([self.__dict__[field] for field in fields])

    def setEvaluation(self, target, prediction):
        target_set = set(target)
        prediction_set = set(prediction)
        false = list(prediction_set - target_set)
        missed = list(target_set - prediction_set)
        correct = list(target_set & prediction_set)
        return (correct, missed, false)

    def evalMultiLabel(self):
        res = [self.setEvaluation(self.org_target[ind], self.org_prediction[ind]) for ind, elem in enumerate(self.target)]
        return pd.DataFrame([list(elem) for elem in zip(*res)], index=['TP', 'FN', 'FP']).transpose()

