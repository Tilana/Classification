import pandas as pd
import pdb
import numpy as np
import random
import cPickle as pickle
import os
from listUtils import sortTupleList
import dataframeUtils as df
from Evaluation import Evaluation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors, linear_model
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2


classifierFunctions = {'DecisionTree': DecisionTreeClassifier(), 'MultinomialNB': MultinomialNB(), 'BernoulliNB': BernoulliNB(), 'RandomForest': RandomForestClassifier(), 'SVM': svm.SVC(probability=True), 'LogisticRegression': linear_model.LogisticRegression(), 'kNN':  neighbors.KNeighborsClassifier(n_neighbors=15)}

classifierParams = {'DecisionTree': {'min_samples_leaf': [2,5], 'max_depth':[3,5,7], 'criterion':['gini','entropy']}, 'SVM': {'kernel':['rbf'], 'gamma':['auto', 0.5, 0.9], 'C':[0.5, 1]}, 'LogisticRegression': {'penalty':['l1','l2'], 'C':[0.3,0.5,1,10]}, 'kNN': {'n_neighbors':[3,5]}, 'MultinomialNB': {'alpha':[0,0.01, 0.3, 0.6, 1], 'fit_prior':[True, False]}, 'BernoulliNB': {'alpha':[0, 0.01, 0.3, 0.6, 1], 'binarize':[True, False], 'fit_prior':[True, False]}, 'RandomForest': {'n_estimators':[5,10,30], 'min_samples_leaf': [2,5], 'max_depth':[None,3,5,7], 'criterion':['gini','entropy']}}


class ClassificationModel:

    def __init__(self, path=None, target=None, data=None, labelOfInterest=None):
        self.data = []
        if path:
            self.data = pd.read_pickle(path)
        if data is not None:
            self.data = data
        if labelOfInterest is not None:
            self.labelOfInterest = labelOfInterest
        self.targetFeature = target


    def splitDataset(self, num, random=True):
        self._generateHalfSplitIndices(num)
        if random:
            self._generateRandomIndices(num)
        self.split()


    def split(self):
        self.trainData = self.data.loc[self.trainIndices]
        self.trainTarget = self.target.loc[self.trainIndices]
        self.testData = self.data.loc[self.testIndices]
        self.testTarget = self.target.loc[self.testIndices]


    def validationSet(self):
        n = len(self.testData)/2
        self.holdout = self.testData[:n]
        self.holdoutTarget = self.testTarget[:n]
        self.testData = self.testData[n:]
        self.testIndices = self.testIndices[n:]
        self.testTarget = self.testTarget[n:]


    def _generateRandomIndices(self, num):
        split = StratifiedKFold(n_splits=2) #, train_size=num)
        X = self.data
        y = self.target
        split.get_n_splits(X,y)
        for trainIndex, testIndex in split.split(X,y):
            self.trainIndices = trainIndex
            self.testIndices = testIndex


    def _generateHalfSplitIndices(self, num):
        self.trainIndices = self.data.index[:num]
        self.testIndices = self.data.index[num:]


    def balanceDataset(self, factor=1):
        trueCases = df.getIndex(df.filterData(self.data, self.targetFeature))
        negativeCases = list(set(df.getIndex(self.data)) - set(trueCases))
        numberSamples = factor * len(trueCases)
        if numberSamples + len(trueCases) >= len(self.data):
            numberSamples = len(negativeCases)
        selectedNegativeCases = self.getRandomSample(negativeCases, numberSamples)
        self.data = self.data.loc[trueCases+selectedNegativeCases, :]
        self.data.reset_index(inplace=True)

    def getRandomSample(self, data, n):
        return random.sample(data, n)


    def cleanDataset(self):
        for field in self.data.columns[self.data.dtypes==object]:
            self.data = self.data.drop(field, axis=1)

    def object2CategoricalFeature(self, column):
        categoricalData = self.data[column].astype('category')
        self.targetCategories = categoricalData.cat.categories.tolist()
        if self.classificationType =='binary' and self.labelOfInterest:
            negCategory = [category for category in self.targetCategories if category != self.labelOfInterest][0]
            categoricalData.cat.set_categories([negCategory, column])
        self.data[column] = categoricalData.cat.codes


    def createTarget(self):
        self.setClassificationType(self.targetFeature)
        #if self.isObject(self.targetFeature):
        self.object2CategoricalFeature(self.targetFeature)
        self.target = self.data[self.targetFeature]
        self.dropNANRows(self.targetFeature)

    def setClassificationType(self, target):
        self.setTargetLabels(target)
        self.classificationType = 'binary'
        if len(self.targetLabels) > 2:
            self.classificationType = 'multi'

    def setTargetLabels(self, target):
        self.targetLabels = list(self.data[target].unique())

    def toNumeric(self, column):
        self.data[column] = self.data[column].astype(int)

    def toBoolean(self, column):
        self.data[column] = self.data[column].astype(bool)

    def isObject(self, column):
        return self.data[column].dtype == object

    def dropFeatures(self):
        if hasattr(self, 'keeplist'):
            keeplist = getattr(self, 'keeplist')
            self.droplist = list(set(self.data.columns.tolist()) - set(keeplist))
        self.data = self.data.drop(self.droplist, axis=1)


    def gridSearch(self, features, scoring='f1', scaling='False', pca='False', components=10):
        self.classifier= GridSearchCV(self.classifier, self.parameters, scoring=scoring)
        self.trainClassifier(features, scaling=scaling, pca=pca, components=components)
        bestScore = self.classifier.best_score_
        self.classifier = self.classifier.best_estimator_
        parameter = self.classifier.get_params()
        return (bestScore, parameter)

    def scaleData(self, data):
        scaler = MinMaxScaler()
        return scaler.fit_transform(data)


    def PCA(self, data, components):
        self.pca = PCA(components)
        self.pca.fit(data)
        return self.pca.transform(data)

    def FeatureSelection(self, X, y):
        self.FeatureSelector = SelectKBest(chi2, k=3000)
        return self.FeatureSelector.fit_transform(X,y)


    def trainClassifier(self, features, scaling=False, pca=False, components=10, selectFeatures=False):
        self.scaling = scaling
        self.pca = pca
        self.selectFeatures = selectFeatures
        target = self.trainTarget.tolist()
        if self.whitelist:
            self.increaseWeights(self.trainData, 'tfIdf', self.whitelist)
        trainData = self.getFeatureList(self.trainData,features)
        if scaling:
            trainData = self.scaleData(trainData)
        if pca:
            trainData = self.PCA(trainData, components)
        if self.selectFeatures:
            trainData = self.FeatureSelection(trainData, target)
        self.classifier.fit(trainData, target)

    def getFeatureList(self, data, features):
        featureList = df.combineColumnValues(data, features)
        return featureList


    def validate(self, features):
        features = self.getFeatureList(self.holdout, features)
        self.holdout['predictedLabel'] = self.classifier.predict(features)
        self.validation = Evaluation(self.holdoutTarget, self.holdout.predictedLabel.tolist())
        self.validation.accuracy()
        self.validation.recall()
        self.validation.precision()


    def predict(self, features):
        if self.whitelist:
            self.increaseWeights(self.testData, 'tfIdf', self.whitelist)
        testData = self.getFeatureList(self.testData,features)
        if self.scaling:
            testData = self.scaleData(testData)
        if self.pca:
            testData = self.pca.transform(testData)
        if self.selectFeatures:
            testData = self.FeatureSelector.transform(testData)
        self.testData['predictedLabel'] = self.classifier.predict(testData)
        if self.classifier.predict_proba:
            probabilities = self.classifier.predict_proba(testData)
            self.testData['probability'] = np.max(probabilities, axis=1)


    def evaluate(self, subset='test', avgType='macro'):
        self.setEvaluationAverage(avgType)
        if subset=='test':
            data = self.testData
            target = self.testTarget
            indices = self.testIndices
        elif subset=='validation':
            data = self.validationData
            target = self.validationTarget
            indices = self.validationIndices

        self.evaluation = Evaluation(target, data.predictedLabel.tolist(), self.evaluationAverage)
        self.evaluation.setAllTags()
        data = self.tagData(data, indices)
        if subset == 'test':
            self.testData = data
        elif subset == 'validation':
            self.validationData = data

        self.evaluation.accuracy()
        self.evaluation.recall()
        self.evaluation.precision()

    def setEvaluationAverage(self, avgType='macro'):
        self.evaluationAverage = avgType
        if self.classificationType=='binary':
            self.evaluationAverage = 'binary'

    def increaseWeights(self, data, col, whitelist):
        wordIndices = [self.getWordIndex(word) for word in whitelist if self.getWordIndex(word)!=None]
        weights = df.combineColumnValues(data, col)
        addValue = self.proportionalValue(weights)
        #addValue = np.ones(len(addValue))*0.8
        weightsDF = pd.DataFrame(weights)
        for index in wordIndices:
            #weightsDF[index] = weightsDF[index].add(addValue)
            weightsDF[index] = weightsDF[index].multiply(2)
        data[col] = df.combineColumnValues(weightsDF, range(0,weightsDF.shape[1]))


    def proportionalValue(self, weights):
        maximum = np.max(weights, axis=1)
        return  maximum/2

    def getWordIndex(self, word):
        if word in self.vocabulary:
            return self.vocabulary.index(word)

    def relevantFeatures(self):
        featureRelevance = self.classifier.feature_importances_
        indicesRelFeatures = np.where(featureRelevance>0)[0]
        relFeatures = [(self.vocabulary[ind], featureRelevance[ind]) for ind in indicesRelFeatures]
        self.featureImportance = sortTupleList(relFeatures)


    def dropNANRows(self, columns=None):
        if columns:
            self.data = self.data[pd.notnull(self.data[columns])]
        else:
            self.data = self.data.dropna()

    def mergeDataset(self, dataset2):
        self.data = pd.merge(self.data, dataset2, on=['id'])


    def addTag(self, tag, data, dataIndices):
        indices = getattr(self.evaluation, tag)
        tagIndices = [dataIndices[position] for position in indices]
        data.loc[tagIndices,'tag'] = tag
        return data

    def tagData(self, data, indices):
        tags = ['T','F']
        if self.classificationType=='binary':
            tags = ['TP', 'FP', 'TN', 'FN']
        for tag in tags:
            self.addTag(tag, data, indices)
        return data

    def oneHotEncoding(self, data):
        return pd.get_dummies(data)

    def buildClassifier(self, classifierType, params=None):
        self.classifierType = classifierType
        self.classifier = classifierFunctions[classifierType]
        if params:
            self.buildParamClassifier(params)
        self.parameters = classifierParams[classifierType]

    def buildParamClassifier(self, params):
        for param in params:
            setattr(self.classifier, param, params[param])


    def getSelectedTopics(self, topicNr, selectedTopics=None):
        self.topicList = self.getTopicList(topicNr)
        if selectedTopics != None:
            self.selectedTopics = [('Topic%d' % topic) for topic in selectedTopics]
            self.addUnselectedTopicsToDroplist()

    def getTopicList(self, topicNr):
        return [('Topic%d' % topic) for topic in range(0, topicNr)]

    def getSimilarDocs(self, nrDocs=5):
        return [('similarDocs%d' % docNr) for docNr in range(1, nrDocs+1)]

    def getRelevantWords(self, nrWords=3):
        return [('relevantWord%d' % docNr) for docNr in range(1, nrWords+1)]

    def weightFScore(self, beta):
        return make_scorer(fbeta_score, beta=beta, average='macro')


    def preprocessTestData(self, preprocessor, vecType='tfIdf'):
        testDocs = self.testData.text.tolist()
        self.testData[vecType] = preprocessor.vectorizeDocs(testDocs)

    def createResultFolder(self, dataset, id_name, classifierType):
        pass






