import pandas as pd
import numpy as np
import random
import cPickle as pickle
import os
import dataframeUtils as df
from Evaluation import Evaluation
from Preprocessor import Preprocessor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.cross_validation import KFold

class ClassificationModel:

    def __init__(self, path=None, target=None):
        self.data = []
        if path != None:
                self.data = pd.read_pickle(path)
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


    def _generateRandomIndices(self, num):
        self.trainIndices = random.sample(self.data.index, num)
        self.testIndices = list(set(self.data.index) - set(self.trainIndices))

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

    def getRandomSample(self, data, n):
        return random.sample(data, n)

    
    def cleanDataset(self):
        for field in self.data.columns[self.data.dtypes==object]:
            self.data = self.data.drop(field, axis=1)
    
    def createNumericFeature(self, column):
        category = 0
        for value in self.data[column].unique():
            rowIndex = self.data[self.data[column]==value].index.tolist()
            self.data.loc[rowIndex, column] = category
            category += 1
        self.toNumeric(column)

    def createTarget(self):
        self.target = self.data[self.targetFeature]

    def toNumeric(self, column):
        self.data[column] = self.data[column].astype(int)

    def toBoolean(self, column):
        self.data[column] = self.data[column].astype(bool)

    def dropFeatures(self):
        if hasattr(self, 'keeplist'):
            keeplist = getattr(self, 'keeplist')
            self.droplist = list(set(self.data.columns.tolist()) - set(keeplist))
        self.data = self.data.drop(self.droplist, axis=1)

    def trainClassifier(self, features):
        trainData = self.trainData[features].tolist()
        target = self.trainTarget.tolist()
        self.classifier.fit(trainData, target)

    
    def predict(self, features):
        self.testData['predictedLabel'] = self.classifier.predict(self.testData[features].tolist())
        if self.classifier.predict_proba:
            probabilities = self.classifier.predict_proba(self.testData[features].tolist())
            self.testData['probability'] = np.max(probabilities, axis=1)


    def evaluate(self):
        self.evaluation = Evaluation(self.testTarget, self.testData.predictedLabel.tolist())
        self.evaluation.setAllTags()
        self.tagTestData()

        self.evaluation.accuracy()
        self.evaluation.recall()
        self.evaluation.precision()

    def computeFeatureImportance(self):
        featureImportance = sorted(zip(map(lambda relevance: round(relevance,4), self.classifier.feature_importances_), self.data.columns), reverse=True)
        self.featureImportance = [(elem[1], elem[0]) for elem in featureImportance if elem[0]>0.0]


    def dropNANRows(self):
        self.data = self.data.dropna()

    def mergeDataset(self, dataset2):
        self.data = pd.merge(self.data, dataset2, on=['id'])


    def addTag(self, tag):
        indices = getattr(self.evaluation, tag)
        tagIndices = [self.testIndices[position] for position in indices]
        self.testData.loc[tagIndices,'tag'] = tag

    def tagTestData(self):
        tags = ['TP', 'FP', 'TN', 'FN']
        for tag in tags:
            self.addTag(tag)

    def oneHotEncoding(self, data):
        return pd.get_dummies(data)

    def buildClassifier(self, classifierType, alpha=0.6):
        self.classifierType = classifierType
        if classifierType == 'DecisionTree':
            self.classifier = DecisionTreeClassifier()
        elif classifierType == 'MultinomialNB':
            self.classifier = MultinomialNB(alpha=alpha)
        elif classifierType == 'BernoulliNB':
            self.classifier = BernoulliNB(alpha=alpha)
        elif classifierType == 'RandomForest':
            self.classifier = RandomForestClassifier()
        elif classifierType == 'SVM':
            self.classifier = svm.SVC(probability=True)
        elif classifierType == 'LogisticRegression':
            self.classifier = linear_model.LogisticRegression()

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


    def buildPreprocessor(self, vecType='tfIdf', min_df=10, max_df=0.5, stop_words='english', ngram_range = (1,2), max_features=8000):
        self.preprocessor = Preprocessor(processor=vecType, min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=ngram_range, max_features=max_features) 


    def trainPreprocessor(self, vecType='tfIdf'):
        trainDocs = self.data.text.tolist()
        self.data[vecType] = self.preprocessor.trainVectorizer(trainDocs)


    def preprocessTestData(self, vecType='tfIdf'):
        testDocs = self.testData.text.tolist()
        self.testData[vecType] = self.preprocessor.vectorizeDocs(testDocs)

    def existsProcessedData(self, path):
        return os.path.exists(path + '.pkl')


    def save(self, path):
        self.savePreprocessor(path)
        with open(path +'.pkl', 'wb') as f:
            pickle.dump(self, f, -1)

    
    def load(self, path):
        model = pickle.load(open(path+'.pkl', 'rb'))
        model.loadPreprocessor(path)
        return model

    def savePreprocessor(self, path):
        if hasattr(self, 'preprocessor'):
            self.preprocessor.save(path)
            del self.preprocessor


    def loadPreprocessor(self, path):
        preprocessor = Preprocessor()
        self.preprocessor = preprocessor.load(path)

        
