import pandas as pd
import numpy as np
import random
import cPickle as pickle
import os
from listUtils import sortTupleList
import dataframeUtils as df
from Evaluation import Evaluation
from Preprocessor import Preprocessor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, neighbors, linear_model
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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


    def validationSet(self):
        n = len(self.testData)/2
        self.holdout = self.testData[:n]
        self.holdoutTarget = self.testTarget[:n]
        self.testData = self.testData[n:]
        self.assertEqual(self.evaluation.tags, result)
        self.testIndices = self.testIndices[n:]
        self.testTarget = self.testTarget[n:]


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
        print numberSamples
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
        category = 0
        for value in self.data[column].unique():
            rowIndex = self.data[self.data[column]==value].index.tolist()
            self.data.loc[rowIndex, column] = category
            category += 1
        self.toNumeric(column)


    def createTarget(self):
        self.setClassificationType(self.targetFeature)
        if self.isObject(self.targetFeature):
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


    def trainClassifier(self, features, scaling=False, pca=False, components=10):
        self.scaling = scaling
        self.pca = pca
        trainData = self.trainData[features].tolist()
        if scaling:
            trainData = self.scaleData(trainData)
        if pca:
            trainData = self.PCA(trainData, components)
        target = self.trainTarget.tolist()
        self.classifier.fit(trainData, target)

    def validate(self, features):
        self.holdout['predictedLabel'] = self.classifier.predict(self.holdout[features].tolist())
        self.validation = Evaluation(self.holdoutTarget, self.holdout.predictedLabel.tolist())
        self.validation.accuracy()
        self.validation.recall()
        self.validation.precision()

    
    def predict(self, features):
        testData = self.testData[features].tolist()
        if self.scaling:
            testData = self.scaleData(testData)
        if self.pca:
            testData = self.pca.transform(testData)
        self.testData['predictedLabel'] = self.classifier.predict(testData)
        if self.classifier.predict_proba:
            probabilities = self.classifier.predict_proba(testData)
            self.testData['probability'] = np.max(probabilities, axis=1)


    def evaluate(self, avgType='macro'):
        self.setEvaluationAverage(avgType)
        self.evaluation = Evaluation(self.testTarget, self.testData.predictedLabel.tolist(),self.evaluationAverage)
        self.evaluation.setAllTags()
        self.tagTestData()

        self.evaluation.accuracy()
        self.evaluation.recall()
        self.evaluation.precision()

    def setEvaluationAverage(self, avgType='macro'):
        self.evaluationAverage = avgType
        if self.classificationType=='binary':
            self.evaluationAverage = 'binary'


    def relevantFeatures(self):
        featureRelevance = self.classifier.feature_importances_
        indicesRelFeatures = np.where(featureRelevance>0)[0]
        vocabulary = self.preprocessor.vocabulary
        relFeatures = [(vocabulary[ind], featureRelevance[ind]) for ind in indicesRelFeatures]
        self.featureImportance = sortTupleList(relFeatures)


    def dropNANRows(self, columns=None):
        if columns:
            self.data = self.data[pd.notnull(self.data[columns])]
        else:
            self.data = self.data.dropna()

    def mergeDataset(self, dataset2):
        self.data = pd.merge(self.data, dataset2, on=['id'])


    def addTag(self, tag):
        indices = getattr(self.evaluation, tag)
        tagIndices = [self.testIndices[position] for position in indices]
        self.testData.loc[tagIndices,'tag'] = tag

    def tagTestData(self):
        tags = ['T','F']
        if self.classificationType=='binary':
            tags = ['TP', 'FP', 'TN', 'FN']
        for tag in tags:
            self.addTag(tag)

    def oneHotEncoding(self, data):
        return pd.get_dummies(data)

    def buildClassifier(self, classifierType):
        self.classifierType = classifierType
        if classifierType == 'DecisionTree':
            self.classifier = DecisionTreeClassifier()
            self.parameters = [{'min_samples_leaf': [2,5], 'max_depth':[3,5,7], 'criterion':['gini','entropy']}]
        elif classifierType == 'MultinomialNB':
            self.classifier = MultinomialNB()
            self.parameters = [{'alpha':[0,0.01, 0.3, 0.6, 1], 'fit_prior':[True, False],}]
        elif classifierType == 'BernoulliNB':
            self.classifier = BernoulliNB()
            self.parameters = [{'alpha':[0, 0.01, 0.3, 0.6, 1], 'binarize':[True, False], 'fit_prior':[True, False],}]
        elif classifierType == 'RandomForest':
            self.classifier = RandomForestClassifier()
            self.parameters = [{'n_estimators':[5,10,30], 'min_samples_leaf': [2,5], 'max_depth':[None,3,5,7], 'criterion':['gini','entropy']}]
        elif classifierType == 'SVM':
            self.classifier = svm.SVC(probability=True)
            self.parameters = [{'kernel':['rbf'], 'gamma':['auto', 0.5, 0.9], 'C':[0.5, 1]}]
        elif classifierType == 'LogisticRegression':
            self.classifier = linear_model.LogisticRegression()
            self.parameters = [{'penalty':['l1','l2'], 'C':[0.3,0.5,1,10]}]
        elif classifierType == 'kNN':
            self.classifier = neighbors.KNeighborsClassifier(n_neighbors=15)
            self.parameters = [{'n_neighbors':[3,5]}]


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


    def buildPreprocessor(self, vecType='tfIdf', min_df=10, max_df=0.5, stop_words='english', ngram_range = (1,2), max_features=8000, vocabulary=None, binary=False):
        self.preprocessor = Preprocessor(processor=vecType, min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=ngram_range, max_features=max_features, vocabulary=vocabulary, binary=binary)


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
        if os.path.exists(path):
            self.preprocessor = preprocessor.load(path)

        
