# encoding=utf8
from docLoader import loadData
from Preprocessor import Preprocessor
#import preprocessor as tweetPreprocessor
from FeatureExtractor import FeatureExtractor
import cPickle as pickle
import listUtils as utils
import pandas as pd
import numpy as np
import os
import pdb

class Collection:

    def __init__(self, path=None):
        self.data = pd.DataFrame()
        if path:
            self.path = path
            self.data = loadData(path)
        self.nrDocs = len(self.data)


    def cleanDataframe(self, textField='text'):
        dataWithText = self.data[self.data[textField].notnull()]
        cleanDataframe = dataWithText.dropna(axis=1, how='all')
        self.data  = cleanDataframe.reset_index()
        self.nrDocs = len(self.data)


    def vectorize(self, vecType='tfidf', vocabulary=None, field='text', ngrams=(1,2), maxFeatures=8000):
        self.buildVectorizer(vecType=vecType, ngram_range=ngrams, min_df=5, max_df=0.50, max_features=maxFeatures, binary=False, vocabulary=vocabulary)
        self.trainVectorizer(vecType, field)


    def buildVectorizer(self, vecType='tfIdf', min_df=10, max_df=0.5, stop_words='english', ngram_range = (1,2), max_features=8000, vocabulary=None, binary=False):
        self.preprocessor = Preprocessor(processor=vecType, min_df=min_df, max_df=max_df, stop_words=stop_words, ngram_range=ngram_range, max_features=max_features, vocabulary=vocabulary, binary=binary)


    def trainVectorizer(self, vecType='tfIdf', field='text'):
        trainDocs = self.data[field].tolist()
        self.data[vecType] = self.preprocessor.trainVectorizer(trainDocs)
        self.vocabulary = self.preprocessor.vocabulary


    def save(self, path):
        self.savePreprocessor(path)
        with open(path +'.pkl', 'wb') as f:
            pickle.dump(self, f, -1)


    def load(self, path):
        model = pickle.load(open(path+'.pkl', 'rb'))
        self.data = model.data
        self.loadPreprocessor(path)
        return self


    def savePreprocessor(self, path):
        if hasattr(self, 'preprocessor'):
            self.preprocessor.save(path)
            del self.preprocessor


    def loadPreprocessor(self, path):
        preprocessor = Preprocessor()
        if os.path.exists(path+'.pkl'):
            self.preprocessor = preprocessor.load(path)
            self.vocabulary = self.preprocessor.vocabulary


    def existsProcessedData(self, path):
        return os.path.exists(path + '.pkl')


    def cleanTexts(self):
        preprocessor = Preprocessor()
        self.applyToRows('text', preprocessor.removeHTMLtags, 'cleanText')
        self.applyToRows('cleanText', preprocessor.cleanText, 'cleanText')
        self.applyToRows('cleanText', preprocessor.numbersInTextToDigits, 'cleanText')


    def removeStopwords(self):
        preprocessor = Preprocessor()
        self.applyToRows('cleanText', preprocessor.removeStopwords, 'cleanTokens')


    def cleanTweets(self):
        self.applyToRows('decodeTweet', tweetPreprocessor.clean, 'cleanTweets')

    def extractDate(self):
        self.applyToRows('tweet_time', separateDateAndTime, 'date')


    def extractDate(self):
        self.data['date'] = self.data.apply(lambda doc: doc['tweet_time'].split('T')[0], axis=1)


    def extractEntities(self, path=None):
        featureExtractor = FeatureExtractor(path)
        self.applyToRows('text', featureExtractor.entities, 'entities')

    def setRelevantWords(self):
        self.applyToRows('tfidf', self.relevantWords, 'relevantWords')

    def relevantWords(self, wordWeights):
        sortedWeights = sorted(enumerate(wordWeights), key=lambda x: x[1], reverse=True)
        sortedWeights = [wordWeight for wordWeight in sortedWeights if wordWeight[1]>0]
        return [(self.vocabulary[wordIndex], weight) for (wordIndex, weight) in sortedWeights]



    def applyToRows(self, field, fun, name, args=None):
        if args:
            self.data[name] = self.data.apply(lambda doc: fun(doc[field], args), axis=1)
        else:
            self.data[name] = self.data.apply(lambda doc: fun(doc[field]), axis=1)
