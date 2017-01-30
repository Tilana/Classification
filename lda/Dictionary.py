#!/usr/bin/env python
# -*- coding: utf-8 -*-
from Entities import Entities
import ImagePlotter
import utils
from nltk.stem import WordNetLemmatizer
from gensim import corpora
import matplotlib.pyplot as plt

class Dictionary:
        
    def __init__(self, stopwords=None):
        self.words = set([])
        self.ids = corpora.Dictionary()
        self.specialCharacters = set([])
        self.stopwords = set([]) if stopwords is None else stopwords           

    def addCollection(self, collection):
        for document in collection:
            self.addDocument(document)

    def addDocument(self, document):
        if document.hasTokenAttribute():
            self.ids.add_documents([document.tokens])
        if document.hasSpecialCharAttribute():
            self.specialCharacters.update(document.specialCharacters)

    def getDictionaryId(self, word):
        return self.ids.keys()[self.ids.values().index(word)]


    def getWord(self, index):
        return self.ids.get(index)

    
    def plotWordDistribution(self, info, start=None, end=None):
        path = 'html/'+info.data + '_' + info.identifier + '/Images/'
        if start != None:
            distribution = [freq+1 for freq in self.ids.dfs.values() if freq>=start and freq <= end]
            title = 'Word-Document Histogram  %d -  %d documents' % (start, end)
            filename = 'wordDistribution_%d_%d.jpg' % (start, end)
            bins = 10
        else:
            distribution = [freq+1 for freq in self.ids.dfs.values()]
            title = 'Word-Document Histogram'
            filename = 'wordDistribution.jpg'
            bins = 20
        ImagePlotter.plotHistogram(distribution, title, path+filename, 'Number of Documents', 'Word Frequency', log=1, bins=bins, open=0)

        
    def createEntities(self, collection):
        [document.createEntities() for document in collection if document.entities.isEmpty()]
        self.entities = Entities('')
        self._addDocumentEntities(collection)
    
    def encodeWord(self, word):
        return self.ids.get(word)

    def invertDFS(self):
        self.inverseDFS = {}
        for key, value in self.ids.dfs.items():
            if value not in self.inverseDFS:
                self.inverseDFS[value] = []
            self.inverseDFS[value].append(self.ids.get(key))
    
    
    def _addDocumentEntities(self, collection):
        for tag in collection[0].entities.__dict__.keys():
            self.entities.addEntities(tag, set().union(*[getattr(document.entities, tag) for document in collection]))
        for entity in self.entities.getEntities():
            self.words.add(entity[0].lower())

    def analyseWordFrequencies(self, info, html, length):
        halfLength = length/2
        self.plotWordDistribution(info)                    
        self.plotWordDistribution(info, 1,10)
        self.plotWordDistribution(info, halfLength, length)

        self.invertDFS()
        html.wordFrequency(self, 1, 10)
        html.wordFrequency(self, 10, halfLength)
        html.wordFrequency(self, halfLength, length) 


    def filter_extremes(self, lowerFilter, upperFilter, whiteList):
        absUpperFilter = int(upperFilter * self.ids.num_docs)
        filteredWords = [key for (key, word) in self.ids.iteritems() if lowerFilter <= self.ids.dfs.get(key,0) <= absUpperFilter or word in whiteList]
        self.ids.filter_tokens(good_ids = filteredWords)


