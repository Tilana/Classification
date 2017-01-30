from Entities import Entities
import utils
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.collocations import *
from collections import Counter
from itertools import repeat
import re

class Document:
    
    def __init__(self, title=None, text=None):
        self.title = title
        self.text = text 
        self.entities = Entities()
       
    def createEntities(self, frequency=1):
        self.entities = Entities(self.text)

    def createTokens(self):
        self.tokens= self._tokenizeDocument()

    def prepareDocument(self, lemmatize=True, includeEntities=True, stopwords=None, specialChars=None, removeShortTokens=True, threshold=1, whiteList = None, bigrams=False):
        self.text = self.text.decode('utf8', 'ignore')
        self.tokens = self._tokenizeDocument()
        if stopwords is None:
            stopwords = []
        if specialChars is None:
            specialChars = []
        if whiteList is None:
            whiteList = list(set(self.tokens) - set(stopwords))
        if lemmatize:
            self.lemmatizeTokens()
        self.tokens = [token for token in self.tokens if (token not in stopwords) and (token in whiteList)]
        self.tokens = [token for token in self.tokens if not utils.containsAny(token, specialChars) and len(token) > threshold]
        if bigrams:
            bigramWhiteList = utils.getBigrams(whiteList)
            self.createBigrams(30, bigramWhiteList)
            self.addBigramsToTokens()
        if includeEntities:
            if self.entities.isEmpty():
                self.createEntities()
            self.appendEntities()

    def lemmatizeTokens(self):
        wordnet = WordNetLemmatizer()
        self.original = self.tokens
        self.tokens = [wordnet.lemmatize(wordnet.lemmatize(word, 'v')) for word in self.tokens]

    def findSpecialCharacterTokens(self, specialCharacters):
        self.specialCharacters =  [word for word in self.tokens if utils.containsAny(word, specialCharacters)]

    def removeSpecialCharacters(self):
        for specialChar in self.specialCharacters:
            self.tokens.remove(specialChar)

    def addBigramsToTokens(self):
        for bigram in self.bigrams:
            frequency = self.bigramCounter(bigram)


    def hasTokenAttribute(self):
        return hasattr(self, 'tokens')

    def hasOriginalAttribute(self):
        return hasattr(self, 'original')
    
    def hasSpecialCharAttribute(self):
        return hasattr(self, 'specialCharacters')

    def setTopicCoverage(self, coverage, name):
        sortedCoverage = utils.sortTupleList(coverage)
        self.setAttribute(('%sCoverage' % name), sortedCoverage)
   
    def _tokenizeDocument(self):
        return [word.lower() for word in nltk.word_tokenize(self.text)]

    def removeShortTokens(self, threshold=1):
        shortWords = [word for word in self.tokens if len(word)<=threshold]
        self.tokens = [word for word in self.tokens if word not in shortWords]


    def removeStopwords(self, stoplist):
        self.tokens = [word for word in self.tokens if word not in stoplist]

    def appendEntities(self):
        entityList = self.entities.getEntities()
        for entity in entityList:
            for frequency in range(0, entity[1]):
                self.tokens.append(entity[0].encode('utf8'))
                self.correctTokenOccurance(entity[0])

    def correctTokenOccurance(self, entity):
        [self.tokens.remove(word) for word in entity.split() if word in self.tokens]

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def setAttribute(self, name, value):
        setattr(self, name, value)

    def countOccurance(self, wordList):
        self.counter = self.tokenCounter
        if hasattr(self, 'bigramCounter'):
            self.counter = self.counter + self.bigramCounter
        return [(word, self.counter[word]) for word in wordList if self.counter[word]>0]

    def createTokenCounter(self):
        self.tokenCounter = Counter(self.tokens)

    def createBigrams(self, n, whiteList):
        finder = BigramCollocationFinder.from_words(self.tokens)
        bigrams = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))[:n]
        self.bigramCounter = Counter(dict(utils.convertTupleToString(bigrams)))
        self.bigrams = self.bigramCounter.keys()

    def findBigrams(self, finder, bigramList, measure):
        return [bigram for bigram in bigramList if bigram in finder.nbest(measure, finder.N)]

    def addBigramsToTokens(self):
        replicateBigramFrequency = [word for bigram in self.bigrams for word in repeat(bigram, self.bigramCounter[bigram])]
        self.tokens = self.tokens + replicateBigramFrequency 
        
    def extractYear(self):
        year = re.findall(r'\[\d*]', self.title)
        if year != []:
            year = year[0].replace('[','').replace(']','')
            self.year = int(year)
        else:
            self.year = 'nan'

    def extractCourt(self):
        court = re.findall(r'\] \w*', self.title)
        if court != []:
            self.court = court[0].replace(']','').strip()
        else:
            self.court = 'nan'

    def predictCases(self, target, info, threshold=0.2):
        LDACoverage = dict(self.LDACoverage)
        topics = getattr(info, target+'Topics')
        setattr(self, 'pred'+target, False)
        coverage = [LDACoverage.get(topicNr, 0.0) for topicNr in topics]
        if max(coverage) >= threshold:
            setattr(self, 'pred'+target, True)

    def tagPrediction(self, feature):
        target = getattr(self, feature)
        prediction = getattr(self, 'pred'+feature)
        tag = feature+'tag'
        if prediction==True:
            if prediction==target:
                setattr(self, tag, 'TP')
            else:
                setattr(self, tag, 'FP')
        else:
            if prediction == target:
                setattr(self, tag, 'TN')
            else:
                setattr(self, tag, 'FN')
        
