import urllib2
import docLoader
from Dictionary import Dictionary
from Document import Document
from Evaluation import Evaluation
from ClassificationModel import ClassificationModel
import sPickle
import pandas as pd
import numpy as np

class Collection:
    
    def __init__(self):
        self.documents = [] 

    def load(self, path=None, fileType=0, startDoc="couchdb", numberDocs=None):
        if path is not None and fileType=="couchdb":
            urllib2.urlopen(urllib2.Request(path)) 
            (titles, texts) = docLoader.loadCouchdb(path)
        elif path is not None and fileType == "folder":
            (titles, texts) = docLoader.loadTxtFiles(path)
        elif path is not None and fileType == "csv":
            (titles, texts) = docLoader.loadCsvFile(path)
        else:
            titles = ['']
            texts = ['']
        if numberDocs is None:
            numberDocs = len(titles)
        self.documents = self.createDocumentList(titles[startDoc:startDoc + numberDocs], texts[startDoc:startDoc + numberDocs])
        self.number = len(self.documents)

    def setDocNumber(self):
        for index, doc in enumerate(self.documents):
            doc.nr = index
    
    def createCorpus(self, dictionary):
        corpus = []
        for document in self.documents:
            vectorRepresentation = dictionary.ids.doc2bow(document.tokens)
            corpus.append(vectorRepresentation)
            document.setAttribute('vectorRepresentation', vectorRepresentation)
        return corpus


    def loadPreprocessedCollection(self, filename):
        collection = []
        for doc in sPickle.s_load(open(filename)):
            collection.append(doc)
        self.documents = collection
        self.number = len(self.documents)


    def createEntityCorpus(self, dictionary):
        self.entityCorpus = [sorted([(dictionary.getDictionaryId(entry[0]), entry[1]) for entry in document.entities.getEntities()]) for document in self.documents]


    def addFeatureToDocuments(self, featureName, featureList):
        if len(featureList)==len(self.documents):
            for index,document in enumerate(self.documents):
                document.setAttribute(featureName, featureList[index])
        else:
            print 'Length of documents and features is not equale'



    def computeRelevantWords(self, tfidf, dictionary, document, N=10):
        docRepresentation = tfidf[document.vectorRepresentation]
        freqWords = sorted(docRepresentation, key=lambda frequency: frequency[1], reverse=True)[0:N]
        freqWords = [(dictionary.getWord(item[0]), item[1], item[0]) for item in freqWords]
        document.setAttribute('freqWords', freqWords)


#    def createFrequentWords(self, N=10):
#        for index, docs in enumerate(self.documents):
#            self.setFrequentWordsInDoc(docs, N=N)


    def applyToAllDocuments(self, f):
        for document in self.documents:
            f(document)


    def computeVectorRepresentation(self, document):
        document.setAttribute('vectorRepresentation', self.dictionary.ids.doc2bow(document.tokens))


    def prepareDocumentCollection(self, lemmatize=True, createEntities=True, includeEntities=True, stopwords=None, specialChars = None, removeShortTokens=True, threshold=2, whiteList = None, bigrams = False):
        for index, document in enumerate(self.documents):
            print index, document.title
            document.prepareDocument(lemmatize, includeEntities, stopwords, specialChars, removeShortTokens=True, threshold=threshold, whiteList = whiteList, bigrams=bigrams)

    def writeDocumentFeatureFile(self, info, topics, keywords):
        columns = self._createColumns(topics) + keywords
        dataframe = pd.DataFrame(np.nan, index = range(0, self.number), columns = columns)
        for ind, document in enumerate(self.documents):
            coverageDictionary = dict(document.LDACoverage)
            coverage = [coverageDictionary.get(nr, 0.0) for nr in topics]
            similarity = [document.LDASimilarity[nr][0] for nr in range(1, 6)]
            relevantWords = [document.freqWords[nr][2] for nr in range(0, 3) if len(document.freqWords)>=3]
            values = [document.title, document.id, document.SA, document.DV, document.court, document.year] + coverage + similarity + relevantWords 
            if hasattr(document, 'targetCategories'):
                values = values + list(zip(*document.targetCategories)[1])
            values = values + [np.nan] * (len(columns) - len(values))
            dataframe.loc[ind] = values
            for word in document.entities.getEntities():
                dataframe.loc[ind,word[0]] = word[1]
        dataframe = dataframe.dropna(axis=1, how='all')
        dataframe = dataframe.fillna(0)
        path = 'html/'+ info.data +'_' + info.identifier + '/DocumentFeatures.csv'
        dataframe.to_csv(path)   

    def _createColumnNames(self, title, number):
        return [(title + '%d' % nr) for nr in range(1, number+1)]

    def _createTopicNames(self, topics):
        return [('Topic%d' % topicNr) for topicNr in topics]

    def _createColumns(self, topics):
        properties = self.documents[0].__dict__.keys()
        columnNamesTopic = self._createTopicNames(topics)
        columnNamesRelevantWords = self._createColumnNames('relevantWord', 3)
        columnNamesSimilarDocs = self._createColumnNames('similarDocs', 5)
        columns = ['File', 'id', 'SA', 'DV', 'court', 'year'] + columnNamesTopic + columnNamesSimilarDocs + columnNamesRelevantWords
        hasTargetCategories = 'targetCategories' in properties
        if hasTargetCategories:
            columnNamesTarget = self._createColumnNames('targetCategory', 3) 
            columns = columns + columnNamesTarget
        return columns

    
    def createDocumentList(self, titles, texts):
        return [Document(title, text) for title, text in zip(titles, texts)]

    def saveDocumentCollection(self, path):
        sPickle.s_dump(self.documents, open(path, 'w'))

    def evaluate(self, feature='SA'):
        target, prediction = zip(*[(getattr(doc, feature), getattr(doc, 'pred'+feature)) for doc in self.documents if doc.id != 'nan'])
        evaluation = Evaluation(target, prediction)
        evaluation.feature = feature
        evaluation.setAllTags()
        evaluation.accuracy()
        evaluation.recall()
        evaluation.precision()
        evaluation.confusionMatrix()
        return evaluation

    def getConfusionDocuments(self, feature):
        matches = ['TP', 'FP', 'TN', 'FN']
        for match in matches:
            docs = self.getTaggedDocuments(feature, match)
            setattr(self, feature+'_'+match, docs)

    def getTaggedDocuments(self, feature, tag):
        return [(doc.title, ind) for ind, doc in enumerate(self.documents) if getattr(doc, feature+'tag')==tag and doc.id != 'nan'] 


