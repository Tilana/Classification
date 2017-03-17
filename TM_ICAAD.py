#!/usr/bin/python
#-*- coding: utf-8 -*-
from lda import Collection, Dictionary, Model, Info, Viewer, utils, Word2Vec, ImagePlotter
from lda.docLoader import loadCategories
from gensim.parsing.preprocessing import STOPWORDS
from nltk.corpus import names
from gensim.models import TfidfModel
import os.path
from lda import dataframeUtils as df
import csv
import pandas as pd
import pdb

def TM_ICAAD():

    path = 'processedData/processedData'
    info = Info()
    # Categories and Keywords
    info.categories = loadCategories('Documents/categories.txt')[0]     #0 -human rights categories   1 - Scientific Paper categories
    keywordFile = 'Documents/ICAAD/CategoryLists.csv'
    keywords_df = pd.read_csv(keywordFile).astype(str)
    keywords = list(df.toListMultiColumns(keywords_df, keywords_df.columns))

    #### PARAMETERS ####
    word2vec = Word2Vec()
    info.data = 'ICAAD'     # 'ICAAD' 'NIPS' 'scifibooks' 'HRC'

    # Preprocessing # 
    info.preprocess = 0
    info.startDoc = 0 
    info.numberDoc= None 
    info.specialChars = set(u'''[,\.\'\`=\":\\\/_+]''')
    info.includeEntities = 0
    info.bigrams = 1

    numbers = [str(nr) for nr in range(0,500)]
    info.whiteList= word2vec.net.vocab.keys() + numbers + keywords
    info.stoplist = list(STOPWORDS) + utils.lowerList(names.words())
    info.stoplist = [x.strip() for x in open('stopwords/english.txt')]

    info.removeNames = 1

    # Dictionary #
    info.analyseDictionary = 0
                                                              
    info.lowerFilter = 8      # in number of documents
    info.upperFilter = 0.3   # in percent

    # LDA #
    info.modelType = 'LDA'  # 'LDA' 'LSI'
    info.numberTopics = 35
    info.tfidf = 0
    info.passes = 32 
    info.iterations = 70 
    info.online = 1 
    info.chunksize = 4100                                        
    info.multicore = 1
    
    info.setup()

    #### EVALUATION ####
    evaluationFile = 'Documents/PACI.csv'
    dataFeatures = pd.read_csv(evaluationFile)
    filenames = dataFeatures['Filename'].tolist()
    filenames = [name.replace('.txt', '') for name in filenames]
    dataFeatures['Filename'] = filenames
    dataFeatures = dataFeatures.rename(columns = {'Unnamed: 0': 'id'})

    #### MODEL ####
    collection = Collection()
    html = Viewer(info) 
    pdb.set_trace()

    if not os.path.exists(info.collectionName) or info.preprocess:
        print 'Load and preprocess Document Collection'
        collection.load(info.path, info.fileType, info.startDoc, info.numberDoc)
        collection.setDocNumber()
        for doc in collection.documents:
            doc.title = doc.title.replace('.rtf.txt', '')
            features = dataFeatures[dataFeatures['Filename']==doc.title]
            doc.id = df.getValue(features, 'id')
            doc.SA = df.getValue(features, 'Sexual.Assault.Manual')
            doc.DV = df.getValue(features, 'Domestic.Violence.Manual')
            doc.extractYear()
            doc.extractCourt()
            
        collection.prepareDocumentCollection(lemmatize=True, includeEntities=info.includeEntities, stopwords=info.stoplist, removeShortTokens=True, threshold=2, specialChars=info.specialChars, whiteList=info.whiteList, bigrams=info.bigrams)
        collection.saveDocumentCollection(info.collectionName)

    else:
        print 'Load Processed Document Collection'
        collection.loadPreprocessedCollection(info.collectionName)

    print 'Create Dictionary'
    dictionary = Dictionary(info.stoplist)
    dictionary.addCollection(collection.documents)

    if info.analyseDictionary:
        'Analyse Word Frequency'
        collectionLength = collection.number
        dictionary.analyseWordFrequencies(info, html, collectionLength)
    
    print 'Filter extremes'
    dictionary.filter_extremes(info.lowerFilter, info.upperFilter, keywords)

    if info.analyseDictionary:
        dictionary.plotWordDistribution(info)
    
    print 'Create Corpus'
    corpus = collection.createCorpus(dictionary)
   
    print 'TF_IDF Model'
    tfidf = TfidfModel(corpus, normalize=True)
    if tfidf:
        corpus = tfidf[corpus]

    print 'Topic Modeling - LDA'
    lda = Model(info)
    lda.createModel(corpus, dictionary.ids, info)
    lda.createTopics(info)

    print 'Topic Coverage'
    topicCoverage = lda.model[corpus]
    
    print 'Get Documents related to Topics'
    lda.getTopicRelatedDocuments(topicCoverage, info)
    
    print 'Similarity Analysis'
    lda.computeSimilarityMatrix(corpus, numFeatures=info.numberTopics, num_best = 7)

    maxTopicCoverage = []
    for document in collection.documents:
        docTopicCoverage = topicCoverage[document.nr]
        document.setTopicCoverage(docTopicCoverage, lda.name)
        lda.computeSimilarity(document)
        collection.computeRelevantWords(tfidf, dictionary, document)
        maxTopicCoverage.append(document.LDACoverage[0][1])
        document.createTokenCounter()
        for category in keywords_df.columns.tolist():
            wordsInCategory = df.getColumn(keywords_df, category) 
            keywordFrequency = document.countOccurance(wordsInCategory)
            document.entities.addEntities(category, utils.sortTupleList(keywordFrequency))
        document.mostFrequentEntities = document.entities.getMostFrequent(5)

    ImagePlotter.plotHistogram(maxTopicCoverage, 'Maximal Topic Coverage', 'html/' + info.data+'_'+info.identifier+'/Images/maxTopicCoverage.jpg', 'Maximal LDA Coverage', 'Number of Docs', log=1)

    print 'Create HTML Files'
    html.htmlDictionary(dictionary)
    html.printTopics(lda)
    
    info.SATopics = input('Sexual Assault Topics:')
    info.DVTopics = input('Domestic Violence Topics:')
    info.otherTopics = input('Other Topics: ')
    selectedTopics = info.SATopics + info.DVTopics + info.otherTopics
    info.SAthreshold = 0.2
    info.DVthreshold = 0.2

    for doc in collection.documents:
        doc.predictCases('SA', info, info.SAthreshold)
        doc.tagPrediction('SA')
        doc.predictCases('DV', info, info.DVthreshold)
        doc.tagPrediction('DV')
    SAevaluation = collection.evaluate('SA')
    collection.getConfusionDocuments('SA')
    html.results(SAevaluation, collection, info)
    DVevaluation = collection.evaluate('DV')
    collection.getConfusionDocuments('DV')
    html.results(DVevaluation, collection, info) 
    
    html.printDocuments(collection.documents, lda)
    html.printDocsRelatedTopics(lda, collection.documents, openHtml=False)
    html.documentOverview(collection.documents)

    print('Write Feature File')
    collection.writeDocumentFeatureFile(info, selectedTopics, keywords)
                                                                   
    info.saveToFile()
   
if __name__ == "__main__":
    TM_ICAAD()

