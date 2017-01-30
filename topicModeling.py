#!/usr/bin/python
#-*- coding: utf-8 -*-
from lda import Collection, Dictionary, Model, Info, Viewer, utils, ImagePlotter, Word2Vec, docLoader
from lda.docLoader import loadCategories
from gensim.parsing.preprocessing import STOPWORDS
import pandas as pd
from nltk.corpus import names
from gensim.models import TfidfModel
import bnpy_run
import os.path
from sklearn.feature_extraction.text import CountVectorizer 

def topicModeling():

    #### PARAMETERS ####
    info = Info()
    info.data = 'Aleph'     # 'ICAAD' 'NIPS' 'scifibooks' 'HRC'
   
    # Preprocessing #
    info.preprocess = 0
    info.startDoc = 0 
    info.numberDoc= None 
    info.specialChars = set(u'''[,\.\'\`=\":\\\/_+]''')
    info.includeEntities = 0
    info.removeNames = 0
    info.bigrams = 1
    info.maxFeatures = 10000

    numbers = [str(nr) for nr in range(0,500)]
    info.whiteList= Word2Vec().net.vocab.keys() #+ numbers
    info.stoplist = list(STOPWORDS) #+ utils.lowerList(names.words())
    info.stoplist = [x.strip() for x in open('stopwords/english.txt')]

    # Dictionary #  
    info.analyseDictionary = 0

    info.lowerFilter = 1     # in number of documents
    info.upperFilter = 95  # in percent

    # LDA Model #
    info.modelType = 'LDA'  # 'LDA' 'LSI'
    info.numberTopics = 20 
    info.tfidf = 0
    info.passes = 5 
    info.iterations = 60 
    info.online = 0 
    info.chunksize = 4100 
    info.multicore = 1

    # Evaluation #
    info.categories = loadCategories('Documents/categories.txt')[2]     #0 -human rights categories   1 - Scientific Paper categories
    
    info.setup()

    #### MODEL ####
    collection = Collection()
    html = Viewer(info)
        
    if not os.path.exists(info.collectionName) or info.preprocess:
        print 'Load and preprocess Document Collection'
        titles, text = docLoader.loadEncodedFiles(info.path)
        data = pd.DataFrame([titles[0:info.numberDoc], text[0:info.numberDoc]], index = ['title', 'text'])
        data = data.transpose()

        text = data['text'].get_values()

        vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2), token_pattern='[a-zA-Z]+', stop_words=info.stoplist, max_features=8000, binary=True, max_df=1200, min_df=5)
        print 'Fit transform'
        word_counts = vectorizer.fit_transform(text)
        print 'Get vocabulary'
        vocabulary = vectorizer.get_feature_names()
        print 'Compute Tokens'
        tokens = [[vocabulary[index] for index in doc.indices] for doc in word_counts]
        

        collection.documents = collection.createDocumentList(data.title.tolist(), data.text.tolist())
        collection.setDocNumber()

        vocabulary, tokens = bnpy_run.preprocess(data)
        #tokenPerDocument = bnpy_data.getSparseDocTypeCountMatrix().toarray()
        #tokens = [[vocabulary[index] for index, freq in enumerate(tokens) if freq>0] for tokens in tokenPerDocument]
        
        collection.addFeatureToDocuments('tokens', tokens)
        #collection.prepareDocumentCollection(lemmatize=True, includeEntities=info.includeEntities, stopwords=info.stoplist, removeShortTokens=True, threshold=2, specialChars=info.specialChars, whiteList=info.whiteList, bigrams=info.bigrams)
        collection.saveDocumentCollection(info.collectionName)
    else:
        print 'Load Processed Document Collection'
        collection.loadPreprocessedCollection(info.collectionName)

        for document in collection.documents:
            print document.nr 
            document.tokens = [word for word in document.tokens if len(word)>3]

    print 'Create Dictionary'
    dictionary = Dictionary(info.stoplist)
    for doc in collection.documents:
        print doc.nr
        dictionary.addDocument(doc)
    #dictionary.addCollection(collection.documents)

    if info.analyseDictionary:
        'Analyse Word Frequency'
        collectionLength = collection.number
        dictionary.analyseWordFrequencies(info, html, collectionLength)
    
    print 'Filter extremes'
    dictionary.ids.filter_extremes(info.lowerFilter, info.upperFilter)
   
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
        document.setTopicCoverage(topicCoverage[document.nr], lda.name)
        lda.computeSimilarity(document)
        collection.computeRelevantWords(tfidf, dictionary, document)
        maxTopicCoverage.append(document.LDACoverage[0][1])

    ImagePlotter.plotHistogram(maxTopicCoverage, 'Maximal Topic Coverage', 'html/' + info.data+'_'+info.identifier+'/Images/maxTopicCoverage.jpg', 'Maximal LDA Coverage', 'Number of Docs', log=1)
    
    print 'Create HTML Files'
    html.printTopics(lda)
    html.htmlDictionary(dictionary)
    html.printDocuments(collection.documents, lda)# , openHtml=True)
    html.printDocsRelatedTopics(lda, collection.documents, openHtml=False)
    html.documentOverview(collection.documents)

    #info.selectedTopics = input('Select Topics: ')
    #collection.writeDocumentFeatureFile(info, info.selectedTopics)
    info.saveToFile()

   
   
if __name__ == "__main__":
    topicModeling()

