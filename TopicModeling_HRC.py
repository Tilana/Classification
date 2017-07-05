from lda import Collection, Model, Info, Viewer
from lda.listUtils import flattenList
import csv
import pdb
import gensim
import numpy as np
import pandas as pd
from scipy import io

def TopicModeling_HRC():

    info = Info()
    info.data = 'HRC'
    info.modelType = 'LDA'
    info.numberTopics = 4
    info.passes = 3
    info.iterations = 100
    info.name = 'HRC'
    info.identifier = 'TM4'
    info.online = 0
    info.multicore = 0
    info.chunksize = 1000

    #with open('Documents/HRC_topics.csv', 'rb') as f:
    #    reader = csv.reader(f)
    #    targets = flattenList(list(reader))
    #info.categories = targets
    info.categories = ['women', 'health', 'democracy', 'terrorism', 'water', 'peasants', 'trafficking', 'children', 'journalists', 'arms', 'torture', 'slavery', 'climate', 'poverty', 'corruption', 'housing', 'religion', 'internet', 'sport', 'governance', 'truth']
    
    lda = Model(info)

    modelPath = 'processedData/RightDocs_topics'
    #path = 'TopicModel/HRC_test_store'

    collection = Collection().load(modelPath)
    #documents = collection.data.cleanText.tolist()

    #vectorizer = collection.preprocessor.vectorizer
    #tfIdf = vectorizer.fit_transform(documents)
    #dictionary = vectorizer.vocabulary_

    #id2word = dict((v, k) for k, v in dictionary.iteritems())

    id2word = np.load('TopicModel/dict.npy').item()
    tfIdf = io.mmread('TopicModel/tfIdf').tocsr()
    corpus = gensim.matutils.Sparse2Corpus(tfIdf, documents_columns=False) 

    lda.createModel(tfIdf, id2word, info)
    lda.createTopics(info)

    topicCoverage = lda.model[corpus]
    
    print 'Get Documents related to Topics'
    lda.getTopicRelatedDocuments(topicCoverage, info)
    
    print 'Similarity Analysis'
    #lda.computeSimilarityMatrix(corpus, numFeatures=info.numberTopics, num_best = 7)


    topicCoverage = gensim.matutils.corpus2csc(topicCoverage)
    colNames = ['Topic'+str(elem) for elem in range(info.numberTopics)]
    topicDF = pd.DataFrame(topicCoverage.todense(), index=colNames)
    collection.data = pd.concat([collection.data, topicDF.T], axis=1)

    viewer = Viewer(info.data+'_'+info.identifier)
    viewer.printTopics(lda)

    displayFeatures = colNames 
    viewer.printDocuments(collection.data, displayFeatures)
    viewer.printDocsRelatedTopics(lda, collection.data)



if __name__ == '__main__':
    TopicModeling_HRC()
