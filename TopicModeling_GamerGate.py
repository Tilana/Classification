from lda import Collection, Model, Info, Viewer
from lda.listUtils import flattenList
import csv
import pdb
import gensim
import numpy as np
import pandas as pd
from scipy import io

def TopicModeling_GamerGate():

    info = Info()
    info.data = 'GamersGate'
    info.modelType = 'LDA'
    info.numberTopics = 5 
    info.passes = 2
    info.iterations = 70 
    info.name = 'GamersGate'
    info.identifier = 'TM5'
    info.online = 1
    info.multicore = 0
    info.chunksize = 5000

    #with open('Documents/HRC_topics.csv', 'rb') as f:
    #    reader = csv.reader(f)
    #    targets = flattenList(list(reader))
    #info.categories = targets
    info.categories = ['women', 'health', 'democracy', 'terrorism', 'water', 'peasants', 'trafficking', 'children', 'journalists', 'arms', 'torture', 'slavery', 'climate', 'poverty', 'corruption', 'housing', 'religion', 'internet', 'sport', 'governance', 'truth']
    
    lda = Model(info)
    #pdb.set_trace()

    modelPath = 'processedData/gamerGate'
    #path = 'TopicModel/HRC_test_store'

    collection = Collection().load(modelPath)
    #collection.data['id'] = range(len(collection.data))

    #pdb.set_trace()
    documents = collection.data.cleanText.tolist()

    vectorizer = collection.preprocessor.vectorizer
    tfIdf = vectorizer.fit_transform(documents)
    #io.mmsave('TopicModel/tfIdf')
    dictionary = vectorizer.vocabulary_

    id2word = dict((v, k) for k, v in dictionary.iteritems())

    #id2word = np.load('TopicModel/dict.npy').item()
    #tfIdf = io.mmread('TopicModel/tfIdf').tocsr()
    corpus = gensim.matutils.Sparse2Corpus(tfIdf, documents_columns=False) 

    lda.createModel(tfIdf, id2word, info)
    lda.createTopics(info)

    #pdb.set_trace()

    topicCoverage = lda.model[corpus]
    
    print 'Get Documents related to Topics'
    lda.getTopicRelatedDocuments(topicCoverage, info)
    
    print 'Similarity Analysis'
    #lda.computeSimilarityMatrix(corpus, numFeatures=info.numberTopics, num_best = 7)

    topicCoverage = gensim.matutils.corpus2csc(topicCoverage)
    colNames = ['Topic'+str(elem) for elem in range(info.numberTopics)]
    topicDF = pd.DataFrame(topicCoverage.todense(), index=colNames)
    collection.data = pd.concat([collection.data, topicDF.T], axis=1)

    #pdb.set_trace()

    viewer = Viewer(info.data+'_'+info.identifier)
    viewer.printTopics(lda)

    displayFeatures = colNames 
    viewer.printDocuments(collection.data, displayFeatures)
    collection.data['title'] = collection.data['user_name']
    viewer.printDocsRelatedTopics(lda, collection.data)

    pdb.set_trace()



if __name__ == '__main__':
    TopicModeling_GamerGate()
