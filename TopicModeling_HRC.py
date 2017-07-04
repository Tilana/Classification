from lda import Collection, Model, Info
import pdb
import gensim
import numpy as np
from scipy import io

def TopicModeling_HRC():

    info = Info()
    info.data = 'HRC'
    info.modelType = 'LDA'
    info.categories = ['human', 'environment']
    info.numberTopics = 15
    info.passes = 3
    info.iterations = 100
    info.name = 'HRC'
    info.identifier = 'test'
    info.online = 0
    info.multicore = 0
    info.chunksize = 1000

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
    
    pdb.set_trace()

    print 'Get Documents related to Topics'
    lda.getTopicRelatedDocuments(topicCoverage, info)
    
    print 'Similarity Analysis'
    lda.computeSimilarityMatrix(corpus, numFeatures=info.numberTopics, num_best = 7)

    pdb.set_trace()
    #collection.setTopicCoverage(topicCoverage)

    topicCoverage = gensim.matutils.corpus2csc(topicCoverage)
    topicDF = pd.DataFrame(topicCoverage.todense())
                                                              
    data = pd.concat([collection.data, topicDF.T], axis=1)


if __name__ == '__main__':
    TopicModeling_HRC()
