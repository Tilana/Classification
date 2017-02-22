import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from lda import ClassificationModel
import pdb

def clustering_ICAAD():

    modelPath = 'processedData/SADV_2gram'
    #modelPath = 'processedData/SADV'
    model = ClassificationModel()
    model = model.load(modelPath)

    nrCluster = 35 

    features = model.data['tfIdf'].tolist()
    vocabulary = model.preprocessor.getVocabDict()

    
    clf = KMeans(n_clusters=nrCluster, init='k-means++', max_iter=100)
    clf.fit(features)

    order_centroids = clf.cluster_centers_.argsort()[:, ::-1]
    for i in range(nrCluster):
        print("Cluster %d words:" % i)
        relevantWords = [vocabulary[ind] for ind in order_centroids[i, :20]]
        print(relevantWords)

    #pdb.set_trace()

    model.data['cluster'] = clf.labels_.tolist()
    SAcluster = model.data[model.data['Sexual.Assault.Manual']].cluster
    print np.histogram(SAcluster, bins=nrCluster)
        



if __name__ == "__main__":
    clustering_ICAAD()
