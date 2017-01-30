import pandas as pd
from sklearn import cluster 
from sklearn import preprocessing
from lda import Collection, Cluster, Info, Viewer
from lda import dataframeUtils as df

def clustering():

    collectionName = 'processedDocuments/ICAAD_noEntities_word2vec_noNames'
    collection = Collection()
    collection.loadPreprocessedCollection(collectionName)

    info = Info()
    info.data = 'ICAAD'
    info.identifier = 'LDA_T60P10I70_tfidf_word2vec'

    path = 'html/%s/DocumentFeatures.csv' % (info.data + '_' + info.identifier)
    originalData = pd.read_csv(path)

    data = originalData.drop(['File', 'id', 'SA', 'DV', 'Unnamed: 0'], 1)

    #data = originalData.drop(['File', 'id', 'SA', 'DV', 'Unnamed: 0','relevantWord1', 'relevantWord2', 'relevantWord3'], 1) # 'similarDocs1', 'similarDocs2', 'similarDocs3', 'similarDocs4', 'similarDocs5'], 1) #, 'relevantWord1', 'relevantWord2', 'relevantWord3', 'final sentence', 'knife', 'hit', 'dead'], 1)
    data = data.dropna()
    df.createNumericFeature(data, 'court')

    scaledMatrix = preprocessing.scale(data)
    scaledData = pd.DataFrame(scaledMatrix, columns = data.columns.tolist())

    print 'Clustering'
    numCluster = 8 
    km = cluster.KMeans(n_clusters = numCluster, n_init=20, max_iter=500)
    #km = cluster.AgglomerativeClustering(n_clusters = numCluster, affinity='cosine', linkage="average") 


    km.fit(scaledData)
    clusters = km.labels_.tolist()

    data['clusters'] = clusters
    html = Viewer(info)

    clusterData = []
    numberDocs = []
    numberSA = []
    numberDV = []

    print 'Create Features of Clusters'
    for clusterNo in range(0, numCluster):
        currCluster = Cluster(clusterNo)
        currCluster.features = data[data['clusters']==clusterNo]
        indices = currCluster.features.index.tolist()
        currCluster.documents = [(collection.documents[ind], ind) for ind in indices]
        currCluster.createBinaryFeature('SA')
        currCluster.createBinaryFeature('DV')
        clusterData.append(currCluster)
        numberDocs.append(len(currCluster.features))
        numberSA.append(len(currCluster.SATrue))
        numberDV.append(len(currCluster.DVTrue))
        html.printCluster(currCluster)
    
    html.printClusterOverview(numberDocs, numberSA, numberDV)



if __name__ == "__main__":
    clustering()

