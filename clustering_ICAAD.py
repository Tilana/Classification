import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def clustering_ICAAD():

    data = pd.read_pickle('Documents/ICAAD/ICAAD.pkl')
    texts = data['text'].get_values()

    stopwords = set([x.strip() for x in open("stopwords/english.txt")])

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,2), token_pattern='[a-zA-Z]+', max_df=1200, min_df=8, stop_words=stopwords, max_features=8000)
    
    
    word_counts = vectorizer.fit_transform(texts).toarray()
    #data['token'] = word_counts.tolist()

    vocabulary = vectorizer.get_feature_names()
    vocab_dict = dict(zip(range(0,len(vocabulary)), vocabulary))

    model = KMeans(n_clusters=40, init='k-means++', max_iter=100)
    model.fit(word_counts)

    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    for i in range(40):
        print("Cluster %d words:" % i)
        relevantWords = [vocab_dict[ind] for ind in order_centroids[i, :20]]
        print(relevantWords)

    data['cluster'] = model.labels_.tolist()

    SAcluster = data[data['Sexual.Assault.Manual']].cluster
    np.histogram(SAcluster, bins=40)
        



if __name__ == "__main__":
    clustering_ICAAD()
