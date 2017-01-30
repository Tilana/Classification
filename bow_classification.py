import pandas as pd
from lda import Collection, Info, docLoader, Dictionary
from lda import ClassificationModel, Viewer
import bnpy_run
from nltk.corpus import names
from lda import dataframeUtils as df
import logging
from sklearn.feature_extraction.text import CountVectorizer

info = Info()

info.data = 'ICAAD'
info.startDoc = 0 
info.numberDoc= None 
info.includeEntities = 0
info.whiteList= 1 
info.removeNames = 1 

info.stoplist = [x.strip() for x in open('stopwords/english.txt')]
info.lowerFilter = 8
info.upperFilter = 0.3

keywordFile = 'Documents/ICAAD/CategoryLists.csv'
keywords_df = pd.read_csv(keywordFile).astype(str)
keywords = list(df.toListMultiColumns(keywords_df, keywords_df.columns))

info.setCollectionName()

collection = Collection()
collection.loadPreprocessedCollection(info.collectionName)

dictionary = Dictionary(info.stoplist)
dictionary.addCollection(collection.documents)
dictionary.filter_extremes(info.lowerFilter, info.upperFilter, keywords)

corpus = collection.createCorpus(dictionary)


pd.option('chained_assignment', None)
evaluationFile = 'Documents/PACI.csv'
dataFeatures = pd.read_csv(evaluationFile)
dataFeatures = dataFeatures.rename(columns = {'Unnamed: 0': 'id'})

features = ['Domestic.Violence.Manual', 'Sexual.Assault.Manual']


#vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2), vocabulary = dictionary.ids.token2id, binary=False, max_features=8000, stop_words = info.stoplist, max_df = 0.3, min_df = 8)
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,2), binary=False, max_features=8000, stop_words = info.stoplist, max_df = 0.28, min_df = 8)

text = [doc.text for doc in collection.documents]

word_counts = vectorizer.fit_transform(text)
vocab = vectorizer.get_feature_names()

model = ClassificationModel()
model.data = pd.DataFrame(word_counts.toarray(), columns = vocab)

ids = [doc.id for doc in collection.documents]
model.data['id'] = ids

for feature in features:
    model.targetFeature = feature

    column = dataFeatures[['id', model.targetFeature]]
    model.data = pd.merge(model.data, column, on=['id'])
    model.data = model.data.set_index('Unnamed: 0')

    model.dropNANRows()
