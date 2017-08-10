from lda import Collection
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


def Doc2VecModel():

    name = 'RightDocs'
    dataPath = 'processedData/' + name #RightDocs_topics'
    
    collection = Collection().load(dataPath)
    collection.removeStopwords()

    documents = [TaggedDocument(row['cleanTokens'], ['Doc '+str(index)]) for index, row in collection.data.iterrows()]
    doc2vecModel = Doc2Vec(documents, size=100, min_count=5, iter=55, window=8)
    doc2vecModel.save('doc2vecModels/'+name)

    collection.data['docVec'] = doc2vecModel.docvecs
    collection.save(dataPath+'_doc2vec')


if __name__=='__main__':
    Doc2VecModel()

