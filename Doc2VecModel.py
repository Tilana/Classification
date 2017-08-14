from lda import Collection
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import pandas as pd
import pickle
import pdb


def Doc2VecModel():

    dataFrame = pd.DataFrame()
    collection = Collection()

    dataPath = 'processedData/ICAAD'
    reader = pd.read_csv(dataPath + '.csv', chunksize=100)
    
    for num, chunk in enumerate(reader):
        print 'Chunk %i' % num
        collection.data = chunk
        collection.cleanTexts()
        collection.removeStopwords()
        dataFrame = pd.concat([dataFrame, collection.data])

    pdb.set_trace()

    dataFrame.to_csv(dataPath + '_cleanTokens.csv', encoding='utf8')
    collection.data = dataFrame
    collection.save(dataPath+'_doc2vec')


    print 'Compute Doc2Vec Model'
    documents = [TaggedDocument(row['cleanTokens'], ['Doc '+str(index)]) for index, row in collection.data.iterrows()]
    doc2vecModel = Doc2Vec(documents, size=100, min_count=5, iter=55, window=8)
    doc2vecModel.save('doc2vecModels/'+name)

    print 'Save Model'
    collection.data['docVec'] = doc2vecModel.docvecs
    collection.save(dataPath+'_doc2vec')


if __name__=='__main__':
    Doc2VecModel()

