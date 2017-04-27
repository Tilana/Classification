from lda import Viewer, ClassificationModel, FeatureExtractor
from sklearn.cross_validation import KFold
import pandas as pd
import pdb
from gensim.models import Doc2Vec
from lda import Preprocessor
from collections import namedtuple
from gensim.models.doc2vec import TaggedDocument
import os

def classification_doc2vec():

    path = 'Documents/ICAAD/ICAAD.pkl'
    docPath = 'processedData/doc2vec'
    modelPath = 'doc2vecModels/allDocs'
    model = ClassificationModel(path)
    
    if not model.existsProcessedData(docPath):
        print 'Preprocess Data'
        preprocessor = Preprocessor()
        model.data['cleanTokens'] = model.data.apply(lambda doc: preprocessor.cleanText(doc['text']).split(), axis=1)
        model.save(docPath)

    model = model.load(docPath)

    if not os.path.exists(modelPath):
        print 'Build Doc2Vec Model'
        #trainDocs = [TaggedDocument(row['cleanTokens'], [row[target]]) for index, row in model.trainData.iterrows()]
        trainDocs = [TaggedDocument(row['cleanTokens'], ['Doc '+str(index)]) for index, row in model.data.iterrows()]
        
        doc2vecModel = Doc2Vec(trainDocs, size=100, min_count=5, iter=55, window=8)
        doc2vecModel.save(modelPath)
        model.data['docVec'] = doc2vecModel.docvecs
        model.save(docPath)
    
    doc2vecModel = Doc2Vec.load(modelPath)
    
    print 'Test Doc2Vec Model'
    ranks = []
    for ind, row in model.data.iterrows():
        vector = doc2vecModel.infer_vector(row['cleanTokens'])
        similarDocs =  doc2vecModel.docvecs.most_similar([vector], topn=10) 
        rank = [docid for docid, sim in similarDocs].index(ind)
        ranks.append(rank)

    pdb.set_trace()


if __name__=='__main__':
    classification_doc2vec()

