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
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim']
    target = targets[0]
    docPath = 'processedData/SADV'
    docPath = 'processedData/processedData'
    docPath = 'processedData/doc2vec'

    modelPath = 'doc2vecModels/test'
    selectedFeatures = 'tfIdf'
    
    model = ClassificationModel(path, target)
    model.classifierType='Doc2vec'
    
    if not model.existsProcessedData(docPath):
        print 'Preprocess Data'
        preprocessor = Preprocessor()
        model.data['cleanTokens'] = model.data.apply(lambda doc: preprocessor.cleanText(doc['text']).split(), axis=1)
        model.save(docPath)

    model = model.load(docPath)

    model.targetFeature = target
    model.createTarget()
    model.splitDataset(3, random=True)
    nrDocs = len(model.data)
   
    if not os.path.exists(modelPath):
        print 'Build Doc2Vec Model'
        #trainDocs = [TaggedDocument(row['cleanTokens'], [row[target]]) for index, row in model.trainData.iterrows()]
        trainDocs = [TaggedDocument(row['cleanTokens'], ['Doc '+str(index)]) for index, row in model.trainData.iterrows()]
        
        print 'Train Classifier'
        pdb.set_trace()
        doc2vecModel = Doc2Vec(size=50, min_count=2, iter=55)
        doc2vecModel.build_vocab(trainDocs)
        doc2vecModel.train(trainDocs) 
        
        doc2vecModel.save(modelPath)
    
    doc2vecModel = Doc2Vec.load(modelPath)
    

    print 'Infere training Vectors'
    predictedValues = []
    for index, row in model.testData.iterrows():
        vector = doc2vecModel.infer_vector(row['text'].lower().split())
        probsa, probsb =  doc2vecModel.docvecs.most_similar([vector], topn=10) 
        model.testData.loc[index, 'predictedLabel'] = probsb[0]
        if probsa[1]>probsb[1]:
            model.testData.loc[index, 'predictedLabel'] = probsa[0]


    pdb.set_trace()


    print 'Evaluation'
    model.evaluate()
    model.evaluation.confusionMatrix()

    
    
    viewer = Viewer(model)
    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability']
    viewer.printDocuments(model.testData, displayFeatures)
    viewer.classificationResults(model)


if __name__=='__main__':
    classification_doc2vec()

