from lda import Viewer, ClassificationModel, FeatureExtractor
from sklearn.cross_validation import KFold
import pandas as pd
import pdb
from gensim.models import Doc2Vec
from collections import namedtuple
from gensim.models.doc2vec import TaggedDocument

def classification_doc2vec():

    path = 'Documents/ICAAD/ICAAD.pkl'
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim']
    target = targets[0]
    modelPath = 'processedData/SADV'
    modelPath = 'processedData/processedData'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'LogisticRegression']
    classifierType = classifierTypes[1]
    alpha = 0.01 
    selectedFeatures = 'tfIdf'
    
    
    model = ClassificationModel(path, target)
    model.data = model.data[model.data['Sexual.Assault.Manual'] | model.data['Domestic.Violence.Manual']]
    
    if not model.existsProcessedData(modelPath):
        print 'Preprocess Data'
        model.buildPreprocessor(ngram_range=(1,2), min_df=5, max_df=0.50, max_features=8000)
        model.trainPreprocessor()
        model.data.reset_index(inplace=True)
        model.save(modelPath)

    model = model.load(modelPath)


    #extractor = FeatureExtractor()
    #model.data['Type'] = model.data.apply(lambda doc: extractor.caseType(doc.text), axis=1)
    #model.data = model.data[model.data.Type=='SENTENCE']
    #model.data.reset_index(inplace=True)

    
    model.targetFeature = target
    model.createTarget()
    model.splitDataset(6000, random=True)
    nrDocs = len(model.data)
   
    results = pd.DataFrame()


    trainDocs = []
    for index, row in model.trainData.iterrows(): 
        trainDocs.append(TaggedDocument(row['text'].lower().split(), [row[target]]))

    testDocs = []
    for index, row in model.testData.iterrows(): 
        testDocs.append(row['text'].lower().split())
    

    #pdb.set_trace()

    print 'Train Classifier'
    doc2vecModel = Doc2Vec(size=50, min_count=2, iter=55)
    doc2vecModel.build_vocab(trainDocs)
    doc2vecModel.train(trainDocs) 

    pdb.set_trace()


    print 'Infere training Vectors'
    ranks = []
    secondRanks = []
    for doc in testDocs:
        vector = doc2vecModel.infer_vector(doc)
        model.docvecs.most_similar([vector], topn=10)











    model.buildClassifier(classifierType, alpha=alpha) 
    model.trainClassifier(selectedFeatures)

    print 'Evaluation'
    model.predict(selectedFeatures)
    model.evaluate()
    model.evaluation.confusionMatrix()

    results['Fold '+ str(foldNr)] = model.evaluation.toSeries()
    
    
    print 'Display Results'
    results.index=['accuracy','precision', 'recall']
    print results
    
    viewer = Viewer(classifierType)
    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability']
    viewer.printDocuments(model.testData, displayFeatures)
    viewer.classificationResults(model)


if __name__=='__main__':
    classification_doc2vec()

