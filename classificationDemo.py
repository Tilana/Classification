from lda import Viewer, ClassificationModel, FeatureExtractor
from sklearn.cross_validation import KFold
import pandas as pd
import pdb

def classificationDemo():

    path = 'Documents/ICAAD/ICAAD.pkl'
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim', 'SGBV']
    target = targets[4]
    modelPath = 'processedData/SADV'
    modelPath = 'processedData/processedData'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'LogisticRegression']
    classifierType = classifierTypes[3]
    alpha = 0.01 
    selectedFeatures = 'tfIdf'
    
    
    model = ClassificationModel(path, target)
    #model.data = model.data[model.data['Sexual.Assault.Manual'] | model.data['Domestic.Violence.Manual']]
    
    if not model.existsProcessedData(modelPath):
        print 'Preprocess Data'
        model.buildPreprocessor(ngram_range=(1,2), min_df=5, max_df=0.50, max_features=8000)
        model.trainPreprocessor()
        model.data.reset_index(inplace=True)
        model.save(modelPath)

    model = model.load(modelPath)
    model.data['SGBV'] = model.data['Sexual.Assault.Manual'] | model.data['Domestic.Violence.Manual']
    #pdb.set_trace()

    model.targetFeature = target
    model.createTarget()

    model.splitDataset(5000, random=True)
    nrDocs = len(model.data)
   

    print 'Train Classifier'
    model.buildClassifier(classifierType, alpha=alpha) 
    model.trainClassifier(selectedFeatures)

    print 'Evaluation'
    model.predict(selectedFeatures)
    model.evaluate()
    model.evaluation.confusionMatrix()

    model.relevantFeatures()
    #pdb.set_trace()

    viewer = Viewer(classifierType)
    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability']
    viewer.printDocuments(model.testData, displayFeatures)
    viewer.classificationResults(model)
    pdb.set_trace()


if __name__=='__main__':
    classificationDemo()

