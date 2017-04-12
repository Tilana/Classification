from lda import Viewer, ClassificationModel, FeatureExtractor
from sklearn.cross_validation import KFold
import pandas as pd
import pdb

def classificationDemo():

    path = 'Documents/ICAAD/ICAAD.pkl'
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim', 'SGBV']
    target = targets[1]
    #modelPath = 'processedData/SADV'
    modelPath = 'processedData/processedData'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'LogisticRegression']
    classifierType = classifierTypes[2]
    alpha = 0.01 
    alphaRange = [0.0001, 0.01, 0.1, 0.3, 0.5, 0.9]
    selectedFeatures = 'tfIdf'
    
    model = ClassificationModel(path, target)
    
    if not model.existsProcessedData(modelPath):
        print 'Preprocess Data'
        model.buildPreprocessor(ngram_range=(1,2), min_df=5, max_df=0.50, max_features=8000)
        model.trainPreprocessor()
        model.data.reset_index(inplace=True)
        model.save(modelPath)

    model = model.load(modelPath)
    #pdb.set_trace()

    model.targetFeature = target
    model.createTarget()
    
    model.balanceDataset(factor=2)
    model.splitDataset(4500, random=False)
    model.validationSet()
    nrDocs = len(model.data)

    for alpha in alphaRange:
        print alpha

        print 'Train Classifier'
        model.buildClassifier(classifierType, alpha=alpha) 
        model.trainClassifier(selectedFeatures)

        print 'Validation'
        model.validate(selectedFeatures)
        print 'Accuraccy: {:f}'.format(model.validation.accuracy)
        print 'Precision: {:f}'.format(model.validation.precision)
        print 'Recall: {:f}'.format(model.validation.recall)
    
    
    pdb.set_trace()

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

