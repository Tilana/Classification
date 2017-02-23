from lda import Viewer, ClassificationModel
from sklearn.cross_validation import KFold
import pandas as pd
import pdb

def buildClassificationModel():

    path = 'Documents/ICAAD/ICAAD.pkl'
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim']
    target = targets[1]
    modelPath = 'processedData/SADV'
    modelPath = 'processedData/processedData'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'LogisticRegression']
    classifierType = classifierTypes[3]
    alpha = 0.5 
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
    model.targetFeature = target
    model.createTarget()
    nrDocs = len(model.data)
   
    results = pd.DataFrame()

    for foldNr, (trainInd, testInd) in enumerate(KFold(nrDocs, n_folds=2, shuffle=True)):
        model.trainIndices = trainInd
        model.testIndices = testInd
        model.split()

        print 'Train Classifier'
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
    buildClassificationModel()

