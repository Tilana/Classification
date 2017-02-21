from lda import Viewer, ClassificationModel
from sklearn.cross_validation import KFold
import pandas as pd
import pdb

def buildClassificationModel():

    path = 'Documents/ICAAD/ICAAD.pkl'
    target = 'Sexual.Assault.Manual'
    target = 'Domestic.Violence.Manual'
    target = 'Age'
    modelPath = 'processedData/processedData'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'LogisticRegression']
    classifierType = classifierTypes[1]
    alpha = 0.8 
    selectedFeatures = 'tfIdf'
    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag']

    model = ClassificationModel(path, target)

    if not model.existsProcessedData(modelPath):
        print 'Preprocess Data'
        model.buildPreprocessor(ngram_range=(1,2), min_df=10, max_df=0.5, max_features=8000)
        model.trainPreprocessor()
        model.save(modelPath)

    model = model.load(modelPath)
    model.data.reset_index(inplace=True)

    model.targetFeature = target
    model.createTarget()
    nrDocs = len(model.data)
   
    results = pd.DataFrame()

    for foldNr, (trainInd, testInd) in enumerate(KFold(nrDocs, n_folds=4, shuffle=True)):
        model.trainIndices = trainInd
        model.testIndices = testInd
        model.split()

        #pdb.set_trace() 

        print 'Train Classifier'
        model.buildClassifier(classifierType, alpha=alpha) 
        model.trainClassifier(selectedFeatures)

        print 'Evaluation'
        model.predict(selectedFeatures)
        model.evaluate()
        model.evaluation.confusionMatrix()

        #pdb.set_trace()

        results['Fold '+ str(foldNr)] = model.evaluation.toSeries()
        
    
    print 'Display Results'
    results.index=['accuracy','precision', 'recall']
    print results
    #pdb.set_trace()
    
    viewer = Viewer(classifierType)
    viewer.printDocuments(model.testData, displayFeatures)
    viewer.classificationResults(model)


if __name__=='__main__':
    buildClassificationModel()

