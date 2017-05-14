from lda import Viewer, ClassificationModel, FeatureExtractor
from sklearn.cross_validation import KFold
import pandas as pd
import pdb

def buildClassificationModel():

    path = 'Documents/ICAAD/ICAAD.pkl'
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim']
    target = targets[0]
    modelPath = 'processedData/SADV'
    modelPath = 'processedData/processedData'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'LogisticRegression']
    classifierType = classifierTypes[-1]
    alpha = 0.01 
    selectedFeatures = 'tfIdf'
    
    
    model = ClassificationModel(path, target)
    SAcases = model.data['Sexual.Assault.Manual']
    DVcases = model.data['Domestic.Violence.Manual']
    SADVcases = model.data[SAcases | DVcases]

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

    #pdb.set_trace()
    
    model.targetFeature = target
    model.createTarget()
    nrDocs = len(model.data)
   
    results = pd.DataFrame()

    for foldNr, (trainInd, testInd) in enumerate(KFold(nrDocs, n_folds=2, shuffle=True)):
        model.trainIndices = trainInd
        model.testIndices = testInd
        model.split()

        print 'Train Classifier'
        model.buildClassifier(classifierType) 
        model.trainClassifier(selectedFeatures)
        #pdb.set_trace()

        print 'Evaluation'
        model.predict(selectedFeatures)
        model.evaluate()
        model.evaluation.confusionMatrix(model.targetLabels)
        #pdb.set_trace()

        results['Fold '+ str(foldNr)] = model.evaluation.toSeries()
        
    
    print 'Display Results'
    results.index=['accuracy','precision', 'recall']
    print results
    
    viewer = Viewer(classifierType)
    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age']
    viewer.printDocuments(model.testData, displayFeatures)
    viewer.classificationResults(model)


if __name__=='__main__':
    buildClassificationModel()

