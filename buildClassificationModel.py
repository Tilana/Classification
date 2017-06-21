from lda import Viewer, ClassificationModel, FeatureExtractor
from sklearn.model_selection import KFold
import pandas as pd
import pdb

def buildClassificationModel():

    path = 'Documents/ICAAD/ICAAD.pkl'
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim']
    target = targets[2]
    modelPath = 'processedData/SADV'
    #modelPath = 'processedData/processedData'

    classifierTypes = ['DecisionTree', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'LogisticRegression']
    classifierType = classifierTypes[0]
    alpha = 0.01 
    #selectedFeatures = ['tfIdf', 'Sexual.Assault.Manual']
    selectedFeatures = ['tfIdf']
    
    
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

    kFold = KFold(n_splits=2, shuffle=True)
    kFold.get_n_splits(model.data)

    for foldNr, (trainInd, testInd) in enumerate(kFold.split(model.data)):
        model.trainIndices = trainInd
        model.testIndices = testInd
        model.split()

        print 'Train Classifier'
        #pdb.set_trace()
        model.buildClassifier(classifierType) 
        model.trainClassifier(selectedFeatures)
        #pdb.set_trace()

        print 'Evaluation'
        model.predict(selectedFeatures)
        model.evaluate()
        model.evaluation.confusionMatrix(model.targetLabels)
        model.evaluation.classificationReport(model.targetLabels)

        pdb.set_trace()

        results['Fold '+ str(foldNr)] = model.evaluation.toSeries()
        
    
    print 'Display Results'
    results.index=['accuracy','precision', 'recall']
    print results
    #pdb.set_trace()
    
    viewer = Viewer(classifierType)
    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age']
    viewer.printDocuments(model.testData, displayFeatures)
    viewer.classificationResults(model, normalized=False)


if __name__=='__main__':
    buildClassificationModel()

