from lda import Viewer, ClassificationModel, FeatureExtractor
import pandas as pd
import pdb

def modelSelection():

    path = 'Documents/ICAAD/ICAAD.pkl'
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim', 'SGBV']
    target = targets[1]
    #modelPath = 'processedData/SADV'
    modelPath = 'processedData/processedData'

    classifierTypes = ['MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'LogisticRegression', 'kNN', 'DecisionTree']
    selectedFeatures = 'tfIdf'
    
    model = ClassificationModel(path, target)
    
    if not model.existsProcessedData(modelPath):
        print 'Preprocess Data'
        model.buildPreprocessor(ngram_range=(1,2), min_df=5, max_df=0.50, max_features=8000)
        model.trainPreprocessor()
        model.data.reset_index(inplace=True)
        model.save(modelPath)


    model = model.load(modelPath)
    model.targetFeature = target
    model.createTarget()
    
#    model.balanceDataset(factor=2)
    model.splitDataset(6000, random=True)
    #model.validationSet()
    nrDocs = len(model.data)

    #pdb.set_trace()

    results = pd.DataFrame(classifierTypes)

    for classifierType in classifierTypes:
        print classifierType

        model.buildClassifier(classifierType) 
        weightedFscore = model.weightFScore(2)
        (bestScore, params) = model.gridSearch(selectedFeatures, scoring=weightedFscore)
        print('Best score: %0.3f' % bestScore)
        print params

        model.predict(selectedFeatures)
        model.evaluate()
        #model.validate(selectedFeatures)
        print 'Accuraccy: {:f}'.format(model.evaluation.accuracy)
        print 'Precision: {:f}'.format(model.evaluation.precision)
        print 'Recall: {:f}'.format(model.evaluation.recall)
    
    
    pdb.set_trace()

    print 'Evaluation'
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
    modelSelection()

