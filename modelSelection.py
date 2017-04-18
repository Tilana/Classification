from lda import Viewer, ClassificationModel, FeatureExtractor
import pandas as pd
import pdb

def modelSelection():

    path = 'Documents/ICAAD/ICAAD.pkl'
    targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim', 'SGBV']
    target = targets[0]
    #modelPath = 'processedData/SADV'
    modelPath = 'processedData/processedData'
    resultPath = 'modelSelection/SA_3cv_6000docs.csv'

    classifierTypes = ['LogisticRegression', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'SVM', 'kNN', 'DecisionTree']
    selectedFeatures = 'tfIdf'
    
    model = ClassificationModel(path, target)
    results = pd.DataFrame(columns=classifierTypes, index=['Best score', 'params', 'Test Accuracy', 'Test Precision', 'Test Recall'])
    
    if not model.existsProcessedData(modelPath):
        print 'Preprocess Data'
        model.buildPreprocessor(ngram_range=(1,2), min_df=5, max_df=0.50, max_features=8000)
        model.trainPreprocessor()
        model.data.reset_index(inplace=True)
        model.save(modelPath)


    model = model.load(modelPath)
    model.targetFeature = target
    model.createTarget()
    
    model.splitDataset(6000, random=True)
    nrDocs = len(model.data)

    for classifierType in classifierTypes:
        print classifierType

        model.buildClassifier(classifierType) 
        weightedFscore = model.weightFScore(2)
        (bestScore, params) = model.gridSearch(selectedFeatures, scoring=weightedFscore)
        print('Best score: %0.3f' % bestScore)
        print params

        model.predict(selectedFeatures)
        model.evaluate()
        print 'Accuraccy: {:f}'.format(model.evaluation.accuracy)
        print 'Precision: {:f}'.format(model.evaluation.precision)
        print 'Recall: {:f}'.format(model.evaluation.recall)

        results[classifierType] = [bestScore, params, model.evaluation.accuracy, model.evaluation.precision, model.evaluation.recall]
    
    
    print results
    results.to_csv(resultPath)

    #print 'Evaluation'
    #model.evaluate()
    #model.evaluation.confusionMatrix()

    #model.relevantFeatures()

    #viewer = Viewer(classifierType)
    #displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability']
    #viewer.printDocuments(model.testData, displayFeatures)
    #viewer.classificationResults(model)
    #pdb.set_trace()


if __name__=='__main__':
    modelSelection()

