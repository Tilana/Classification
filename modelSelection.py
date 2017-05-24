from lda import Viewer, ClassificationModel, FeatureExtractor
from lda.dataframeUtils import toCSV
import pandas as pd
import pdb

def createResultPath(dataPath, target,  **args):
    path = 'modelSelection/'
    dataPath = dataPath.split('/')[1]
    target = target.replace('.','')
    #if pca:
    #    path = path + 'pca' + str(pcaComponents)+'.csv'    
    path = path+dataPath+'_'+target+'.csv'
    return path


def modelSelection(modelPath, target, features):

    pca=False
    pcaComponents = 130 
    resultPath = createResultPath(modelPath, target)

    classifierTypes = ['LogisticRegression', 'MultinomialNB', 'BernoulliNB', 'RandomForest', 'DecisionTree', 'SVM', 'kNN']
    classifierTypes = ['LogisticRegression', 'BernoulliNB', 'RandomForest', 'DecisionTree', 'SVM', 'kNN']
    classifierTypes = ['kNN', 'DecisionTree', 'RandomForest']
    #features = ['tfIdf']
    #features = ['docVec']
    nrTrainingDocs = 100 

    #pdb.set_trace()
    
    model = ClassificationModel()
    model = model.load(modelPath)
    
    results = pd.DataFrame(columns=classifierTypes, index=['Best score', 'params', 'Test Accuracy', 'Test Precision', 'Test Recall'])
    
    #model.data = model.data[model.data['Sexual.Assault.Manual'] | model.data['Domestic.Violence.Manual']]
    model.targetFeature = target
    model.createTarget()

    model.splitDataset(nrTrainingDocs, random=True)
    nrDocs = len(model.data)

    bestClassifier = None
    bestScore = 0
    bestParams = []

    for classifierType in classifierTypes:
        print classifierType

        model.buildClassifier(classifierType) 
        weightedFscore = model.weightFScore(2)
        (score, params) = model.gridSearch(features, scoring=weightedFscore, scaling=False, pca=pca, components=pcaComponents)
        print('Best score: %0.3f' % score)
        model.predict(features)
        model.evaluate()
        print 'Accuraccy: {:f}'.format(model.evaluation.accuracy)
        print 'Precision: {:f}'.format(model.evaluation.precision)
        print 'Recall: {:f}'.format(model.evaluation.recall)

        results[classifierType] = [score, params, model.evaluation.accuracy, model.evaluation.precision, model.evaluation.recall]

        if score > bestScore:
            bestClassifer = classifierType
            print bestClassifier
            bestScore=score
            bestParams = params

    pdb.set_trace()
    toCSV(results,resultPath)
    

    return (bestClassifier, bestParams, bestScore)


if __name__=='__main__':
    modelSelection()

