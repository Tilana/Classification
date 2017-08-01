from lda import Viewer, ClassificationModel, FeatureExtractor
from lda.dataframeUtils import toCSV
import pandas as pd
import pdb

classifierTypes = ['LogisticRegression', 'BernoulliNB', 'RandomForest', 'DecisionTree'] #, 'SVM', 'kNN']
#classifierTypes = ['kNN', 'DecisionTree']
classifierTypes = ['LogisticRegression', 'kNN', 'DecisionTree', 'BernoulliNB']
classifierTypes = ['DecisionTree']


def createResultPath(name, target,  **args):
    path = 'modelSelection/'
    target = target.replace('.','')
    path = path+name+'_'+target+'.csv'
    return path


def modelSelection(collection, target, features, nrTrainingDocs=None, whitelist=None):

    pca=False
    pcaComponents = 130 
    resultPath = createResultPath(collection.name, target)

    model = ClassificationModel()
    model.data = collection.data

    nrTrainingDocs = nrTrainingDocs
    if not nrTrainingDocs:
        nrTrainingDocs = len(model.data)/100*70
    
    results = pd.DataFrame(columns=classifierTypes, index=['Best score', 'params', 'Test Accuracy', 'Test Precision', 'Test Recall'])
    
    model.targetFeature = target
    model.features = features
    model.whitelist = whitelist
    model.createTarget()

    model.splitDataset(nrTrainingDocs, random=False)
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
            bestClassifier = classifierType
            bestScore=score
            bestParams = params

    bestModel = model
    bestModel.buildClassifier(bestClassifier, bestParams)
    bestModel.name = collection.name
    #toCSV(results, resultPath)

    return bestModel 


if __name__=='__main__':
    modelSelection()

