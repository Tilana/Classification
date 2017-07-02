from lda import Viewer, ClassificationModel, FeatureExtractor
from sklearn.model_selection import KFold
import pandas as pd
import pdb

def validateModel(model, features):

    nrDocs = len(model.data)
    model.splitDataset(2*nrDocs/3, random=False)
    #model.validationSet()

    print 'Train Classifier'
    model.trainClassifier(features)

    print 'Evaluation'
    model.predict(features)
    model.evaluate()
    model.evaluation.confusionMatrix(model.targetLabels)
    #model.evaluation.classificationReport(model.targetLabels)
    try:
        model.relevantFeatures()
    except:
        pass

    print 'Display Results'
    viewer = Viewer(model.name + '/' + model.classifierType)
    #pdb.set_trace()
    #displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age']
    displayFeatures = ['predictedLabel', 'probability', 'tag', 'Year', 'entities', 'DocType', 'Type1', 'Type2', 'agenda', 'is_last', 'order', 'favour_count', 'agains_count']
    viewer.printDocuments(model.testData, displayFeatures)
    viewer.classificationResults(model, normalized=False)


if __name__=='__main__':
    validateModel()

