from lda import Viewer, ClassificationModel, FeatureExtractor
from sklearn.model_selection import KFold
import pandas as pd
import pdb

def validateModel(model, features):

    #pdb.set_trace()

    nrDocs = len(model.data)
    model.splitDataset(2*nrDocs/3, random=True)
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



if __name__=='__main__':
    validateModel()

