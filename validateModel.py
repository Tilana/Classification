from lda import Viewer, ClassificationModel, FeatureExtractor
from sklearn.model_selection import KFold
import pandas as pd
import pdb

def validateModel(model, features):

    #pdb.set_trace()

    nrDocs = len(model.data)
    model.splitDataset(train_size=1*nrDocs/3)
    #model.validationSet()

    print 'Train Classifier'
    model.trainClassifier(features)

    print 'Evaluation'
    #pdb.set_trace()
    model.predict(features)
    model.evaluate()
    #model.evaluation.classificationReport(model.targetLabels)
    try:
        model.evaluation.confusionMatrix(model.targetLabels)
        model.relevantFeatures()
    except:
        pass



if __name__=='__main__':
    validateModel()

