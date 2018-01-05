#! /usr/bin/env python
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath('../'))
from lda import ClassificationModel, Viewer, Collection, data_helpers

def padDocuments(doc, nrSentences, cols=['0','1','2','3']):
    pads = np.zeros((nrSentences, len(cols)))
    activation = doc[cols].as_matrix()
    pads[:len(activation)] = activation[:nrSentences]
    return pads

def getTargetValue(data, target):
    return data[target].tolist()[0]



PATH = '../../data/'
DATASET = 'ICAAD'
ID = 'SA'
ID = 'DV'
TARGET = 'Sexual.Assault.Manual'
TARGET = 'Domestic.Violence.Manual'
MODEL_PATH = '../runs/' + DATASET + '_' + ID + '/'
BATCH_SIZE = 50
ITERATIONS = 900
multilayer = 1
storeEvidenceText = True

def docStandardClassification():

    data_path = os.path.join(PATH, DATASET, DATASET + '.pkl')
    data = pd.read_pickle(data_path)
    data.set_index('id', inplace=True, drop=False)

    model = ClassificationModel()
    model.data = data
    model.targetFeature = TARGET
    model.target = data[TARGET]
    model.classificationType = 'binary'
    model.validation = False

    indices = pd.read_csv(MODEL_PATH + 'trainTest_split.csv', index_col=0)


    train_path = PATH + DATASET + '/' + ID + '_sentence_activation_train.csv'
    test_path = PATH + DATASET + '/' + ID + '_sentence_activation_test.csv'

    trainSentences = pd.read_csv(train_path, index_col=0)
    testSentences = pd.read_csv(test_path, index_col=0)
    trainSentences.sort_values('id', inplace=True)

    trainSentences['maxActivation'] = trainSentences[['0', '1']].max(axis=1)
    testSentences['maxActivation'] = testSentences[['0', '1']].max(axis=1)


    trainSentences['prediction'] = trainSentences['3'] > trainSentences['2']
    trainSentences = trainSentences[trainSentences['prediction']==1]

    testSentences['prediction'] = testSentences['3'] > testSentences['2']
    testSentences = testSentences[testSentences['prediction']==1]
    testSentences.sort_values('id', inplace=True)
    #testSentences.sort_values('maxActivation', inplace=True)
    #trainSentences.sort_values('maxActivation', inplace=True)

    train_docs = trainSentences.groupby('id')
    test_docs = testSentences.groupby('id')

    # combine evidence sentences
    if storeEvidenceText:
        evidenceText = train_docs.sentences.apply(' '.join)
        evidenceText = evidenceText.append(test_docs.sentences.apply(' '.join))
        evidenceText = pd.DataFrame(evidenceText.rename('evidenceText_'+ID))

        data = data.merge(evidenceText, how='left', left_on='id', right_index=True)
        path_evidenceText = os.path.join(PATH, DATASET, DATASET + '_evidenceSummary.pkl')
        data.to_pickle(path_evidenceText)

    maxNumberEvidenceSentences = max(train_docs.apply(len))
    #maxNumberEvidenceSentences = 100
    print 'Maximum Number of Evidence Sentences ' + str(maxNumberEvidenceSentences)
    nrClasses = len(trainSentences[TARGET].unique())

    cols = ['0', '1']
    #cols = ['2', '3']
    cols = ['0', '1', '2', '3']
    cols = ['maxActivation']
    nrFeatures = len(cols)

    X_pretrain = train_docs.apply(padDocuments, maxNumberEvidenceSentences, cols)
    X_pretrain = X_pretrain.apply(lambda x: x.reshape(maxNumberEvidenceSentences*nrFeatures))
    X_train= X_pretrain.apply(lambda x: pd.Series(x))

    X_pretest = test_docs.apply(padDocuments, maxNumberEvidenceSentences, cols)
    X_pretest = X_pretest.apply(lambda x: x.reshape(maxNumberEvidenceSentences*nrFeatures))
    X_test = X_pretest.apply(lambda x: pd.Series(x))

    Y_train = pd.get_dummies(train_docs.apply(getTargetValue, TARGET)).as_matrix()
    Y_test = pd.get_dummies(test_docs.apply(getTargetValue, TARGET)).as_matrix()

    train_ids = sorted(train_docs.indices.keys())
    test_ids = sorted(test_docs.indices.keys())
    model.trainIndices = train_ids
    model.testIndices = test_ids
    model.split()

    model.testTarget = test_docs.apply(getTargetValue, TARGET).tolist()

    model.buildClassifier('LogisticRegression')
    #model.buildClassifier('SVM')
    #model.buildClassifier('DecisionTree')
    #model.buildClassifier('RandomForest')
    model.features = X_train.columns
    model.testData = model.testData.merge(X_test, left_index=True, right_index=True)
    model.trainData = X_train
    #model.testData = X_test
    #model.testData['id'] = model.testData.index

    model.whitelist = None
    (score, params)  = model.gridSearch(model.features, scaling=False)
    model.predict(model.features)

    #predictedLabels = sess.run(nn.Y, feed_dict=testData)
    #model.testData['predictedLabel'] = np.argmax(predictedLabels, 1)
    #model.testData['probability'] = np.max(predictedLabels, 1)
    #test_accuracy = sess.run(nn.accuracy, feed_dict=testData)
    #print 'Test Accuracy: ' + str(test_accuracy)

    model.evaluate()
    model.evaluation.confusionMatrix()
    model.classifierType = 'CNN Docs'

    viewer = Viewer('DocsCNN', folder='test', prefix='..')

    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age', 'evidence']
    viewer.printDocuments(model.testData, displayFeatures, TARGET)
    viewer.classificationResults(model, normalized=False)

if __name__=='__main__':
    docStandardClassification()

