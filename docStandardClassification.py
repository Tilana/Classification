#! /usr/bin/env python
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import data_helpers
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from lda import ClassificationModel, Viewer, NeuralNet, Collection
from data_helpers import splitInSentences
import pdb
from tensorflow.python import debug as tf_debug
from modelSelection import modelSelection
from validateModel import validateModel


def textToSentenceData(data):
    sentences = splitInSentences(data, TARGET)
    return pd.DataFrame(sentences, columns=['id', TARGET, 'sentences'])

def sentenceLength(sentence):
    return len(sentence.split())

def loadProcessor(directory):
    vocab_path = os.path.join(directory, "vocab")
    return learn.preprocessing.VocabularyProcessor.restore(vocab_path)

def padDocuments(doc, nrSentences, cols=['0','1','2','3']):
    #pdb.set_trace()
    pads = np.zeros((nrSentences, len(cols)))
    activation = doc[cols].as_matrix()
    pads[:len(activation)] = activation[:nrSentences]
    return pads

def highestActivation(data, col=0, thresh=100):
    pads = np.zeros((thresh, 1))
    sortedData = data.sort_values(col, ascending=False)
    data = sortedData[[0]].as_matrix()
    pads[:len(sortedData)] = data[:thresh]
    return pads

def getTargetValue(data, target):
    return data[target].tolist()[0]



PATH = '../data/'
DATASET = 'ICAAD'
ID = 'SA'
ID = 'DV'
TARGET = 'Sexual.Assault.Manual'
TARGET = 'Domestic.Violence.Manual'
MODEL_PATH = './runs/' + DATASET + '_' + ID + '/'
BATCH_SIZE = 50
ITERATIONS = 900
multilayer = 1

def docClassification():

    data_path = os.path.join(PATH, DATASET, DATASET + '.pkl')
    data = pd.read_pickle(data_path)
    data.set_index('id', inplace=True, drop=False)

    model = ClassificationModel()
    model.data = data
    model.targetFeature = TARGET
    model.target = data[TARGET]
    model.classificationType = 'binary'

    indices = pd.read_csv(MODEL_PATH + 'trainTest_split.csv', index_col=0)


    train_path = PATH + DATASET + '/' + ID + '_sentence_activation_train.csv'
    test_path = PATH + DATASET + '/' + ID + '_sentence_activation_test.csv'

    trainSentences = pd.read_csv(train_path, index_col=0)
    testSentences = pd.read_csv(test_path, index_col=0)
    trainSentences.sort_values('id', inplace=True)

    trainSentences['maxActivation'] = trainSentences[['0', '1']].max(axis=1)
    testSentences['maxActivation'] = testSentences[['0', '1']].max(axis=1)


    trainSentences['prediction'] = trainSentences['3'] > trainSentences['2']
    #trainSentences = trainSentences[trainSentences['prediction']==1]

    testSentences['prediction'] = testSentences['3'] > testSentences['2']
    #testSentences = testSentences[testSentences['prediction']==1]
    testSentences.sort_values('id', inplace=True)
    #testSentences.sort_values('maxActivation', inplace=True)
    #trainSentences.sort_values('maxActivation', inplace=True)

    train_docs = trainSentences.groupby('id')
    test_docs = testSentences.groupby('id')

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

    #nn = NeuralNet(maxNumberEvidenceSentences*nrFeature, nrClasses)


    #pdb.set_trace()

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

    #pdb.set_trace()


    model.evaluate()
    model.evaluation.confusionMatrix()
    model.classifierType = 'CNN Docs'

    viewer = Viewer('DocsCNN', 'test')

    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age', 'evidence']
    viewer.printDocuments(model.testData, displayFeatures, TARGET)
    viewer.classificationResults(model, normalized=False)


    pdb.set_trace()



if __name__=='__main__':
    docClassification()

