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
from lda import ClassificationModel, Viewer, NeuralNet
from data_helpers import splitInSentences
import pdb
from tensorflow.python import debug as tf_debug


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
    model.data.sort_values('id', inplace=True)
    model.targetFeature = TARGET
    model.target = data[TARGET]
    model.classificationType = 'binary'

    indices = pd.read_csv(MODEL_PATH + 'trainTest_split.csv', index_col=0)


    train_path = PATH + DATASET + '/' + ID + '_sentence_activation_train.csv'
    test_path = PATH + DATASET + '/' + ID + '_sentence_activation_test.csv'

    trainSentences = pd.read_csv(train_path, index_col=0)
    testSentences = pd.read_csv(test_path, index_col=0)
    trainSentences.sort_values('id', inplace=True)

    trainSentences['prediction'] = trainSentences['3'] > trainSentences['2']
    trainSentences = trainSentences[trainSentences['prediction']==1]

    testSentences['prediction'] = testSentences['3'] > testSentences['2']
    testSentences = testSentences[testSentences['prediction']==1]
    testSentences.sort_values('id', inplace=True)

    train_docs = trainSentences.groupby('id')
    test_docs = testSentences.groupby('id')

    maxNumberEvidenceSentences = max(train_docs.apply(len))
    #maxNumberEvidenceSentences = 200
    nrClasses = len(trainSentences[TARGET].unique())

    #pdb.set_trace()

    cols = ['0', '1']
    cols = ['2', '3']
    cols = ['0', '1', '2', '3']
    nrFeature = len(cols)

    X_train = np.array(train_docs.apply(padDocuments, maxNumberEvidenceSentences, cols).tolist())
    X_train = X_train.reshape(X_train.shape[0], maxNumberEvidenceSentences*nrFeature)

    X_test = np.array(test_docs.apply(padDocuments, maxNumberEvidenceSentences, cols).tolist())
    X_test = X_test.reshape(X_test.shape[0], maxNumberEvidenceSentences*nrFeature)

    tt = train_docs.apply(getTargetValue, TARGET)

    Y_train = pd.get_dummies(train_docs.apply(getTargetValue, TARGET)).as_matrix()
    Y_test = pd.get_dummies(test_docs.apply(getTargetValue, TARGET)).as_matrix()

    train_ids = sorted(train_docs.indices.keys())
    test_ids = sorted(test_docs.indices.keys())
    model.trainIndices = train_ids
    model.testIndices = test_ids
    model.split()
    model.testData.sort_values('id', inplace=True)

    #model.trainData = data.loc[train_ids]
    #model.testData = data.loc[test_ids]
    model.testTarget = test_docs.apply(getTargetValue, TARGET).tolist()

    nn = NeuralNet(maxNumberEvidenceSentences*nrFeature, nrClasses)

    #pdb.set_trace()

    with tf.Session() as sess:

        nn.setSummaryWriter('runs/Test3/', tf.get_default_graph())
        nn.buildNeuralNet(multilayer, hidden_layer_size=3, optimizerType='Adam')

        sess.run(tf.global_variables_initializer())

        batches = data_helpers.batch_iter(list(zip(X_train, Y_train)), BATCH_SIZE, ITERATIONS)

        c=0
        dropout = 0.75

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            learning_rate = nn.learningRate(c)
            print 'Learning Rate:' + str(learning_rate)

            train_data = {nn.X: x_batch, nn.Y_: y_batch, nn.step:c, nn.learning_rate: learning_rate, nn.pkeep:dropout}
            _, train_summary, grad_summary = sess.run([nn.train_step, nn.summary, nn.grad_summaries], feed_dict=train_data)

            nn.writeSummary(train_summary, c)
            nn.writeSummary(grad_summary, c)

            entropy = sess.run(nn.cross_entropy, feed_dict=train_data)
            acc = sess.run(nn.accuracy, feed_dict=train_data)
            print 'Entropy: ' + str(entropy)
            print 'Accuracy: ' + str(acc)

            if c % 100 == 0:
                testData = {nn.X: X_test, nn.Y_: Y_test, nn.learning_rate: 0, nn.pkeep:1.0}
                predictedLabels, test_summary = sess.run([nn.Y, nn.summary], feed_dict=testData)

                nn.writeSummary(test_summary, c, 'test')

            c = c + 1


        predictedLabels = sess.run(nn.Y, feed_dict=testData)
        model.testData['predictedLabel'] = np.argmax(predictedLabels, 1)
        model.testData['probability'] = np.max(predictedLabels, 1)
        test_accuracy = sess.run(nn.accuracy, feed_dict=testData)
        print 'Test Accuracy: ' + str(test_accuracy)

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

