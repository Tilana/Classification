#! /usr/bin/env python
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from lda import ClassificationModel, Viewer, NeuralNet
import pdb
from tensorflow.python import debug as tf_debug


PATH = '../data/'
DATASET = 'ICAAD'
ID = 'SA'
ID = 'DV'
TARGET = 'Sexual.Assault.Manual'
TARGET = 'Domestic.Violence.Manual'
MODEL_PATH = './runs/' + DATASET + '_' + ID + '/'
BATCH_SIZE = 50
ITERATIONS = 200
cnnType = 'cnn'
textCol = 'evidenceText_'+ID

def cnnClassification():

    data_path = os.path.join(PATH, DATASET, DATASET + '_evidenceSummary.pkl')
    data = pd.read_pickle(data_path)
    data.set_index('id', inplace=True, drop=False)

    model = ClassificationModel()
    model.data = data
    model.data.dropna(subset=[textCol], inplace=True)
    model.data.drop_duplicates(subset='id', inplace=True)
    model.targetFeature = TARGET
    model.target = data[TARGET]
    model.classificationType = 'binary'

    #indices = pd.read_csv(MODEL_PATH + 'trainTest_split.csv', index_col=0)

    numTrainingDocs = int(len(model.data)*0.7)
    model.splitDataset(numTrainingDocs, random=False)
    #model.trainIndices = indices.loc['train'].dropna()
    #model.testIndices= indices.loc['test'].dropna()
    #model.split()

    max_document_length = 300
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X_train = np.array(list(vocab_processor.transform(model.trainData[textCol].tolist())))
    X_test = np.array(list(vocab_processor.transform(model.testData[textCol].tolist())))

    vocabulary = vocab_processor.vocabulary_

    Y_train = pd.get_dummies(model.trainTarget.tolist()).as_matrix()
    Y_test = pd.get_dummies(model.testTarget.tolist()).as_matrix()

    nrClasses = Y_train.shape[1]
    nn = NeuralNet(X_train.shape[1], nrClasses)


    with tf.Session() as sess:

        nn.setSummaryWriter('runs/Test3/', tf.get_default_graph())
        nn.buildNeuralNet(cnnType, sequence_length=max_document_length, vocab_size=len(vocabulary), optimizerType='Adam')

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

            if c % 50 == 0:
                testData = {nn.X: X_test, nn.Y_: Y_test, nn.learning_rate: 0, nn.pkeep:1.0}
                predictedLabels, test_summary = sess.run([nn.Y, nn.summary], feed_dict=testData)
                nn.writeSummary(test_summary, c, 'test')

            c = c + 1


        predictedLabels = sess.run(nn.Y, feed_dict=testData)
        model.testData['predictedLabel'] = predictedLabels
        #model.testData['probability'] = np.max(predictedLabels, 1)
        test_accuracy = sess.run(nn.accuracy, feed_dict=testData)
        print 'Test Accuracy: ' + str(test_accuracy)


    model.testTarget = model.testTarget.tolist()
    model.evaluate()
    model.evaluation.confusionMatrix()
    model.classifierType = 'CNN Docs'

    viewer = Viewer('DocsCNN', 'testModu')

    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age', 'evidence', textCol]
    viewer.printDocuments(model.testData, displayFeatures, TARGET)
    viewer.classificationResults(model, normalized=False)


    pdb.set_trace()



if __name__=='__main__':
    cnnClassification()

