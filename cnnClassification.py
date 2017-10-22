#! /usr/bin/env python
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import time
from datetime import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
from lda import ClassificationModel, Viewer, NeuralNet
import pdb
from tensorflow.python import debug as tf_debug
from sklearn.model_selection import train_test_split


PATH = '../data/'
DATASET = 'ICAAD'
ID = 'SA'
ID = 'DV'
TARGET = 'Sexual.Assault.Manual'
TARGET = 'Domestic.Violence.Manual'
MODEL_PATH = './runs/' + DATASET + '_' + ID + '/'
BATCH_SIZE = 64
ITERATIONS = 100
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

    #pdb.set_trace()
    y = pd.get_dummies(model.target).values

    x_train, x_test, y_train, y_dev = train_test_split(model.data[textCol], y, test_size=0.3, random_state=200)

    #model.splitDataset(numTrainingDocs, random=False)
    #model.trainIndices = indices.loc['train'].dropna()
    #model.testIndices= indices.loc['test'].dropna()
    model.trainIndices = x_train.index
    model.testIndices = x_test.index
    model.split()

    max_document_length = max([len(x.split(" ")) for x in model.trainData[textCol]])
    print 'Maximal sentence length ' + str(max_document_length)

    #max_document_length = 300
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X_train = np.array(list(vocab_processor.fit_transform(model.trainData[textCol].tolist())))
    X_test = np.array(list(vocab_processor.fit_transform(model.testData[textCol].tolist())))

    vocabulary = vocab_processor.vocabulary_

    Y_train = pd.get_dummies(model.trainTarget.tolist()).as_matrix()
    Y_test = pd.get_dummies(model.testTarget.tolist()).as_matrix()

    nrClasses = Y_train.shape[1]
    nn = NeuralNet(X_train.shape[1], nrClasses)


    with tf.Session() as sess:

        nn.setSummaryWriter('runs/Test3/', tf.get_default_graph())
        nn.buildNeuralNet(cnnType, sequence_length=max_document_length, vocab_size=len(vocabulary), optimizerType='Adam')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        batches = data_helpers.batch_iter(list(zip(X_train, Y_train)), BATCH_SIZE, ITERATIONS)

        c=0
        dropout = 0.5

        for batch in batches:

            x_batch, y_batch = zip(*batch)
            #x_batch = np.array(x_batch)
            #y_batch = np.array(y_batch)

            #pdb.set_trace()

            #learning_rate = nn.learningRate(c)
            learning_rate =1e-3
            #print 'Learning Rate:' + str(learning_rate)

            train_data = {nn.X: x_batch, nn.Y_: y_batch, nn.step:c, nn.learning_rate: learning_rate, nn.pkeep:dropout}
            _, train_summary, grad_summary = sess.run([nn.train_step, nn.summary, nn.grad_summaries], feed_dict=train_data)

            nn.writeSummary(train_summary, c)
            nn.writeSummary(grad_summary, c)

            entropy = sess.run(nn.cross_entropy, feed_dict=train_data)
            #acc = sess.run([nn.accuracy], feed_dict=train_data)
            #prec = sess.run([nn.precision], feed_dict=train_data)
            acc, prec, rec  = sess.run([nn.accuracy, nn.precision, nn.recall], feed_dict=train_data)
            print('Train step:')
            print('{}: step {}, loss {:g}, acc {:g}, prec {:g}, rec {:g}'.format(datetime.now().isoformat(), c, entropy, acc, prec, rec))
            #print('acc' + str(acc))
            #print('prec' + str(prec))


            if c % 50 == 0:
                testData = {nn.X: X_test, nn.Y_: Y_test, nn.learning_rate: 0, nn.pkeep:1.0}
                predictedLabels, test_summary = sess.run([nn.Y, nn.summary], feed_dict=testData)
                nn.writeSummary(test_summary, c, 'test')

            c = c + 1


        predictedLabels = sess.run(nn.Y, feed_dict=testData)
        model.testData['predictedLabel'] = predictedLabels
        #model.testData['probability'] = np.max(predictedLabels, 1)
        #test_accuracy = sess.run(nn.accuracy, feed_dict=testData)
        #print 'Test Accuracy: ' + str(test_accuracy)

        acc, prec, rec  = sess.run([nn.accuracy, nn.precision, nn.recall], feed_dict=testData)
        print('Evaluation:')
        print('{}: step {}, acc {:g}, prec {:g}, rec {:g}'.format(datetime.now().isoformat(), c, acc, prec, rec))


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

