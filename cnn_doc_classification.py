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


def loadProcessor(directory):
    vocab_path = os.path.join(directory, "vocab")
    return learn.preprocessing.VocabularyProcessor.restore(vocab_path)

def padDocuments(doc, nrSentences):
    pads = np.zeros((nrSentences, 2))
    activation = doc[[0,1]].as_matrix()
    pads[:len(activation)] = activation[:nrSentences]
    return pads



PATH = '../data/'
DATASET = 'ICAAD'
ID = 'SA'
TARGET = 'Sexual.Assault.Manual'
MODEL_PATH = './runs/' + DATASET + '_' + ID + '/'
BATCH_SIZE = 100
ITERATIONS = 100
multilayer = 1

def cnn_doc_classification():

    data_path = os.path.join(PATH, DATASET, DATASET + '.pkl')
    data = pd.read_pickle(data_path)
    data.set_index('id', inplace=True, drop=False)

    model = ClassificationModel()
    model.data = data
    model.targetFeature = TARGET
    model.target = data[TARGET]
    model.classificationType = 'binary'

    indices = pd.read_csv(MODEL_PATH + 'trainTest_split.csv', index_col=0)
    model.trainIndices = indices.loc['train'].dropna()
    model.testIndices= indices.loc['test'].dropna()
    model.split()

    def getSentenceActivation(sentences):

        checkpoint_path = os.path.join(MODEL_PATH, "checkpoints")
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)

        graph = tf.Graph()
        with graph.as_default():
            session_conf = tf.ConfigProto()
            sess = tf.Session(config=session_conf)
            with sess.as_default():

                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)

                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

                # Tensors we want to evaluate
                activation = graph.get_operation_by_name("output/scores").outputs[0]
                probability = tf.nn.softmax(activation)

                # Generate batches for one epoch
                batches = data_helpers.batch_iter(list(sentences), BATCH_SIZE, 1, shuffle=False)

                all_activations = np.empty((0,2))
                all_probabilities = np.empty((0,2))

                for x_test_batch in batches:
                    batch_activation = sess.run(activation, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_activations = np.concatenate([all_activations, batch_activation])
                    batch_probabilities = sess.run(probability, {input_x: x_test_batch, dropout_keep_prob:1.0})
                    all_probabilities = np.concatenate([all_probabilities, batch_probabilities])

        return all_probabilities



    vocab_processor = loadProcessor(MODEL_PATH)
    trainSentences = textToSentenceData(model.trainData)
    testSentences = textToSentenceData(model.testData)

    x_train = np.array(list(vocab_processor.transform(trainSentences.sentences.tolist())))
    x_test = np.array(list(vocab_processor.transform(testSentences.sentences.tolist())))

    activations_train = getSentenceActivation(x_train)
    activations_test = getSentenceActivation(x_test)

    trainSentences = trainSentences.merge(pd.DataFrame(activations_train), left_index=True, right_index=True)
    testSentences = testSentences.merge(pd.DataFrame(activations_test), left_index=True, right_index=True)

    train_docs = trainSentences.groupby('id')
    test_docs = testSentences.groupby('id')
    numberSentences = train_docs.count()[TARGET]
    maxNumberSentences = max(numberSentences)
    maxNumberSentences = 100
    nrClasses = len(trainSentences[TARGET].unique())

    print 'Maximal number of sentences in document: {:d}'.format(maxNumberSentences)

    X_train = np.array(train_docs.apply(padDocuments, maxNumberSentences).tolist())
    X_train = X_train.reshape(X_train.shape[0], maxNumberSentences*2)

    X_test = np.array(test_docs.apply(padDocuments, maxNumberSentences).tolist())
    X_test = X_test.reshape(X_test.shape[0], maxNumberSentences*2)


    nn = NeuralNet(maxNumberSentences*activations_train.shape[1], nrClasses)

    with tf.Session() as sess:
        nn.setSession(sess)

        nn.setSummaryWriter('runs/Test2/', sess.graph)
        nn.buildNeuralNet(multilayer=1, hidden_layer_size=100, optimizerType='GD')
        nn.initializeVariables()

        #pdb.set_trace()

        Y_train = pd.get_dummies(model.trainTarget.tolist()).as_matrix()
        Y_test = pd.get_dummies(model.testTarget.tolist()).as_matrix()

        batches = data_helpers.batch_iter(list(zip(X_train, Y_train)), BATCH_SIZE, ITERATIONS)

        c=0

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            train_data = {nn.X: x_batch, nn.Y_: y_batch, nn.step:c}
            _, train_summary = sess.run([nn.train_step, nn.summary], feed_dict=train_data)

            nn.writeSummary(train_summary, c)

            entropy = sess.run(nn.cross_entropy, feed_dict=train_data)
            acc = sess.run(nn.accuracy, feed_dict=train_data)
            print 'Entropy: ' + str(entropy)
            print 'Accuracy: ' + str(acc)

            if c % 10 == 0:
                testData = {nn.X: X_test, nn.Y_: Y_test}
                predictedLabels, test_summary = sess.run([nn.Y, nn.summary], feed_dict=testData)

                nn.writeSummary(test_summary, c, 'test')

            c = c + 1


        testData = {nn.X: X_test, nn.Y_: Y_test}
        predictedLabels = sess.run(nn.Y, feed_dict=testData)
        model.testData['predictedLabel'] = np.argmax(predictedLabels, 1)
        model.testData['probability'] = np.max(predictedLabels, 1)
        test_accuracy = sess.run(nn.accuracy, feed_dict=testData)
        print 'Test Accuracy: ' + str(test_accuracy)


    model.testTarget = model.testTarget.tolist()
    model.evaluate()
    model.evaluation.confusionMatrix()
    model.classifierType = 'CNN Docs'

    viewer = Viewer('DocsCNN', 'test')

    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age', 'evidence']
    viewer.printDocuments(model.testData, displayFeatures, TARGET)
    viewer.classificationResults(model, normalized=False)


    pdb.set_trace()



if __name__=='__main__':
    cnn_doc_classification()

