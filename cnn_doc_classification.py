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
from lda import ClassificationModel
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
    pads[:len(activation)] = activation
    return pads



PATH = '../data/'
DATASET = 'ICAAD'
ID = 'SA'
TARGET = 'Sexual.Assault.Manual'
MODEL_PATH = './runs/' + DATASET + '_' + ID + '/'
BATCH_SIZE = 50
ITERATIONS = 100

def cnn_doc_classification():

    data_path = os.path.join(PATH, DATASET, DATASET + '.pkl')
    data = pd.read_pickle(data_path)
    data.set_index('id', inplace=True, drop=False)

    model = ClassificationModel()
    model.data = data
    model.targetFeature = TARGET
    model.target = data[TARGET]

    indices = pd.read_csv(MODEL_PATH + 'trainTest_split.csv', index_col=0)
    model.trainIndices = indices.loc['train'].dropna()
    model.testIndices= indices.loc['test'].dropna()
    model.split()

    nrDocuments = 500


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

                # Generate batches for one epoch
                batches = data_helpers.batch_iter(list(sentences), BATCH_SIZE, 1, shuffle=False)

                all_activations = np.empty((0,2))

                for x_test_batch in batches:
                    batch_activation = sess.run(activation, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                    all_activations = np.concatenate([all_activations, batch_activation])

        return all_activations



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

    print 'Maximal number of sentences in document: {:d}'.format(maxNumberSentences)

    X_train = np.array(train_docs.apply(padDocuments, maxNumberSentences).tolist())
    X_train = X_train.reshape(X_train.shape[0], maxNumberSentences*2)

    X_test = np.array(test_docs.apply(padDocuments, maxNumberSentences).tolist())
    X_test = X_test.reshape(X_test.shape[0], maxNumberSentences*2)


    # Initialize Document Classification Model
    X = tf.placeholder(tf.float32, [None, maxNumberSentences*2])
    Y_ = tf.placeholder(tf.float32, [None, 2])
    W = tf.Variable(tf.zeros([maxNumberSentences*2, 2]))
    bias = tf.Variable(tf.zeros([2]))

    init = tf.global_variables_initializer()

    Y = tf.nn.softmax(tf.matmul(X, W) + bias)

    cross_entropy = -tf.reduce_sum(Y_ * tf.log(Y))
    is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.00001)
    train_step = optimizer.minimize(cross_entropy)


    with tf.Session() as sess:
        sess.run(init)

        Y_train = pd.get_dummies(model.trainTarget.tolist()).as_matrix()

        batches = data_helpers.batch_iter(list(zip(X_train, Y_train)), BATCH_SIZE, ITERATIONS)

        for batch in batches:
            x_batch, y_batch = zip(*batch)
            x_batch = np.array(x_batch)
            y_batch = np.array(y_batch)

            train_data = {X: x_batch, Y_: y_batch}
            sess.run(train_step, feed_dict=train_data)

            entropy = sess.run(cross_entropy, feed_dict=train_data)
            print 'Entropy: ' + str(entropy)

        Y_test = pd.get_dummies(model.testTarget.tolist()).as_matrix()
        testData = {X: X_test, Y_: Y_test}
        accuracy = sess.run(accuracy, feed_dict=testData)
        print accuracy


    pdb.set_trace()



if __name__=='__main__':
    cnn_doc_classification()

