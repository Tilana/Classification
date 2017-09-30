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
from lda import ClassificationModel
from data_helpers import splitInSentences
import pdb


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
BATCH_SIZE = 20

def cnn_doc_classification():

    data_path = os.path.join(PATH, DATASET, DATASET + '.pkl')
    data = pd.read_pickle(data_path)

    model = ClassificationModel()
    model.data = data[:20]
    model.targetFeature = TARGET
    model.target = data[TARGET]

    indices = pd.read_csv(MODEL_PATH + 'trainTest_split.csv', index_col=0)
    model.trainIndices = indices.loc['train'].dropna()
    model.testIndices= indices.loc['test'].dropna()
    model.split()


    sentenceDF = textToSentenceData(data[:100])
    x_raw = sentenceDF.sentences.tolist()

    vocab_processor = loadProcessor(MODEL_PATH)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

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
            batches = data_helpers.batch_iter(list(x_test), BATCH_SIZE, 1, shuffle=False)

            all_activations = np.empty((0,2))

            for x_test_batch in batches:
                batch_activation = sess.run(activation, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                all_activations = np.concatenate([all_activations, batch_activation])


    #pdb.set_trace()
    sentenceDF = sentenceDF.merge(pd.DataFrame(all_activations), left_index=True, right_index=True)
    docs = sentenceDF.groupby('id')
    numberSentences = docs.count()[TARGET]
    maxNumberSentences = max(numberSentences)

    #pdb.set_trace()

    X_raw = np.array(docs.apply(padDocuments, maxNumberSentences).tolist())

    #.as_matrix()
    print 'Maximal number of sentences in document: {:d}'.format(maxNumberSentences)


    #XX = tf.reshape(X_raw, [-1, maxNumberSentences])

    # Initialize Document Classification Model
    X = tf.placeholder(tf.float32, [None, maxNumberSentences, 2])
    Y_ = tf.placeholder(tf.float32, [None, 2])
    W = tf.Variable(tf.zeros([maxNumberSentences, 2]))
    bias = tf.Variable(tf.zeros([2]))

    init = tf.global_variables_initializer()

    Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, maxNumberSentences]), W) + bias)

    cross_entropy = -tf.reduce_sum(Y_ *tf.log(Y))
    is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_,1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

    optimizer = tf.train.GradientDescentOptimizer(0.003)
    train_step = optimizer.minimize(cross_entropy)

    sess = tf.Session()
    sess.run(init)
    for i in range(20):
        batch_X = X_raw
        batch_Y = pd.get_dummies(data[TARGET].tolist()[:len(X_raw)])
        train_data = {X: batch_X, Y_: batch_Y}

        sess.run(train_step, feed_dict=train_data)



    #predictedEvidenceSentences = sentenceDF[sentenceDF['predictedLabel']==1]
    #predictedEvidenceSentences.set_index('id', inplace=True)
    #evidencePerDoc = predictedEvidenceSentences.groupby('id')
    #evidenceSentences = evidencePerDoc.apply(storeEvidence)
    #data = data.merge(evidenceSentences.to_frame('evidence'), left_on='id', right_index=True, how='outer')

    #pdb.set_trace()




    pdb.set_trace()



if __name__=='__main__':
    cnn_doc_classification()

