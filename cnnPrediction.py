#! /usr/bin/env python
import json
import tensorflow as tf
from lda import NeuralNet
import numpy as np
import os
import data_helpers
from tensorflow.contrib import learn
import pandas as pd
import pdb


def cnnPrediction():


    configFile = 'dataConfig.json'
    config_name = 'ICAAD_DV_sentences'


    with open(configFile) as data_file:
        data_config = json.load(data_file)[config_name]

    classifierType = 'CNN'


    out_folder = '_'.join([data_config['DATASET'], data_config['ID'], classifierType]) + '/'
    output_dir = os.path.join(os.path.curdir, 'runs', out_folder)
    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
    processor_dir = os.path.join(output_dir, 'preprocessor')

    sentences_filename = '../data/ICAAD/' + data_config['ID'] + '_sentencesValidationData.csv'
    sentenceDB = pd.read_csv(sentences_filename)
    sentenceDB = sentenceDB[:1000]

    max_document_length = 1683
    vocab_processor = learn.preprocessing.VocabularyProcessor.restore(processor_dir)


    Y_val = pd.get_dummies(sentenceDB[data_config['label']].tolist()).as_matrix()
    X_val = np.array(list(vocab_processor.transform(sentenceDB.text.tolist())))

    batches = data_helpers.batch_iter(list(zip(X_val, Y_val)), 50, 1, shuffle=False)

    predictions = []

    nn = NeuralNet()
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:

            nn.loadCheckpoint(graph, sess, checkpoint_dir)

            predictions = []

            for batch in batches:
                x_batch, y_batch = zip(*batch)
                x_test = [elem.tolist() for elem in x_batch]
                x_test = np.asarray(x_test)

                validationData = {nn.X: x_test, nn.pkeep:1.0}
                predLabels= sess.run(nn.Y, feed_dict=validationData)

                predictions.append(predLabels.tolist())


            pdb.set_trace()

            predictions =sum(predictions, [])
            sentenceDB['predictedLabel'] = predictions
            #sentenceDB.to_csv(sentences_filename, index=False)


if __name__=='__main__':
    cnnPrediction()

