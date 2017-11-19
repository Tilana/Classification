#! /usr/bin/env python
import tensorflow as tf
from lda import NeuralNet
import numpy as np
import os
import data_helpers
import pandas as pd


def cnnPrediction(data, label, output_dir):


    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    processor_dir = os.path.join(output_dir, 'preprocessor')

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(processor_dir)

    Y_val = pd.get_dummies(data[label].tolist()).as_matrix()
    X_val = np.array(list(vocab_processor.transform(data.text.tolist())))

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


            predictions =sum(predictions, [])
            data['predictedLabel'] = predictions

            sess.close()

    return data


if __name__=='__main__':
    cnnPrediction()

