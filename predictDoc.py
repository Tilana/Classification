#! /usr/bin/env python
import tensorflow as tf
from lda import NeuralNet
import numpy as np
import os
import data_helpers
import pandas as pd
from nltk.tokenize import sent_tokenize
from createSentenceDB import filterSentenceLength, setSentenceLength
import pdb


def predictDoc(doc, output_dir, splitInSentences=True):

    checkpoint_dir = os.path.join(output_dir, 'checkpoints')
    processor_dir = os.path.join(output_dir, 'preprocessor')

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(processor_dir)

    sentences = sent_tokenize(doc.text)
    sentenceDB = pd.DataFrame(sentences, columns=['text'])

    sentenceDB['sentenceLength'] = sentenceDB.text.map(setSentenceLength)
    sentenceDB = sentenceDB[sentenceDB.sentenceLength.map(filterSentenceLength)]
    sentenceDB['text'] = sentenceDB['text'].str.lower()

    X_val = np.array(list(vocab_processor.transform(sentenceDB.text.tolist())))

    nn = NeuralNet()
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            nn.loadCheckpoint(graph, sess, checkpoint_dir)
            predictions = []

            validationData = {nn.X: np.asarray(X_val), nn.pkeep:1.0}
            predictions = sess.run(nn.Y, feed_dict=validationData)

            pdb.set_trace()

            sentenceDB['predictedLabel'] = predictions

            sess.close()

    evidenceSentences = sentenceDB[sentenceDB['predictedLabel']==1]
    return evidenceSentences


if __name__=='__main__':
    predictDoc()

