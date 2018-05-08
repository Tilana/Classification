#! /usr/bin/env python
import tensorflow as tf
import pickle
from lda import NeuralNet, Preprocessor, Info
import numpy as np
import os
import pandas as pd
from sentenceTokenizer import tokenize
from scripts.createSentenceDB import filterSentenceLength, setSentenceLength
from lda.osHelper import generateModelDirectory


def predictDoc(doc, category):

    model_path = generateModelDirectory(category)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    processor_dir = os.path.join(model_path, 'processor.pkl')

    infoFile = os.path.join(model_path, 'info.json')
    info = Info(infoFile)

    sentences = tokenize(doc.text)
    sentenceDB = pd.DataFrame(sentences, columns=['sentence'])

    preprocessor = Preprocessor().load(processor_dir)
    sentenceDB['tokens'] = sentenceDB.sentence.apply(preprocessor.tokenize)
    vocabIds = sentenceDB.tokens.apply(preprocessor.mapVocabularyIds).tolist()
    sentenceDB['mapping'], sentenceDB['oov'] = zip(*vocabIds)
    sentenceDB['mapping'] = sentenceDB.mapping.apply(preprocessor.padding)

    X_val = np.array(sentenceDB.mapping.tolist())

    nn = NeuralNet()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:

            nn.loadCheckpoint(graph, sess, checkpoint_dir)
            predictions = []

            validationData = {nn.X: np.asarray(X_val), nn.pkeep:1.0}
            predictions, probability = sess.run([nn.predictions, nn.probability], feed_dict=validationData)

            sentenceDB['predictedLabel'] = predictions
            sentenceDB['probability'] = probability

            sess.close()

    evidenceSentences = sentenceDB[sentenceDB['predictedLabel']==1]
    return evidenceSentences



if __name__=='__main__':
    predictDoc()

