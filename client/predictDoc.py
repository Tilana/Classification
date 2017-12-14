#! /usr/bin/env python
import tensorflow as tf
from lda import NeuralNet
import numpy as np
import os
import pandas as pd
from nltk.tokenize import sent_tokenize
from createSentenceDB import filterSentenceLength, setSentenceLength
from lda.osHelper import generateModelDirectory
import pdb


def predictDoc(doc, category):

    model_path = generateModelDirectory(category)

    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    processor_dir = os.path.join(model_path, 'preprocessor')

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(processor_dir)

    sentences = sent_tokenize(doc.text)
    sentenceDB = pd.DataFrame(sentences, columns=['text'])

    sentenceDB['sentenceLength'] = sentenceDB.text.map(setSentenceLength)
    sentenceDB = sentenceDB[sentenceDB.sentenceLength.map(filterSentenceLength)]
    sentenceDB['text'] = sentenceDB['text'].str.lower()

    X_val = np.array(list(vocab_processor.transform(sentenceDB.text.tolist())))

    nn = NeuralNet()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:

            nn.loadCheckpoint(graph, sess, checkpoint_dir)
            predictions = []

            validationData = {nn.X: np.asarray(X_val), nn.pkeep:1.0}
            predictions = sess.run(nn.Y, feed_dict=validationData)

            sentenceDB['predictedLabel'] = predictions

            sess.close()

    evidenceSentences = sentenceDB[sentenceDB['predictedLabel']==1]
    return evidenceSentences



if __name__=='__main__':
    predictDoc()

