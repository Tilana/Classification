#! /usr/bin/env python
import tensorflow as tf
from lda import NeuralNet, Preprocessor, Info
import numpy as np
import os
import pandas as pd
from nltk.tokenize import sent_tokenize
from scripts.createSentenceDB import filterSentenceLength, setSentenceLength
from lda.osHelper import generateModelDirectory

def predictDoc(doc, category):

    model_path = generateModelDirectory(category)

    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    processor_dir = os.path.join(model_path, 'preprocessor')

    infoFile = os.path.join(model_path, 'info.json')
    info = Info(infoFile)

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(processor_dir)

    preprocessor = Preprocessor()

    #sentences = sent_tokenize(doc.text)
    sentences = preprocessor.splitInChunks(doc.text)
    sentenceDB = pd.DataFrame(sentences, columns=['text'])

    if info.preprocessing:
        sentenceDB['text'] = sentenceDB['text'].apply(preprocessor.cleanText)
    else:
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
            predictions, probability = sess.run([nn.predictions, nn.probability], feed_dict=validationData)

            sentenceDB['predLabel'] = predictions
            sentenceDB['probability'] = probability

            sess.close()

    evidenceSentences = sentenceDB[sentenceDB['predLabel']==1]
    return evidenceSentences



if __name__=='__main__':
    predictDoc()

