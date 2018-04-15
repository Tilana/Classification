#! /usr/bin/env python
import tensorflow as tf
from lda import NeuralNet, Preprocessor, Info
import numpy as np
import os
import pandas as pd
from nltk.tokenize import sent_tokenize
from scripts.createSentenceDB import filterSentenceLength, setSentenceLength
from lda.osHelper import generateModelDirectory
import pdb
import pickle


def predictDoc(doc, category):

    model_path = generateModelDirectory(category)

    model_dir = os.path.join(model_path, 'model.pkl')
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    processor_dir = os.path.join(model_path, 'preprocessor')

    infoFile = os.path.join(model_path, 'info.json')
    info = Info(infoFile)

    preprocessor = Preprocessor().load(processor_dir)

    sentences = preprocessor.splitInChunks(doc.text)
    sentenceDB = pd.DataFrame(sentences, columns=['text'])

    sentenceDB['text'] = sentenceDB['text'].apply(preprocessor.cleanText)

    texts = sentenceDB.text.tolist()
    sentenceDB['tfidf'] = preprocessor.vectorizer.transform(texts)

    with open(model_dir, 'rb') as f:
        classifier = pickle.load(f)

    data = sentenceDB['tfidf'].tolist()

    probability = classifier.predict_proba(sentenceDB['tfidf'].tolist()[0])
    sentenceDB['predLabel'] = np.argmax(probability, axis=1)
    sentenceDB['probability'] = np.max(probability, axis=1)

    evidenceSentences = sentenceDB[sentenceDB['predLabel']==1]
    return evidenceSentences



if __name__=='__main__':
    predictDoc()

