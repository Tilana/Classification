import lda.osHelper as osHelper
from scripts import setUp
from scripts.getPretrainedEmbedding import getPretrainedEmbedding
from sklearn.naive_bayes import MultinomialNB
from collections import Counter
import tensorflow as tf
from lda import Preprocessor, NeuralNet, Info, ClassificationModel
from tensorflow.python import debug as tf_debug
import numpy as np
import pdb
import pandas as pd
import pdb
import os
import json
import pickle

DROPOUT = 0.5
STEP=20
FILTER_SIZES = [2,2,2]
PRETRAINED_WORDEMBEDDINGS = True
debug = 0
PREPROCESSING = 1

VOCABULARY_PATH = 'vocabulary.txt'

def train(evidences, category):

    model_path = osHelper.generateModelDirectory(category)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    model_dir = os.path.join(model_path, 'model.pkl')
    processor_dir = os.path.join(model_path, 'preprocessor')
    infoFile = os.path.join(model_path, 'info.json')
    memoryFile = os.path.join(model_path, 'memory.csv')

    vocabulary = pd.read_pickle(VOCABULARY_PATH)


    info = Info(infoFile)
    info.updateTrainingCounter(evidences.label.tolist())

    preprocessor = Preprocessor(vocabulary=vocabulary)
    if os.path.exists(processor_dir + '_preprocessor.pkl'):
        preprocessor = Preprocessor(vocabulary=vocabulary).load(processor_dir)
    evidences['cleanSentence'] = evidences.sentence.apply(preprocessor.cleanText)
    texts = evidences.cleanSentence.tolist()
    evidences['tfidf'] = preprocessor.trainVectorizer(texts)
    preprocessor.save(processor_dir)


    if os.path.exists(model_dir):
        with open(model_dir, 'rb') as f:
            classifier = pickle.load(f)
    else:
        classifier = MultinomialNB().fit(evidences['tfidf'].tolist(), evidences['label'].tolist())


    with open(model_dir, 'wb') as f:
        pickle.dump(classifier, f)


    memory = pd.read_csv(memoryFile, encoding='utf8')
    memory = memory.append(evidences, ignore_index=True)

    info.global_step += 1
    info.save()
    memory.to_csv(memoryFile, index=False, encoding='utf8')

    return True


