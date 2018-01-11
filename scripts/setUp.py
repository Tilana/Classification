import lda.osHelper as osHelper
import tensorflow as tf
import gensim.models.keyedvectors as w2v_model
import json
import os
import pandas as pd


MAX_SENTENCE_LENGTH = 90

def setUp(categoryID):

    modelPath = osHelper.generateModelDirectory(categoryID)
    processorDir = os.path.join(modelPath, 'preprocessor')
    checkpointDir = os.path.join(modelPath, 'checkpoints')
    infoFile = os.path.join(modelPath, 'info.json')
    batchFile = os.path.join(modelPath, 'batch.csv')

    osHelper.createFolderIfNotExistent(modelPath)
    osHelper.deleteFolderWithContent(checkpointDir)

    word2vec = w2v_model.KeyedVectors.load_word2vec_format('Word2Vec/GoogleNews-vectors-negative300.bin', binary=True)
    vocabulary = {key:value.index for key, value in word2vec.vocab.iteritems() if key.islower() and '_' not in key}

    vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SENTENCE_LENGTH)
    vocabProcessor.fit(vocabulary)
    vocabProcessor.save(processorDir)

    info = {'TOTAL_NR_TRAIN_SENTENCES':0}
    json.dump(info, open(infoFile, 'wb'))

    batch = pd.DataFrame(columns=['sentence', 'label'])
    batch.to_csv(batchFile, index=False)

    return True


if __name__=='__main__':
    setUp()
