import lda.osHelper as osHelper
from lda import Info
import tensorflow as tf
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

    vocabPath = 'vocabulary.txt'
    vocabulary = pd.read_pickle(vocabPath)

    vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor(MAX_SENTENCE_LENGTH)
    vocabProcessor.fit(vocabulary)
    vocabProcessor.save(processorDir)

    info = Info(infoFile)
    info.setup()

    batch = pd.DataFrame(columns=['sentence', 'label'])
    batch.to_csv(batchFile, index=False)

    return True


if __name__=='__main__':
    setUp()
