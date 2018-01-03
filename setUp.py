from lda.osHelper import generateModelDirectory, createFolderIfNotExistent, deleteFolderWithContent
import tensorflow as tf
from nltk.tokenize import sent_tokenize
import json
import os
import pandas as pd

MIN_NUMBER_DOCS = 5

def setUp(data, categoryID):

    modelPath = generateModelDirectory(categoryID)
    processorDir = os.path.join(modelPath, 'preprocessor')
    checkpointDir = os.path.join(modelPath, 'checkpoints')
    infoFile = os.path.join(modelPath, 'info.json')
    batchFile = os.path.join(modelPath, 'batch.csv')

    createFolderIfNotExistent(modelPath)
    deleteFolderWithContent(checkpointDir)

    sentences = [sent_tokenize(doc) for doc in data.text]
    sentences = sum(sentences, [])
    sentenceLength = [len(sentence.split(' ')) for sentence in sentences]
    maxSentenceLength = max(sentenceLength)

    if len(data)>=MIN_NUMBER_DOCS:
        vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor(maxSentenceLength)
        vocabProcessor.fit(data.text.str.lower())
        vocabulary = vocabProcessor.vocabulary_._mapping

        vocabProcessor.save(processorDir)

    else:
        # TODO: use default vocabulary
        pass

    info = {'NR_DOCS_VOCABULARY': len(data), 'TOTAL_NR_TRAIN_SENTENCES':0,}
    json.dump(info, open(infoFile, 'wb'))

    batch = pd.DataFrame(columns=['sentence', 'label'])
    batch.to_csv(batchFile, index=False)

    return True


if __name__=='__main__':
    setUp()
