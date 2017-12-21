from lda.osHelper import generateModelDirectory, createFolderIfNotExistent
import tensorflow as tf
from nltk.tokenize import sent_tokenize
import os

MIN_NUMBER_DOCS = 5

def setUp(data, categoryID):

    model_path = generateModelDirectory(categoryID)
    processor_dir = os.path.join(model_path, 'preprocessor')

    createFolderIfNotExistent(model_path)

    sentences = [sent_tokenize(doc) for doc in data.text]
    sentences = sum(sentences, [])
    sentenceLength = [len(sentence.split(' ')) for sentence in sentences]
    maxSentenceLength = max(sentenceLength)

    if len(data)>=MIN_NUMBER_DOCS:
        vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor(maxSentenceLength)
        vocabProcessor.fit(data.text.str.lower())
        vocabulary = vocabProcessor.vocabulary_._mapping

        vocabProcessor.save(processor_dir)

    else:
        # TODO: use default vocabulary
        pass

    return True


if __name__=='__main__':
    setUp()
