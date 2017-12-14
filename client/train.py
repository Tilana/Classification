from lda.osHelper import generateModelDirectory
from lda import Preprocessor
import pdb
import os

MAX_SENTENCE_LENGTH = 64

def train(sentence, category, valid):

    model_path = generateModelDirectory(category)

    preprocessor = Preprocessor()
    cleanSentence = preprocessor.cleanText(sentence)

    if os.path.isdir(model_path):

        checkpoint_dir = os.path.join(model_path, 'checkpoints')
        processor_dir = os.path.join(model_path, 'preprocessor')

        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(processor_dir)

    pdb.set_trace()


    #else:
    #    create new Model

    #Train Model

    #Save Model

    return True

if __name__=='__main__':
    train()

