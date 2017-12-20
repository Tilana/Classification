from lda.osHelper import generateModelDirectory, createFolderIfNotExistent
import tensorflow as tf
import pdb

MIN_NUMBER_DOCS = 30

def setUp(data, categoryID):

    model_path = generateModelDirectory(categoryID)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    processor_dir = os.path.join(model_path, 'preprocessor')

    if len(data)>=MIN_NUMBER_DOCS:
        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor

        pdb.set_trace()



    # If enough documents available:
    #   create vocabulary
    # else:
    #   use default vocabulary

    # determine MAX_SENTENCE_LENGTH
    # determine VOCAB_SIZE


    # later: get number of categories in select to determine number of output layers

if __name__=='__main__':
    setUp()
