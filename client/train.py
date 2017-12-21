from lda.osHelper import generateModelDirectory, createFolderIfNotExistent
import tensorflow as tf
from lda import Preprocessor, NeuralNet
import numpy as np
import os

DROPOUT = 0.5
LEARNING_RATE = 1e-3
STEP=20
FILTER_SIZES = [2,3,4]

def train(sentence, category, valid):

    model_path = generateModelDirectory(category)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    processor_dir = os.path.join(model_path, 'preprocessor')

    preprocessor = Preprocessor()
    cleanSentence = preprocessor.cleanText(sentence)

    nn = NeuralNet()
    tf.reset_default_graph()
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(processor_dir)
            vocabSize = len(vocabProcessor.vocabulary_)
            maxSentenceLength = vocabProcessor.max_document_length

            if os.path.exists(checkpoint_dir):
                nn.loadCheckpoint(graph, sess, checkpoint_dir)
            else:
                nn = NeuralNet(maxSentenceLength, 2)
                nn.buildNeuralNet('cnn', sequence_length=maxSentenceLength, vocab_size=vocabSize, optimizerType='Adam', filter_sizes=FILTER_SIZES)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())
                nn.setSaver()

            X = np.array(list(vocabProcessor.transform([sentence])))
            Y = np.zeros(2).reshape(1,2)
            Y[0, int(valid)] = 1

            trainData = {nn.X: X, nn.Y_:Y, nn.step:STEP, nn.learning_rate: LEARNING_RATE,  nn.pkeep:DROPOUT}
            _ = sess.run(nn.train_step, feed_dict=trainData)
            nn.saveCheckpoint(sess, checkpoint_dir + '/model', STEP)

            sess.close()

    return True

if __name__=='__main__':
    train()

