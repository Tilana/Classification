from lda.osHelper import generateModelDirectory
import tensorflow as tf
from lda import Preprocessor, NeuralNet
import numpy as np
import pdb
import os

MAX_SENTENCE_LENGTH = 64
DROPOUT = 0.5
LEARNING_RATE = 1e-3
STEP=20

def train(sentence, category, valid):

    model_path = generateModelDirectory(category)

    preprocessor = Preprocessor()
    cleanSentence = preprocessor.cleanText(sentence)

    if os.path.isdir(model_path):

        checkpoint_dir = os.path.join(model_path, 'checkpoints')
        processor_dir = os.path.join(model_path, 'preprocessor')

        vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(processor_dir)
        X = np.array(list(vocab_processor.transform([sentence])))

        Y = np.zeros(2).reshape(1,2)
        Y[0, int(valid)] = 1

        nn = NeuralNet()
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:

                nn.loadCheckpoint(graph, sess, checkpoint_dir)

                trainData = {nn.X: X, nn.Y_:Y, nn.step:STEP, nn.learning_rate: LEARNING_RATE,  nn.pkeep:DROPOUT}

                _ = sess.run(nn.train_step, feed_dict=trainData)


                nn.saveCheckpoint(sess, checkpoint_dir + '/model', STEP)


                sess.close()

    return True

if __name__=='__main__':
    train()

