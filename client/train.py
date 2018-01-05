from lda.osHelper import generateModelDirectory, createFolderIfNotExistent
import tensorflow as tf
from lda import Preprocessor, NeuralNet
import numpy as np
import pandas as pd
import os
import json

DROPOUT = 0.5
LEARNING_RATE = 1e-3
STEP=20
FILTER_SIZES = [2,3,4]
BATCH_SIZE = 3

def train(sentence, category, valid):

    model_path = generateModelDirectory(category)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    processor_dir = os.path.join(model_path, 'preprocessor')
    infoFile = os.path.join(model_path, 'info.json')
    batchFile = os.path.join(model_path, 'batch.csv')

    preprocessor = Preprocessor()
    cleanSentence = preprocessor.cleanText(sentence)

    batch = pd.read_csv(batchFile) ##, index_col='index')
    batch = batch.append({'sentence':cleanSentence, 'label':valid}, ignore_index=True)

    info = json.load(open(infoFile))
    info['TOTAL_NR_TRAIN_SENTENCES'] += 1
    json.dump(info, open(infoFile, 'wb'))


    if len(batch)==BATCH_SIZE:
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

                X = np.array(list(vocabProcessor.transform(batch.sentence.tolist())))

                categories = [0,1]
                Y = pd.get_dummies(batch.label.tolist(), prefix='', prefix_sep='')
                Y = Y.T.reindex(categories).T.fillna(0).as_matrix()

                trainData = {nn.X: X, nn.Y_:Y, nn.step:STEP, nn.learning_rate: LEARNING_RATE,  nn.pkeep:DROPOUT}
                _ = sess.run(nn.train_step, feed_dict=trainData)
                nn.saveCheckpoint(sess, checkpoint_dir + '/model', STEP)

                sess.close()


        batch = pd.DataFrame(columns=['index', 'sentence', 'label'])

    batch.to_csv(batchFile, index=False)

    return True

if __name__=='__main__':
    train()

