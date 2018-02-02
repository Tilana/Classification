import lda.osHelper as osHelper
from scripts import setUp
from scripts.getPretrainedEmbedding import getPretrainedEmbedding
import tensorflow as tf
from lda import Preprocessor, NeuralNet
import numpy as np
import pandas as pd
import os
import json

DROPOUT = 0.5
LEARNING_RATE = 1e-3
STEP=20
FILTER_SIZES = [2,2,2]
BATCH_SIZE = 3
PRETRAINED_WORDEMBEDDINGS = True

def train(sentence, category, valid):

    model_path = osHelper.generateModelDirectory(category)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    processor_dir = os.path.join(model_path, 'preprocessor')
    infoFile = os.path.join(model_path, 'info.json')
    batchFile = os.path.join(model_path, 'batch.csv')

    if not os.path.exists(processor_dir):
        setUp(category)

    preprocessor = Preprocessor()
    cleanSentence = preprocessor.cleanText(sentence)

    batch = pd.read_csv(batchFile) ##, index_col='index')
    batch = batch.append({'sentence':cleanSentence, 'label':valid}, ignore_index=True)

    info = json.load(open(infoFile))
    info['TOTAL_NR_TRAIN_SENTENCES'] += 1


    if len(batch)==BATCH_SIZE:
        nn = NeuralNet()
        tf.reset_default_graph()
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:

                vocabProcessor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(processor_dir)
                vocabulary = vocabProcessor.vocabulary_._mapping
                maxSentenceLength = vocabProcessor.max_document_length

                if os.path.exists(checkpoint_dir):
                    nn.loadCheckpoint(graph, sess, checkpoint_dir)
                else:
                    nn = NeuralNet(maxSentenceLength, 2)
                    nn.buildNeuralNet('cnn', sequence_length=maxSentenceLength, vocab_size=len(vocabulary), optimizerType='Adam', filter_sizes=FILTER_SIZES)
                    sess.run(tf.global_variables_initializer())
                    sess.run(tf.local_variables_initializer())

                    if PRETRAINED_WORDEMBEDDINGS:
                        embedding = getPretrainedEmbedding(vocabulary)
                        sess.run(nn.W.assign(embedding))

                    nn.setSaver()


                sentence = batch.sentence.tolist()[2]

                def getOOV(sentence):
                    return [word for word in sentence.split(' ') if word not in vocabulary]

                batch['oov'] = batch.sentence.apply(getOOV)
                OOV = set(info['OOV'])
                OOV.update(sum(batch.oov.tolist(), []))
                info['OOV'] = list(OOV)

                X = np.array(list(vocabProcessor.transform(batch.sentence.tolist())))

                Ylabels = batch.label.astype('category', categories=[0,1])
                Y = pd.get_dummies(Ylabels).as_matrix()

                trainData = {nn.X: X, nn.Y_:Y, nn.step:STEP, nn.learning_rate: LEARNING_RATE,  nn.pkeep:DROPOUT}
                _ = sess.run(nn.train_step, feed_dict=trainData)
                nn.saveCheckpoint(sess, checkpoint_dir + '/model', STEP)

                sess.close()


        batch = pd.DataFrame(columns=['index', 'sentence', 'label'])


    json.dump(info, open(infoFile, 'wb'))
    batch.to_csv(batchFile, index=False)

    return True

if __name__=='__main__':
    train()

