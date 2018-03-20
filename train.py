import lda.osHelper as osHelper
from scripts import setUp
from scripts.getPretrainedEmbedding import getPretrainedEmbedding
from collections import Counter
import tensorflow as tf
from lda import Preprocessor, NeuralNet, Info
from tensorflow.python import debug as tf_debug
import pickle
import numpy as np
import pdb
import pandas as pd
import pdb
import os
import json

DROPOUT = 0.5
FILTER_SIZES = [2,2,2]
PREPROCESSING = 1
VOCAB_SIZE = 40000
MAX_SENTENCE_LENGTH = 90

def train(evidences, category):

    model_path = osHelper.generateModelDirectory(category)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    vocab_file = os.path.join(model_path, 'vocabulary.pkl')
    infoFile = os.path.join(model_path, 'info.json')
    memoryFile = os.path.join(model_path, 'memory.csv')

    if not os.path.exists(checkpoint_dir):
        setUp(category, PREPROCESSING)

    info = Info(infoFile)
    info.updateTrainingCounter(evidences.label.tolist())

    memory = pd.read_csv(memoryFile)

    nn = NeuralNet()
    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():
        with tf.Session() as sess:

            if os.path.exists(checkpoint_dir):
                nn.loadCheckpoint(graph, sess, checkpoint_dir)
                with open(vocab_file, 'rb') as vocab:
                    vocabulary = pickle.load(vocab)
                summaryCache = tf.summary.FileWriterCache()
                summaryWriter = summaryCache.get(checkpoint_dir)
            else:
                nn = NeuralNet(MAX_SENTENCE_LENGTH, 2)
                nn.buildNeuralNet(vocab_size=VOCAB_SIZE, filter_sizes=FILTER_SIZES)
                nn.setupSummaries(sess.graph, checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                embedding, vocabulary = getPretrainedEmbedding()
                sess.run(nn.W.assign(embedding))

                with open(vocab_file, 'wb') as vocab:
                    pickle.dump(vocabulary, vocab, protocol=pickle.HIGHEST_PROTOCOL)

                summaryWriter = tf.summary.FileWriter(checkpoint_dir, sess.graph)
                nn.setSaver()

            preprocessor = Preprocessor(vocabulary=vocabulary, maxSentenceLength=MAX_SENTENCE_LENGTH)
            evidences['tokens'] = evidences.sentence.apply(preprocessor.tokenize)
            vocabIds = evidences.tokens.apply(preprocessor.mapVocabularyIds).tolist()
            evidences['mapping'], evidences['oov'] = zip(*vocabIds)
            evidences['mapping'] = evidences.mapping.apply(preprocessor.padding)
            info.updateWordFrequency(evidences.groupby('label'), vocabulary)

            X = np.array(evidences.mapping.tolist())

            Ylabels = evidences.label.astype('category', categories=[0,1])
            Y = pd.get_dummies(Ylabels).as_matrix()

            trainData = {nn.X: X, nn.Y_:Y, nn.pkeep:DROPOUT}

            [_, summary, step] = sess.run([nn.train_step, nn.summaries, nn.global_step], feed_dict=trainData)
            summaryWriter.add_session_log(tf.SessionLog(status=tf.SessionLog.START), info.global_step+1)
            summaryWriter.add_summary(summary, info.global_step) #, 10)
            nn.saveCheckpoint(sess, checkpoint_dir + '/model', info.global_step)

            sess.close()

    memory = memory.append(evidences, ignore_index=True)

    info.global_step += 1

    info.save()
    memory.to_csv(memoryFile, index=False)

    return True


