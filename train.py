import lda.osHelper as osHelper
from scripts import setUp
from scripts.getPretrainedEmbedding import getPretrainedEmbedding
from collections import Counter
import tensorflow as tf
from lda import Preprocessor, NeuralNet, Info
from tensorflow.python import debug as tf_debug
import numpy as np
import pdb
import pandas as pd
import pdb
import os
import json

DROPOUT = 0.5
STEP=20
FILTER_SIZES = [2,2,2]
PRETRAINED_WORDEMBEDDINGS = True
debug = 0
PREPROCESSING = 1

def train(evidences, category):

    model_path = osHelper.generateModelDirectory(category)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    processor_dir = os.path.join(model_path, 'preprocessor')
    infoFile = os.path.join(model_path, 'info.json')
    memoryFile = os.path.join(model_path, 'memory.csv')

    if not os.path.exists(checkpoint_dir):
        setUp(category, PREPROCESSING)

    info = Info(infoFile)
    info.updateTrainingCounter(evidences.label.tolist())

    if info.preprocessing:
        preprocessor = Preprocessor()
        evidences['cleanSentence'] = evidences.sentence.apply(preprocessor.cleanText)
    else:
        evidences['cleanSentence'] = evidences.sentence.str.lower()

    memory = pd.read_csv(memoryFile, encoding='utf8')

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
                summaryCache = tf.summary.FileWriterCache()
                summaryWriter = summaryCache.get(checkpoint_dir)
            else:
                nn = NeuralNet(maxSentenceLength, 2)
                nn.buildNeuralNet(vocab_size=len(vocabulary), filter_sizes=FILTER_SIZES)
                nn.setupSummaries(sess.graph, checkpoint_dir)
                sess.run(tf.global_variables_initializer())
                sess.run(tf.local_variables_initializer())

                if PRETRAINED_WORDEMBEDDINGS:
                    embedding = getPretrainedEmbedding(vocabulary)
                    sess.run(nn.W.assign(embedding))

                summaryWriter = tf.summary.FileWriter(checkpoint_dir, sess.graph)
                nn.setSaver()

            info.updateWordFrequency(evidences.groupby('label'), vocabulary)

            X = np.array(list(vocabProcessor.transform(evidences.cleanSentence.tolist())))

            Ylabels = evidences.label.astype('category', categories=[0,1])
            Y = pd.get_dummies(Ylabels).as_matrix()

            trainData = {nn.X: X, nn.Y_:Y, nn.pkeep:DROPOUT}

            if debug:
                debug_sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                [_, summary] = debug_sess.run([nn.train_step, nn.summaries], feed_dict=trainData)

            [_, summary, step] = sess.run([nn.train_step, nn.summaries, nn.global_step], feed_dict=trainData)
            #_ = sess.run(nn.train_step, feed_dict=trainData)
            #inputSummary = sess.run([nn.inputImages], feed_dict=trainData)

            #summaryWriter.add_session_log(
            #summaryWriter.add_summary(summary, len(memory)) #, 10)

            #summaryWriter = tf.summary.FileWriter(os.path.join(checkpoint_dir, 'run' + str(info.global_step)), sess.graph)
            #sessionLog = tf.SessionLog(status=tf.SessionLog.START)
            summaryWriter.add_session_log(tf.SessionLog(status=tf.SessionLog.START), info.global_step+1)
            summaryWriter.add_summary(summary, info.global_step) #, 10)
            #summaryWriter.add_summary(summary, step) #, 10)
            #summaryWriter.flush()
            #summaryWriter.add_summary(summary) #, 10)
            #print info.global_step
            #nn.saveSummary(summary, len(memory)) #, 10)
            #nn.saveCheckpoint(sess, checkpoint_dir + '/model', len(memory))
            nn.saveCheckpoint(sess, checkpoint_dir + '/model', info.global_step)
            #nn.saveCheckpoint(sess, checkpoint_dir + '/model', step)

            sess.close()

    memory = memory.append(evidences, ignore_index=True)

    info.global_step += 1

    info.save()
    memory.to_csv(memoryFile, index=False, encoding='utf8')

    return True


