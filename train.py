import lda.osHelper as osHelper
from scripts import setUp
from collections import Counter
import tensorflow as tf
from lda import Preprocessor, NeuralNet, Info, data_helpers
from tensorflow.python import debug as tf_debug
import pickle
import numpy as np
import pdb
import pandas as pd
import os
import json
from systemd import journal

#BATCH_SIZE = 10
EPOCHS = 150
DROPOUT = 0.5
FILTER_SIZES = [1,2,3]

PREPROCESSING = 1
MAX_SENTENCE_LENGTH = 40


def train(evidences, category):

    BATCH_SIZE = len(evidences)

    model_path = osHelper.generateModelDirectory(category)
    checkpoint_dir = os.path.join(model_path, 'checkpoints')
    vocab_file = os.path.join(model_path, 'vocabulary.pkl')
    processor_dir = os.path.join(model_path, 'processor.pkl')
    infoFile = os.path.join(model_path, 'info.json')
    memoryFile = os.path.join(model_path, 'memory.csv')

    if not os.path.exists(checkpoint_dir):
        setUp(category, PREPROCESSING)

    info = Info(infoFile)
    memory = pd.read_csv(memoryFile)

    nn = NeuralNet()
    tf.reset_default_graph()
    with tf.Session() as sess:

        if os.path.exists(checkpoint_dir):
            journal.send('LOAD PRETRAINED MODEL')
            nn.loadCheckpoint(sess.graph, sess, checkpoint_dir)

            preprocessor = Preprocessor().load(processor_dir)
            summaryCache = tf.summary.FileWriterCache()
            summaryWriter = summaryCache.get(checkpoint_dir)

        else:
            journal.send('BUILD NEURAL NETWORK')
            preprocessor = Preprocessor(maxSentenceLength=MAX_SENTENCE_LENGTH)
            preprocessor.setupWordEmbedding()

            nn = NeuralNet(MAX_SENTENCE_LENGTH, 2)
            nn.buildNeuralNet(vocab_size=preprocessor.vocabSize, filter_sizes=FILTER_SIZES)
            nn.setupSummaries(sess.graph, checkpoint_dir)
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            sess.run(nn.W.assign(preprocessor.embedding))

            summaryWriter = tf.summary.FileWriter(checkpoint_dir, sess.graph)
            nn.setSaver()

        journal.send('PROCESS EVIDENCE SENTENCES')
        evidences['tokens'] = evidences.sentence.apply(preprocessor.tokenize)
        vocabIds = evidences.tokens.apply(preprocessor.mapVocabularyIds).tolist()
        evidences['mapping'], evidences['oov'] = zip(*vocabIds)
        evidences['mapping'] = evidences.mapping.apply(preprocessor.padding)


        X = np.array(evidences.mapping.tolist())

        if evidences.label.dtype == 'O':
            mapping = {'True':True, 'False':False}
            evidences['label'] = evidences['label'].map(mapping)

        Ylabels = evidences.label.astype(pd.api.types.CategoricalDtype(categories=[0,1]))
        Y = pd.get_dummies(Ylabels).as_matrix()

        batches = data_helpers.batch_iter(list(zip(X, Y)), BATCH_SIZE, EPOCHS, shuffle=True)
        journal.send('START BATCH TRAINING')

        for batch in batches:

            x_batch, y_batch = zip(*batch)
            trainData = {nn.X: x_batch, nn.Y_:y_batch, nn.pkeep:DROPOUT}

            [_, summary, step] = sess.run([nn.train_step, nn.summaries, nn.global_step], feed_dict=trainData)
            summaryWriter.add_session_log(tf.SessionLog(status=tf.SessionLog.START), info.global_step+1)
            summaryWriter.add_summary(summary, info.global_step) #, 10)


        journal.send('SAVE CNN')
        nn.saveCheckpoint(sess, checkpoint_dir + '/model', info.global_step)
        preprocessor.save(processor_dir)

        sess.close()

    info.update(evidences)
    info.save()

    memory = memory.append(evidences, ignore_index=True)
    memory.to_csv(memoryFile, index=False, encoding='utf8')

    return True


