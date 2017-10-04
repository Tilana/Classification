#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv
import pandas as pd
import pdb
from data_helpers import splitInSentences
from sklearn.metrics import accuracy_score, precision_score, recall_score
from lda import Viewer, ClassificationModel, Evaluation

def setFinalPrediction(data):
    predictions = data.predictedLabel
    if sum(predictions)>=1:
        return 1
    elif sum(predictions)==0:
        return 0
    else:
        print 'Warning: negative predictions'


def textToSentenceData(data):
    sentences = splitInSentences(data, FLAGS.target)
    return pd.DataFrame(sentences, columns=['id', FLAGS.target, 'sentences'])

def storeEvidence(data):
    return zip(data.sentences.tolist(), data.activation.tolist())


def loadProcessor(directory):
    vocab_path = os.path.join(directory, "vocab")
    return learn.preprocessing.VocabularyProcessor.restore(vocab_path)

def getTestData(directory, data):
    trainTest_split = pd.read_csv(os.path.join(directory, 'trainTest_split.csv'), index_col=0)
    testIndex = trainTest_split.loc['test'].dropna()
    return data[data['id'].isin(testIndex)]


# Parameters
# ==================================================

# Data Path
tf.flags.DEFINE_string("dataset", "ICAAD", "Dataset")
#tf.flags.DEFINE_string("id", "DV", "dataset category/target")
tf.flags.DEFINE_string("id", "SA", "dataset category/target")
tf.flags.DEFINE_string("sentence_id", "DV", "sentence category")
tf.flags.DEFINE_string("data_path", "../data", "Data path")
tf.flags.DEFINE_string("model_path", "./runs", "Model path")
#tf.flags.DEFINE_string("target", "Domestic.Violence.Manual", "Target")
tf.flags.DEFINE_string("target", "Sexual.Assault.Manual", "Target")

FLAGS = tf.flags.FLAGS
model_name  = '_'.join([FLAGS.dataset, FLAGS.id])
checkpoint_dir = os.path.join(FLAGS.model_path, model_name)

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", checkpoint_dir, "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")


def predict(data):

    model = ClassificationModel()
    model.targetFeature = FLAGS.target
    model.classifierType = 'CNN sentences'
    model.classificationType = 'binary'


    sentenceDF = textToSentenceData(data)
    x_raw = sentenceDF.sentences.tolist()

    vocab_processor = loadProcessor(FLAGS.checkpoint_dir)
    x_test = np.array(list(vocab_processor.transform(x_raw)))

    print("\nEvaluating...\n")
    checkpoint_path = os.path.join(FLAGS.checkpoint_dir, "checkpoints")
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x = graph.get_operation_by_name("input_x").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

            # Tensors we want to evaluate
            predictions = graph.get_operation_by_name("output/predictions").outputs[0]
            activation = graph.get_operation_by_name("output/scores").outputs[0]

            # Generate batches for one epoch
            batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

            all_predictions = []
            all_activations = []

            for x_test_batch in batches:
                batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                batch_activation = sess.run(activation, {input_x: x_test_batch, dropout_keep_prob: 1.0})
                batch_activation = np.max(batch_activation, axis=1)
                all_predictions = np.concatenate([all_predictions, batch_predictions])
                all_activations = np.concatenate([all_activations, batch_activation])


    sentenceDF['predictedLabel'] = all_predictions
    sentenceDF['activation'] = all_activations
    predictedEvidenceSentences = sentenceDF[sentenceDF['predictedLabel']==1]
    predictedEvidenceSentences.set_index('id', inplace=True)
    evidencePerDoc = predictedEvidenceSentences.groupby('id')
    evidenceSentences = evidencePerDoc.apply(storeEvidence)
    data = data.merge(evidenceSentences.to_frame('evidence'), left_on='id', right_index=True, how='outer')

    #pdb.set_trace()

    docs = sentenceDF.groupby('id')
    predictionPerDoc = docs.apply(setFinalPrediction)
    data = data.merge(predictionPerDoc.to_frame('predictedLabel'), left_on='id', right_index=True)

    print 'Total number of documents: {:d}'.format(len(data))
    print 'Total number of sentences: {:d}'.format(len(sentenceDF))
    print 'Total number of {:s} documents: {:d}'.format(FLAGS.target, sum(data[FLAGS.target]==1))
    print 'Total number of {:s} predicted documents: {:d}'.format(FLAGS.target, sum(data.predictedLabel==1))
    print 'Total number of {:s} predicted sentences: {:d}'.format(FLAGS.target, sum(sentenceDF.predictedLabel==1))

    model.testData = data
    model.testTarget = data[FLAGS.target].tolist()
    model.testIndices = model.testData.index
    model.evaluate()
    model.evaluation.confusionMatrix()

    viewer = Viewer(FLAGS.dataset, FLAGS.target)

    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age', 'evidence']
    viewer.printDocuments(model.testData, displayFeatures, FLAGS.target)
    viewer.classificationResults(model, normalized=False)

    pdb.set_trace()


if __name__=='__main__':
    data_name = FLAGS.dataset + '.pkl'
    data_path = os.path.join(FLAGS.data_path, FLAGS.dataset, data_name)
    data = pd.read_pickle(data_path)

    testData = getTestData(checkpoint_dir, data)
    #predict(testData[:30])
    predict(testData)

