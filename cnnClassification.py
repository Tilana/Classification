#! /usr/bin/env python
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import data_helpers
from tensorflow.contrib import learn
from lda import ClassificationModel, Viewer, NeuralNet, Evaluation, ImagePlotter
import pdb
from tensorflow.python import debug as tf_debug
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize


PATH = '../data/'
#DATASET = 'ICAAD'
#ID = 'SA'
#ID = 'DV'
#TARGET = 'Sexual.Assault.Manual'
#TARGET = 'Domestic.Violence.Manual'
#MODEL_PATH = './runs/' + DATASET + '_' + ID + '/'
analyze = 0

DATASET = 'ICAAD'
ID = 'DV'
#ID = 'SA'
#ID = 'noSADV'
data_path = '../data/ICAAD/sentences_ICAAD.csv'
TARGET = 'category'
categoryOfInterest = 'Evidence.of.{:s}'.format(ID)
#categoryOfInterest = 'Evidence.no.SADV'
negCategory = 'Evidence.no.SADV'
textCol = 'sentence'
classifierType = 'CNN_sentences'

out_folder = '_'.join([DATASET, ID, classifierType]) + '/'
output_dir = os.path.join(os.path.curdir, 'runs', out_folder)
checkpoint_dir = os.path.join(output_dir, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


BATCH_SIZE = 30
ITERATIONS = 30
cnnType = 'cnn'
#textCol = 'evidenceText_'+ID

#DATASET = 'Manifesto'
#data_path = '../data/Manifesto/manifesto_United Kingdom.csv'

def cnnClassification():

    #data_path = os.path.join(PATH, DATASET, DATASET + '_evidenceSummary.pkl')
    #data_path = os.path.join(PATH, DATASET, DATASET + '.pkl')
    #data = pd.read_pickle(data_path)
    data = pd.read_csv(data_path)
    #data.set_index('id', inplace=True, drop=False)

    posSample = data[data[TARGET]==categoryOfInterest]
    negSample = data[data[TARGET] == negCategory].sample(len(posSample))
    data = pd.concat([posSample, negSample])

    model = ClassificationModel(target=TARGET, labelOfInterest=categoryOfInterest)
    model.data = data
    model.createTarget()
    #model.createCheckpointFolder(output_dir)

    #pdb.set_trace()

    #model.data.dropna(subset=[textCol], inplace=True)
    #model.data.drop_duplicates(subset='id', inplace=True)
    #indices = pd.read_csv(MODEL_PATH + 'trainTest_split.csv', index_col=0)

    numTrainingDocs = int(len(model.data)*0.7)

    y = pd.get_dummies(model.target).values
    x_train, x_test, y_train, y_dev = train_test_split(model.data[textCol], y, test_size=0.3, random_state=200)

    #model.splitDataset(numTrainingDocs, random=False)
    #model.trainIndices = indices.loc['train'].dropna()
    #model.testIndices= indices.loc['test'].dropna()
    model.trainIndices = x_train.index
    model.testIndices = x_test.index
    model.split()

    if analyze:
        #document_lengths = [len(x.split(" ")) for x in model.trainData[textCol]]
        coi = model.data[model.data.category==categoryOfInterest]
        document_lengths = [len(word_tokenize(sentence)) for sentence in coi.sentence]
        plotter = ImagePlotter(False)
        figure_path = path=os.path.join(PATH, DATASET, 'figures', ID + '_evidenceSentences' + '.png')

        bins = range(1,100)
        plotter.plotHistogram(document_lengths, log=False, title= ID + ' frequency of evidence sentences length', xlabel='sentence length', ylabel='frequency', bins=bins, path=figure_path)
        print 'max: ' + str(max(document_lengths))
        print 'min: ' + str(min(document_lengths))
        print 'median: ' + str(np.median(document_lengths))

    max_document_length = max([len(x.split(" ")) for x in model.trainData[textCol]])
    #max_document_length = 600
    print 'Maximal sentence length ' + str(max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X_train = np.array(list(vocab_processor.fit_transform(model.trainData[textCol].tolist())))
    X_test = np.array(list(vocab_processor.fit_transform(model.testData[textCol].tolist())))

    vocabulary = vocab_processor.vocabulary_

    Y_train = pd.get_dummies(model.trainTarget.tolist()).as_matrix()
    Y_test = pd.get_dummies(model.testTarget.tolist()).as_matrix()

    nrClasses = Y_train.shape[1]
    nn = NeuralNet(X_train.shape[1], nrClasses)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_config)

    with sess.as_default():

        nn.setSummaryWriter(output_dir, tf.get_default_graph())
        nn.buildNeuralNet(cnnType, sequence_length=max_document_length, vocab_size=len(vocabulary), optimizerType='Adam')

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        nn.setSaver()

        batches = data_helpers.batch_iter(list(zip(X_train, Y_train)), BATCH_SIZE, ITERATIONS)

        c=0
        dropout = 0.5

        for batch in batches:

            x_batch, y_batch = zip(*batch)

            #learning_rate = nn.learningRate(c)
            learning_rate =1e-3

            train_data = {nn.X: x_batch, nn.Y_: y_batch, nn.step:c, nn.learning_rate: learning_rate, nn.pkeep:dropout}

            _, train_summary, grad_summary, entropy, acc, predLabels = sess.run([nn.train_step, nn.summary, nn.grad_summaries, nn.cross_entropy, nn.accuracy, nn.Y], feed_dict=train_data)

            evaluation = Evaluation(np.argmax(y_batch,1), predLabels)
            evaluation.computeMeasures()

            nn.writeSummary(train_summary, c)
            nn.writeSummary(grad_summary, c)

            print('Train step:')
            print('{}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}'.format(datetime.now().isoformat(), c, entropy, acc, evaluation.precision, evaluation.recall))

            if c % 100 == 0:
                testData = {nn.X: X_test, nn.Y_: Y_test, nn.learning_rate: 0, nn.pkeep:1.0}
                predLabels, test_summary = sess.run([nn.Y, nn.summary], feed_dict=testData)

                nn.writeSummary(test_summary, c, 'test')

                evaluation = Evaluation(np.argmax(Y_test,1), predLabels)
                evaluation.computeMeasures()

                print('Test step:')
                print('{}: step {}, loss {:g}, acc {:g}, precision {:g}, recall {:g}'.format(datetime.now().isoformat(), c, entropy, acc, evaluation.precision, evaluation.recall))

            c = c + 1



        predLabels, entropy, accuracy = sess.run([nn.Y, nn.cross_entropy, nn.accuracy], feed_dict=testData)
        evaluation = Evaluation(np.argmax(Y_test,1), predLabels)
        evaluation.computeMeasures()
        nn.saveCheckpoint(sess, checkpoint_prefix, c)

        print('Evaluation 2:')
        print('{}: step {}, entropy {:}, acc {:g}, precision {:g}, recall {:g}'.format(datetime.now().isoformat(), c, entropy, accuracy, evaluation.precision, evaluation.recall))

        model.testData['predictedLabel'] = predLabels

    model.testTarget = model.testTarget.tolist()
    model.evaluate()
    model.evaluation.confusionMatrix()
    model.classifierType = classifierType

    model.testData['text'] = model.testData['sentence']

    viewer = Viewer('DocsCNN', 'newTestTest')

    displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age', 'evidence', textCol]
    viewer.printDocuments(model.testData, displayFeatures, TARGET)
    viewer.classificationResults(model, normalized=False)


    # Prediction Phase
    graph = tf.Graph()
    with graph.as_default():
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_config)
        with sess.as_default():

            nn2 = NeuralNet(X_train.shape[1], nrClasses)
            nn2.loadCheckpoint(graph, sess, checkpoint_dir)

            testData = {nn2.X: X_test, nn2.Y_: Y_test, nn2.learning_rate: 0, nn2.pkeep:1.0}
            predLabels = sess.run([nn2.Y], feed_dict=testData)

    pdb.set_trace()



if __name__=='__main__':
    cnnClassification()

