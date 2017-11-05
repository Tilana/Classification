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
import json


configFile = 'dataConfig.json'
config_name = 'ICAAD_DV_sentences'
config_name = 'ICAAD_SA_sentences'
config_name = 'Manifesto_Minorities'

with open(configFile) as data_file:
    data_config = json.load(data_file)[config_name]

analyze = 0

classifierType = 'CNN'


out_folder = '_'.join([data_config['DATASET'], data_config['ID'], classifierType]) + '/'
output_dir = os.path.join(os.path.curdir, 'runs', out_folder)
checkpoint_dir = os.path.join(output_dir, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, 'model')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


BATCH_SIZE = 30
ITERATIONS = 3
cnnType = 'cnn'

#textCol = 'evidenceText_'+ID
#DATASET = 'Manifesto'
#data_path = '../data/Manifesto/manifesto_United Kingdom.csv'

def cnnClassification():

    #data_path = os.path.join(PATH, DATASET, DATASET + '_evidenceSummary.pkl')
    #data_conf['data_path'] = os.path.join(PATH, DATASET, DATASET + '.pkl')
    #data = pd.read_pickle(data_path)
    data = pd.read_csv(data_config['data_path'], encoding ='utf8')
    #data = data[3000:5000]
    #data.set_index('id', inplace=True, drop=False)
    #pdb.set_trace()

    posSample = data[data[data_config['TARGET']]==data_config['categoryOfInterest']]
    negSample = data[data[data_config['TARGET']] == data_config['negCategory']].sample(len(posSample))
    data = pd.concat([posSample, negSample])

    model = ClassificationModel(target=data_config['TARGET'], labelOfInterest=data_config['categoryOfInterest'])
    model.data = data
    model.createTarget()
    #model.createCheckpointFolder(output_dir)

    #pdb.set_trace()

    #model.data.dropna(subset=[textCol], inplace=True)
    #model.data.drop_duplicates(subset='id', inplace=True)
    #indices = pd.read_csv(MODEL_PATH + 'trainTest_split.csv', index_col=0)

    numTrainingDocs = int(len(model.data)*0.7)

    #pdb.set_trace()

    y = pd.get_dummies(model.target).values
    x_train, x_test, y_train, y_test = train_test_split(model.data[data_config['textCol']], y, test_size=0.4, random_state=200)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5, random_state=200)

    #model.splitDataset(numTrainingDocs, random=False)
    #model.trainIndices = indices.loc['train'].dropna()
    #model.testIndices= indices.loc['test'].dropna()
    model.trainIndices = x_train.index
    model.testIndices = x_test.index
    model.split()

    model.validationIndex = x_val.index
    model.validationData = model.data.loc[model.validationIndex]
    model.validationTarget = model.target.loc[model.validationIndex]

    if analyze:
        coi = model.data[model.data.category==data_config['categoryOfInterest']]
        document_lengths = [len(word_tokenize(sentence)) for sentence in coi.sentence]
        plotter = ImagePlotter(False)
        figure_path = path=os.path.join(PATH, data_config['DATASET'], 'figures', data_config['ID'] + '_evidenceSentences' + '.png')

        bins = range(1,100)
        plotter.plotHistogram(document_lengths, log=False, title= ID + ' frequency of evidence sentences length', xlabel='sentence length', ylabel='frequency', bins=bins, path=figure_path)
        print 'max: ' + str(max(document_lengths))
        print 'min: ' + str(min(document_lengths))
        print 'median: ' + str(np.median(document_lengths))

    max_document_length = max([len(x.split(" ")) for x in model.trainData[data_config['textCol']]])
    #max_document_length = 600
    print 'Maximal sentence length ' + str(max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X_train = np.array(list(vocab_processor.fit_transform(model.trainData[data_config['textCol']].tolist())))
    X_test = np.array(list(vocab_processor.transform(model.testData[data_config['textCol']].tolist())))
    X_validation = np.array(list(vocab_processor.transform(model.validationData[data_config['textCol']].tolist())))

    #pdb.set_trace()

    vocabulary = vocab_processor.vocabulary_

    Y_train = pd.get_dummies(model.trainTarget.tolist()).as_matrix()
    Y_test = pd.get_dummies(model.testTarget.tolist()).as_matrix()
    Y_validation = pd.get_dummies(model.validationTarget.tolist()).as_matrix()

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

        #pdb.set_trace()

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

    viewer = Viewer(config_name, 'newTestTest')

    viewer.printDocuments(model.testData, data_config['features'], data_config['TARGET'])
    viewer.classificationResults(model, normalized=False)


    # Prediction Phase
    graph = tf.Graph()
    with graph.as_default():
        session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_config)
        with sess.as_default():

            nn2 = NeuralNet(X_train.shape[1], nrClasses)
            nn2.loadCheckpoint(graph, sess, checkpoint_dir)

            validationData = {nn2.X: X_validation, nn2.Y_: Y_validation, nn2.learning_rate: 0, nn2.pkeep:1.0}
            predLabels = sess.run(nn2.Y, feed_dict=validationData)
            predLabels = np.argmax(predLabels, 1)

            model.validationData['predictedLabel'] = predLabels
            model.validationTarget = model.validationTarget.tolist()


            model.evaluate(subset='validation')
            model.evaluation.confusionMatrix()
            model.classifierType = classifierType

            viewer = Viewer(config_name, 'newTestTest')

            viewer.printDocuments(model.testData, data_config['features'], data_config['TARGET'])
            viewer.classificationResults(model, normalized=False)








    pdb.set_trace()



if __name__=='__main__':
    cnnClassification()

