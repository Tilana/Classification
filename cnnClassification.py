#! /usr/bin/env python
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from datetime import datetime
import data_helpers
from tensorflow.contrib import learn
from lda import ClassificationModel, Viewer, NeuralNet, Evaluation, ImagePlotter, Preprocessor
import pdb
from tensorflow.python import debug as tf_debug
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize, sent_tokenize
import json


configFile = 'dataConfig.json'
config_name = 'ICAAD_DV_sentences'
#config_name = 'ICAAD_SA_sentences'
#config_name = 'Manifesto_Minorities'
config_name = 'ICAAD_DV_summaries'
#config_name = 'ICAAD_SA_summaries'

splitValidationDataInSentences = False

with open(configFile) as data_file:
    data_config = json.load(data_file)[config_name]

analyze = 0

classifierType = 'CNN'


out_folder = '_'.join([data_config['DATASET'], data_config['ID'], classifierType]) + '/'
output_dir = os.path.join(os.path.curdir, 'runs', out_folder)
checkpoint_dir = os.path.join(output_dir, 'checkpoints')
checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
processor_dir = os.path.join(output_dir, 'preprocessor')

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)


BATCH_SIZE = 100
ITERATIONS = 10
cnnType = 'cnn'


def cnnClassification():

    #data_path = os.path.join(PATH, DATASET, DATASET + '_evidenceSummary.pkl')
    #data_conf['data_path'] = os.path.join(PATH, DATASET, DATASET + '.pkl')
    #data = pd.read_pickle(data_path)
    data = pd.read_csv(data_config['data_path'], encoding ='utf8')

    if data_config['balanceData']:
        posSample = data[data[data_config['TARGET']]==data_config['categoryOfInterest']]
        negSample = data[data[data_config['TARGET']] == data_config['negCategory']].sample(len(posSample))
        data = pd.concat([posSample, negSample])

    model = ClassificationModel(target=data_config['TARGET'], labelOfInterest=data_config['categoryOfInterest'])
    model.data = data
    model.createTarget()


    y = pd.get_dummies(model.target).values

    if data_config['validation']:
        x_train, x_test, y_train, y_test = train_test_split(model.data.text, y, test_size=0.60, random_state=10)
        x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.7, random_state=20)

        model.validationIndices = x_val.index
        model.validationData = model.data.loc[model.validationIndices]
        model.validationTarget = model.target.loc[model.validationIndices]

    else:
        x_train, x_test, y_train, y_test = train_test_split(model.data.text, y, test_size=0.50, random_state=10)


    #model.splitDataset(numTrainingDocs, random=False)
    #model.trainIndices = indices.loc['train'].dropna()
    #model.testIndices= indices.loc['test'].dropna()
    model.trainIndices = x_train.index
    model.testIndices = x_test.index
    model.split()


    if data_config['preprocessing']:
        preprocessor = Preprocessor()
        model.trainData['text'] = model.trainData.text.apply(preprocessor.cleanText)
        model.testData['text'] = model.testData.text.apply(preprocessor.cleanText)


    if analyze:
        coi = model.data[model.data[data_config['TARGET']]==data_config['categoryOfInterest']]
        document_lengths = [len(word_tokenize(sentence)) for sentence in coi.text]
        plotter = ImagePlotter(True)
        #figure_path = path=os.path.join(PATH, data_config['DATASET'], 'figures', data_config['ID'] + '_evidenceSentences' + '.png')

        bins = range(1,100)
        plotter.plotHistogram(document_lengths, log=False, title= data_config['ID'] + ' frequency of evidence sentences length', xlabel='sentence length', ylabel='frequency', bins=bins, path=None)
        print 'max: ' + str(max(document_lengths))
        print 'min: ' + str(min(document_lengths))
        print 'median: ' + str(np.median(document_lengths))

    max_document_length = max([len(x.split(" ")) for x in model.trainData.text])
    #max_document_length = 500
    print 'Maximal sentence length ' + str(max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

    X_train = np.array(list(vocab_processor.fit_transform(model.trainData.text.tolist())))
    X_test = np.array(list(vocab_processor.transform(model.testData.text.tolist())))

    vocab_processor.save(processor_dir)

    Y_train = pd.get_dummies(model.trainTarget.tolist()).as_matrix()
    Y_test = pd.get_dummies(model.testTarget.tolist()).as_matrix()


    if data_config['validation']:
        if data_config['preprocessing']:
            model.validationData['text'] = model.validationData.text.apply(preprocessor.cleanText)
        X_validation = np.array(list(vocab_processor.transform(model.validationData.text.tolist())))
        Y_validation = pd.get_dummies(model.validationTarget.tolist()).as_matrix()

    vocabulary = vocab_processor.vocabulary_

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

        pdb.set_trace()


        ## Test Data
        #model.testTarget = model.testTarget.tolist()
        model.evaluate()
        model.evaluation.confusionMatrix()
        model.classifierType = classifierType

        viewer = Viewer(config_name, model.classifierType + '_test')
        #viewer.printDocuments(model.testData, data_config['features'])
        viewer.classificationResults(model, normalized=False, docPath=data_config['docPath'])

        ## Validation Data
        if data_config['validation']:
            validationData = {nn.X: X_validation, nn.Y_: Y_validation, nn.learning_rate: 0, nn.pkeep:1.0, nn.step:1}
            predLabels = sess.run(nn.Y, feed_dict=validationData)

            model.validationData['predictedLabel'] = predLabels
            #model.validationTarget = model.validationTarget.tolist()

            model.evaluate(subset='validation')
            model.evaluation.confusionMatrix()
            model.classifierType = classifierType


            viewer = Viewer(config_name, model.classifierType + '_validation')
            #viewer.printDocuments(model.validationData, data_config['features'])
            viewer.classificationResults(model, subset='validation', normalized=False, docPath=data_config['docPath'])

        #pdb.set_trace()

        ### Prediction of Sentences from Documents:
        sentences_filename = '../data/ICAAD/' + data_config['ID'] + '_sentencesValidationData.csv'

        if splitValidationDataInSentences:
            from createSentenceDB import filterSentenceLength, setSentenceLength

            data = pd.read_pickle('../data/ICAAD/ICAAD.pkl')
            validationIndices = model.validationData.docID.unique()
            data = data[data.id.isin(validationIndices)]

            def splitInSentences(row):
                sentences = sent_tokenize(row.text)
                return [(row.id, row[data_config['label']], sentence) for sentence in sentences]

            sentenceDB = data.apply(splitInSentences, axis=1)
            sentenceDB = sum(sentenceDB.tolist(), [])
            sentenceDB = pd.DataFrame(sentenceDB, columns=['docID', data_config['label'], 'text'])

            sentenceDB['sentenceLength'] = sentenceDB.text.map(setSentenceLength)
            sentenceDB = sentenceDB[sentenceDB.sentenceLength.map(filterSentenceLength)]

            sentenceDB['text'] = sentenceDB['text'].str.lower()
            sentenceDB.to_csv(sentences_filename)

        else:

            sentenceDB = pd.read_csv(sentences_filename)

        #sentenceDB = sentenceDB[:1000]

        Y_val = pd.get_dummies(sentenceDB[data_config['label']].tolist()).as_matrix()
        X_val = np.array(list(vocab_processor.transform(sentenceDB.text.tolist())))

        batches = data_helpers.batch_iter(list(zip(X_val, Y_val)), 5000, 1, shuffle=False)

        predictions = []
        activations = []

        c = 0

        for batch in batches:
            print 'New batch: ' + str(c)

            x_batch, y_batch = zip(*batch)

            validationData = {nn.X: x_batch, nn.learning_rate: 0, nn.pkeep:1.0, nn.step:1}
            #predLabels = sess.run(nn.predictions, feed_dict=validationData)
            predLabels, activation = sess.run([nn.Y, nn.Ylogits], feed_dict=validationData)
            predictions.append(predLabels.tolist())
            activations.append(activation.tolist())

            c = c + 1

        pdb.set_trace()

        predictions = sum(predictions, [])
        activations = sum(activations, [])
        sentenceDB['predictedLabel'] = predictions
        sentenceDB['activation'] = activations

        sentenceDB.to_csv(sentences_filename, index=False)


        pdb.set_trace()




if __name__=='__main__':
    cnnClassification()

