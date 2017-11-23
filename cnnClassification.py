#! /usr/bin/env python
import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
import data_helpers
from lda import Viewer, NeuralNet, Evaluation


def cnnClassification(model, cnnType='cnn', BATCH_SIZE=64, ITERATIONS=100, filter_sizes=[3,4,5]):

    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(model.max_document_length)

    X_train = np.array(list(vocab_processor.fit_transform(model.trainData.text.tolist())))
    X_test = np.array(list(vocab_processor.transform(model.testData.text.tolist())))

    vocab_processor.save(model.output_dir + 'preprocessor')

    Y_train = pd.get_dummies(model.trainTarget.tolist()).as_matrix()
    Y_test = pd.get_dummies(model.testTarget.tolist()).as_matrix()

    if model.validation:
        X_validation = np.array(list(vocab_processor.transform(model.validationData.text.tolist())))
        Y_validation = pd.get_dummies(model.validationTarget.tolist()).as_matrix()

    nrTrainData = str(len(X_train))


    vocabulary = vocab_processor.vocabulary_

    nrClasses = Y_train.shape[1]
    nn = NeuralNet(X_train.shape[1], nrClasses)

    session_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess = tf.Session(config=session_config)

    with sess.as_default():

        nn.setSummaryWriter(model.output_dir, tf.get_default_graph())
        nn.buildNeuralNet(cnnType, sequence_length=model.max_document_length, vocab_size=len(vocabulary), optimizerType='Adam', filter_sizes=filter_sizes)

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
        nn.saveCheckpoint(sess, model.output_dir + 'checkpoints/model', c)

        print('Evaluation 2:')
        print('{}: step {}, entropy {:}, acc {:g}, precision {:g}, recall {:g}'.format(datetime.now().isoformat(), c, entropy, accuracy, evaluation.precision, evaluation.recall))

        model.testData['predictedLabel'] = predLabels

        ## Test Data
        model.evaluate()
        model.evaluation.confusionMatrix()
        model.classifierType = 'CNN'

        viewer = Viewer(model.name)
        viewer.classificationResults(model, name= nrTrainData + '_test', normalized=False, docPath=model.doc_path)

        ## Validation Data
        if model.validation:
            validationData = {nn.X: X_validation, nn.Y_: Y_validation, nn.learning_rate: 0, nn.pkeep:1.0, nn.step:1}
            predLabels = sess.run(nn.Y, feed_dict=validationData)

            model.validationData['predictedLabel'] = predLabels

            model.evaluate(subset='validation')
            model.evaluation.confusionMatrix()


            viewer = Viewer(model.name)
            viewer.classificationResults(model, name= nrTrainData + '_validation', subset='validation', normalized=False, docPath=model.doc_path)

        sess.close()
        tf.reset_default_graph()


