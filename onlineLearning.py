from lda.docLoader import loadConfigFile
import tensorflow as tf
from lda import ClassificationModel, Evaluation, NeuralNet, osHelper
from predictDoc import predictDoc
from train import train
import pandas as pd
import time
from shutil import rmtree
import os
import matplotlib.pyplot as plt


def onlineLearning(NR_TRAIN_DATA=20):

    configFile = 'dataConfig.json'
    sentences_config_name = 'ICAAD_DV_sentences'
    sentences_config_name = 'ICAAD_SA_sentences'
    categoryID = 'ICAAD_DV_sentences'
    categoryID = 'ICAAD_SA_sentences'

    model_path = osHelper.generateModelDirectory(categoryID)

    # Get Sentence dataset
    sentences_config = loadConfigFile(configFile, sentences_config_name)
    sentences = pd.read_csv(sentences_config['data_path'], encoding ='utf8')

    noDV = sentences[sentences.category=='Evidence.no.SADV'].sample(655)
    sentences = sentences[sentences.category=='Evidence.of.DV'].append(noDV)

    classifier = ClassificationModel(target=sentences_config['TARGET'])
    classifier.data = sentences
    classifier.createTarget()
    classifier.splitDataset(train_size=0.90, random_state=20)

    rmtree(model_path, ignore_errors=True)

    print '*** TRAINING ***'
    trainSample = classifier.trainData.sample(NR_TRAIN_DATA, random_state=42)
    trainSample = trainSample[['text', 'category']]. rename(columns={'text':'sentence', 'category':'label'})
    t0 = time.time()
    train(trainSample, categoryID)
    print 'Number of Training Samples: ' + str(NR_TRAIN_DATA)
    training_time = time.time() - t0
    print 'TIME: ' + str(training_time)

    classifier.testData['predictedLabel'] = 0
    t0 = time.time()
    print '*** PREDICTION ***'

    nn = NeuralNet()
    tf.reset_default_graph()
    graph = tf.Graph()

    with graph.as_default():
        with tf.Session() as session:
            print categoryID
            checkpoint_dir = os.path.join(model_path, 'checkpoints')
            nn.loadCheckpoint(graph, session, checkpoint_dir)

            for ind, sample in classifier.testData.iterrows():
                evidenceSentences = predictDoc(sample, categoryID, nn, session)
                if len(evidenceSentences)>=1:
                    classifier.testData.loc[ind, 'predictedLabel'] = 1
                    classifier.testData.loc[ind, 'probability'] = evidenceSentences.probability.tolist()[0]
            session.close()
    print 'Number of Test Sentences: ' + str(len(classifier.testData))
    prediction_time = time.time() - t0
    print 'TIME: ' + str(time.time() - t0)


    evaluation = Evaluation(target=classifier.testData.category.tolist(), prediction=classifier.testData.predictedLabel.tolist())
    evaluation.computeMeasures()
    evaluation.confusionMatrix()
    print 'Accuracy: ' + str(evaluation.accuracy)
    print 'Recall: ' + str(evaluation.recall)
    print 'Precision: ' + str(evaluation.precision)
    print evaluation.confusionMatrix

    return (training_time, prediction_time, evaluation.accuracy, evaluation.recall, evaluation.precision, len(classifier.testData))


if __name__=='__main__':
    results = []
    NR_TRAIN_DATA_ARY = [10,20,40,60,100,200]
    for NR_TRAIN_DATA in NR_TRAIN_DATA_ARY:
        results.append(onlineLearning(NR_TRAIN_DATA))

    performance = pd.DataFrame(results, index=NR_TRAIN_DATA_ARY, columns=['training_time', 'prediction_time', 'accuracy', 'recall', 'precision', 'nrTestData'])
    print 'Number of test sentences: ' + str(performance['nrTestData'].tolist())

    ax = performance[['training_time', 'prediction_time']].plot(kind='bar', legend=True, title='CNN Computation Time - ICAAD SA sentences')
    ax.set_ylabel('Time in s')
    ax.set_xlabel('Number of Training Sentences')
    ax.legend(loc="upper left")
    plt.savefig('ICAAD_SA_computation_time.png')

    ax = performance[['accuracy', 'recall', 'precision']].plot(kind='bar', legend=True, title='CNN Performance - ICAAD SA sentences', ylim=(0.6,1))
    ax.set_ylabel('Performance in %')
    ax.set_xlabel('Number of Training Sentences')
    ax.legend(loc="upper left")
    plt.savefig('ICAAD_SA_performance.png')
