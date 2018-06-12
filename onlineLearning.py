from lda.docLoader import loadConfigFile
import tensorflow as tf
from lda import ClassificationModel, Evaluation, NeuralNet, osHelper
from predictDoc import predictDoc
from train import train
import pandas as pd
import time
from shutil import rmtree
import os

NR_TRAIN_DATA = 15

def onlineLearning():

    configFile = 'dataConfig.json'
    sentences_config_name = 'ICAAD_DV_sentences'
    categoryID = 'ICAAD_DV_sentences'

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
    trainSample = classifier.trainData.sample(NR_TRAIN_DATA)
    trainSample = trainSample[['text', 'category']]. rename(columns={'text':'sentence', 'category':'label'})
    t0 = time.time()
    train(trainSample, categoryID)
    print 'Number of Training Samples: ' + str(NR_TRAIN_DATA)
    print 'TIME: ' + str(time.time() - t0)

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
    print 'TIME: ' + str(time.time() - t0)


    evaluation = Evaluation(target=classifier.testData.category.tolist(), prediction=classifier.testData.predictedLabel.tolist())
    evaluation.computeMeasures()
    evaluation.confusionMatrix()
    print 'Accuracy: ' + str(evaluation.accuracy)
    print 'Recall: ' + str(evaluation.recall)
    print 'Precision: ' + str(evaluation.precision)
    print evaluation.confusionMatrix


if __name__=='__main__':
    onlineLearning()
