from lda.docLoader import loadConfigFile
from lda import Preprocessor, ClassificationModel
from scripts import cnnClassification, cnnPrediction
import random
import json
import pandas as pd
import pdb


TRAINSIZES = [20, 50, 100]
FILTERSIZES = [2,3,5,10,12]
RUNS = 10

configFile = 'dataConfig.json'
configName = 'ICAAD_DV_sentences'

config = loadConfigFile(configFile, configName)
data = pd.read_csv(config['data_path'], encoding='utf8')

results = pd.DataFrame()
info = {}


for trainSize in TRAINSIZES:

    for idx in range(RUNS):

        posSample = data[data[config['TARGET']]==config['categoryOfInterest']]
        negSample = data[data[config['TARGET']] == config['negCategory']].sample(len(posSample))
        sentences = pd.concat([posSample, negSample])

        preprocessor = Preprocessor()
        sentences.text = sentences.text.apply(preprocessor.cleanText)

        classifier = ClassificationModel(target=config['TARGET'], labelOfInterest=config['categoryOfInterest'])
        classifier.data = sentences
        classifier.createTarget()
        classifier.splitDataset(train_size=trainSize)

        classifier.max_document_length = max([len(x.split(" ")) for x in classifier.trainData.text])
        info['_'.join([str(idx), str(trainSize)])] = classifier.max_document_length

        for filterSize in FILTERSIZES:
            model = cnnClassification(classifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=[filterSize], pretrainedWordEmbeddings=True, storeModel=0)
            modelName = '_'.join([config['DATASET'], config['ID'], str(trainSize), str(filterSize)])

            results.loc[str(idx)+'_acc', modelName] = model.evaluation.accuracy
            results.loc[str(idx)+'_prec', modelName] = model.evaluation.precision
            results.loc[str(idx)+'_rec', modelName] = model.evaluation.recall

results.to_csv('results/' + configName + '_gridSearch.csv')
infoFile = open('results/' + configName + '_sentenceLength.json', 'w')
infoFile.write(str(info))
infoFile.close()



