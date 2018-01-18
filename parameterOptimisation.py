from lda.docLoader import loadConfigFile
from lda import Preprocessor, ClassificationModel
from scripts import cnnClassification, cnnPrediction
import random
import json
import pandas as pd
import pdb


TRAINSIZES = [25]
FILTERSIZES = [[2], [2,2,2]]
WORD2VEC = [False, True]

RUNS = 3

configFile = 'dataConfig.json'
configName = 'ICAAD_DV_sentences'
#configName = 'ICAAD_SA_sentences'
#configName = 'Manifesto_Minorities'

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

            for useWord2Vec in WORD2VEC:
                model = cnnClassification(classifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=filterSize, pretrainedWordEmbeddings=useWord2Vec, storeModel=0, secondLayer=False)
                modelName = '_'.join([config['DATASET'], config['ID'], str(trainSize), str(filterSize), str(useWord2Vec)])

                results.loc[str(idx)+'_acc', modelName] = model.evaluation.accuracy
                results.loc[str(idx)+'_prec', modelName] = model.evaluation.precision
                results.loc[str(idx)+'_rec', modelName] = model.evaluation.recall

results.to_csv('results/' + configName + '_gridSearch.csv')
infoFile = open('results/' + configName + '_sentenceLength.json', 'w')
infoFile.write(str(info))
infoFile.close()



