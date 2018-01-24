from lda.docLoader import loadConfigFile
from lda import Preprocessor, ClassificationModel
from scripts import cnnClassification, cnnPrediction
import random
import json
import pandas as pd
import pdb

TRAINSIZES = [20, 50, 100]
FILTERSIZES = [[2], [2,2,2]]
SEQUENCE_LENGTHS = [25,50,90,120]

BASELINE_MODEL = 'LogisticRegression'

VOCABULARY_PATH = 'vocabulary.txt'

RUNS = 10

configFile = 'dataConfig.json'
configName = 'ICAAD_DV_sentences'
#configName = 'ICAAD_SA_sentences'
#configName = 'Manifesto_Minorities'

config = loadConfigFile(configFile, configName)
data = pd.read_csv(config['data_path'], encoding='utf8')

vocabulary = pd.read_pickle(VOCABULARY_PATH)

results = pd.DataFrame()

for trainSize in TRAINSIZES:

    for idx in range(RUNS):

        posSample = data[data[config['TARGET']] == config['categoryOfInterest']]
        negSample = data[data[config['TARGET']] == config['negCategory']].sample(len(posSample))
        sentences = pd.concat([posSample, negSample])

        preprocessor = Preprocessor(vocabulary=vocabulary)
        sentences.text = sentences.text.apply(preprocessor.cleanText)
        sentences['tfidf'] = preprocessor.trainVectorizer(sentences.text.tolist())

        classifier = ClassificationModel(target=config['TARGET'], labelOfInterest=config['categoryOfInterest'])
        classifier.data = sentences

        classifier.createTarget()
        classifier.splitDataset(train_size=trainSize)

        #classifier.max_document_length = max([len(x.split(" ")) for x in classifier.trainData.text])

        for filterSize in FILTERSIZES:

            for sequenceLength in SEQUENCE_LENGTHS:

                classifier.max_document_length = sequenceLength

                model = cnnClassification(classifier, ITERATIONS=200, BATCH_SIZE=50, filter_sizes=filterSize, pretrainedWordEmbeddings=True, storeModel=0, secondLayer=False)
                modelName = '_'.join([config['DATASET'], config['ID'], str(trainSize), str(filterSize), str(sequenceLength)])

                results.loc[str(idx)+'_acc', modelName] = model.evaluation.accuracy
                results.loc[str(idx)+'_prec', modelName] = model.evaluation.precision
                results.loc[str(idx)+'_rec', modelName] = model.evaluation.recall

        classifier.trainData.to_csv('results/' + configName + '_trainData' + str(trainSize) + '_RUN_' + str(idx) + '.csv')

        classifier.buildClassifier(BASELINE_MODEL)
        classifier.whitelist = None
        classifier.trainClassifier(['tfidf'])
        classifier.predict(['tfidf'])
        classifier.evaluate()

        results.loc[str(idx)+'_acc', BASELINE_MODEL + '_' + str(trainSize)] = classifier.evaluation.accuracy
        results.loc[str(idx)+'_prec', BASELINE_MODEL + '_' + str(trainSize)] = classifier.evaluation.precision
        results.loc[str(idx)+'_rec', BASELINE_MODEL + '_' + str(trainSize)] = classifier.evaluation.recall


for measure in ['acc', 'prec', 'rec']:
    rows = [name for name in results.index if name.find(measure)!=-1]
    res = results.loc[rows]
    results.loc['mean_' + measure] = res.mean(axis=0)

results.to_csv('results/' + configName + '_gridSearch.csv')



