import sys
import os
sys.path.append(os.path.abspath('../'))
from lda.docLoader import loadTargets, loadConfigFile
from lda import Preprocessor, ClassificationModel, Viewer
import pandas as pd

VOCABULARY_PATH = 'vocabulary.txt'
classifier = 'BernoulliNB'

def classificationScript():

    configFile = '../dataConfig.json'
    data_config_name = 'ICAAD_DV_sentences'
    #data_config_name = 'ICAAD_SA_sentences'

    vocabulary = pd.read_pickle(VOCABULARY_PATH)

    data_config = loadConfigFile(configFile, data_config_name)
    data = pd.read_csv('../' + data_config['data_path'], encoding ='utf8')

    analyze = False
    balanceData = 1
    validation = 1
    preprocessing = 1
    train_size = 40

    features = ['tfidf']

    if balanceData:
        posSample = data[data[data_config['TARGET']]==data_config['categoryOfInterest']]
        negSample = data[data[data_config['TARGET']] == data_config['negCategory']].sample(len(posSample))
        data = pd.concat([posSample, negSample])

    if preprocessing:
        preprocessor = Preprocessor(vocabulary=vocabulary)
        data.text = data.text.apply(preprocessor.cleanText)
        texts = data.text.tolist()
        data['tfidf'] = preprocessor.trainVectorizer(texts)

    model = ClassificationModel(target=data_config['TARGET'], labelOfInterest=data_config['categoryOfInterest'])
    model.data = data
    model.createTarget()

    model.setDataConfig(data_config)
    model.validation = validation

    model.splitDataset(train_size=train_size)

    nrTrainData = str(len(model.trainData))


    if analyze:
        analyser.frequencyPlots(collection)
        collection.correlation =  analyser.correlateVariables(collection)


    model.whitelist = None
    (score, params) = model.gridSearch(features) #, scoring=weightedFscore, scaling=False, pca=pca, components=pcaComponents)
    model.buildClassifier(classifier, params=params)
    model.trainClassifier(features)

    model.predict(features)
    model.evaluate()

    try:
        model.evaluation.confusionMatrix(model.targetLabels)
        model.relevantFeatures()
    except:
        pass

    viewer = Viewer(model.name, prefix='../')
    viewer.classificationResults(model, name=nrTrainData, normalized=False, docPath=model.doc_path)



if __name__=='__main__':
    classificationScript()
