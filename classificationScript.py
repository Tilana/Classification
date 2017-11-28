from lda import Viewer
from lda.docLoader import loadTargets, loadConfigFile
from lda import Preprocessor, ClassificationModel
import pandas as pd
import pdb


def classificationScript():

    configFile = 'dataConfig.json'
    data_config_name = 'ICAAD_DV_sentences'
    data_config_name = 'ICAAD_SA_sentences'

    data_config = loadConfigFile(configFile, data_config_name)
    data = pd.read_csv(data_config['data_path'], encoding ='utf8')

    analyze = False
    balanceData = 1
    validation = 1
    preprocessing = 1
    train_size = 60


    features = ['tfidf']

    if balanceData:
        posSample = data[data[data_config['TARGET']]==data_config['categoryOfInterest']]
        negSample = data[data[data_config['TARGET']] == data_config['negCategory']].sample(len(posSample))
        data = pd.concat([posSample, negSample])

    if preprocessing:
        preprocessor = Preprocessor()
        data.text = data.text.apply(preprocessor.cleanText)
        texts = data.text.tolist()
        data['tfidf'] = preprocessor.trainVectorizer(texts)


    model = ClassificationModel(target=data_config['TARGET'], labelOfInterest=data_config['categoryOfInterest'])
    model.data = data
    model.createTarget()

    model.setDataConfig(data_config)
    model.validation = validation

    model.splitDataset(train_size=train_size, random_state=20)

    nrTrainData = str(len(model.trainData))


    if analyze:
        analyser.frequencyPlots(collection)
        collection.correlation =  analyser.correlateVariables(collection)


    model.buildClassifier('LogisticRegression')
    model.whitelist = None

    (score, params) = model.gridSearch(features) #, scoring=weightedFscore, scaling=False, pca=pca, components=pcaComponents)
    print('Best score: %0.3f' % score)
    model.predict(features)
    model.evaluate()


    try:
        model.evaluation.confusionMatrix(model.targetLabels)
        model.relevantFeatures()
    except:
        pass

    viewer = Viewer(model.name)
    viewer.classificationResults(model, name=nrTrainData, normalized=False, docPath=model.doc_path)

    pdb.set_trace()



if __name__=='__main__':
    classificationScript()
