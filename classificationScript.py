from modelSelection import modelSelection
from validateModel import validateModel
from lda import Collection, FeatureAnalyser, Viewer
from lda.docLoader import loadTargets, loadConfigFile
from lda import Preprocessor, ClassificationModel
import pandas as pd
import pdb

PATH = '../data/'


def classificationScript():

    configFile = 'dataConfig.json'
    data_config_name = 'ICAAD_DV_sentences'

    data_config = loadConfigFile(configFile, data_config_name)
    data = pd.read_csv(data_config['data_path'], encoding ='utf8')

    analyze = False
    balanceData = 1
    validation = 1
    preprocessing = 1
    splitValidationDataIndata = 0
    test_size = 0.6


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

    model.splitDataset(test_size=test_size)

    nrTrainData = str(len(model.trainData))



    #dataPath = PATH + 'ICAAD/ICAAD.pkl'
    ##dataPath = PATH + 'RightDocs.csv'
    ##modelPath = 'processedData/processedData_TF_binary'
    ##modelPath = 'processedData/processedData'
    ##modelPath = 'processedData/doc2vec'
    ##modelPath = 'processedData/processedData_whitelist'
    ##modelPath = 'processedData/SADV_whitelist'
    #modelPath = PATH + 'processedData/SADV'
    ##modelPath = 'processedData/RightDocs_topics'
    ##modelPath = 'processedData/RightDocs_topics_5grams'
    #categoryPath = PATH + name + '/CategoryLists.csv'

    #collection = Collection()

    #if not collection.existsProcessedData(modelPath):
    #    collection = Collection(dataPath)

    #    #pdb.set_trace()
    #    collection.name = name
    #    #collection.emptyDocs = 10111
    #    print 'Preprocessing'
    #    collection.cleanDataframe()
    #    collection.data['id'] = range(len(collection.data))
    #    collection.cleanTexts()
    #    print 'Extract Entities'
    #    #collection.extractEntities(categoryPath)
    #    print 'Vectorize'
    #    collection.vectorize('tfidf', whitelist, ngrams=(1,5), maxFeatures=10000)
    #    print 'Set Relevant Words'
    #    collection.setRelevantWords()
    #    collection.save(modelPath)

    #collection = Collection().load(modelPath)
    #collection.name = name
    ##pdb.set_trace()

    #print 'Feature Extraction'
    #data = FeatureExtraction_ICAAD(collection.data[:5])

    ##analyser = FeatureAnalyser()
    #plotPath = 'results/' + collection.name + '/' + target + '/frequencyDistribution.jpg'
    #analyser.frequencyPlots(collection, [target], plotPath)


    if analyze:
        analyser.frequencyPlots(collection)
        collection.correlation =  analyser.correlateVariables(collection)


    model.buildClassifier('LogisticRegression')

    #weightedFscore = model.weightFScore(2)
    model.whitelist = None

    #pdb.set_trace()
    (score, params) = model.gridSearch(features) #, scoring=weightedFscore, scaling=False, pca=pca, components=pcaComponents)
    print('Best score: %0.3f' % score)
    model.predict(features)
    model.evaluate()




    #model  = modelSelection(collection, target, features, whitelist=whitelist)
    validateModel(model, features)

    print 'Display Results'
    #displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age']
    #displayFeatures = ['predictedLabel', 'probability', 'tag', 'Year', 'entities', 'DocType', 'Type1', 'Type2', 'Session', 'Date', 'agenda', 'is_last', 'order', 'favour_count', 'agains_count', 'topics', 'sponsors', 'relevantWords']
    #viewer.printDocuments(model.testData, displayFeatures, target)

    viewer = Viewer(model.name)
    viewer.classificationResults(model, name=nrTrainData, normalized=False, docPath=model.doc_path)



if __name__=='__main__':
    classificationScript()
