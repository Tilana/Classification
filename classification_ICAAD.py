from lda import ClassificationModel, Viewer, Info 
import pandas as pd
import logging

def classification_ICAAD():

    ##### PARAMETERS #####
    pd.set_option('chained_assignment', None)
    logging.basicConfig()
    evaluationFile = 'Documents/PACI.csv'
    dataFeatures = pd.read_csv(evaluationFile)
    dataFeatures = dataFeatures.rename(columns={'Unnamed: 0': 'id'})

    features = [feature for feature in dataFeatures.columns.tolist() if dataFeatures[feature].dtypes==bool]
    features = ['Domestic.Violence.Manual', 'Sexual.Assault.Manual']

    info = Info()
    info.data = 'ICAAD'
    info.topicNr = 55 
    info.identifier = 'LDA_T%dP12I100_tfidf_word2vec' % info.topicNr
    info.classifierType = 'NeuralNet'

    selectedTopics = [0,2,3,6,7,8,9,12,19,24,25,34,35,36,38,42,47] # set to None selects all topics
    #selectedTopics = [0,3,6,7,8,12,24,25,34,35,36,38,42,47] # set to None selects all topics
    #selectedTopics = None
    
    ### LOAD DATA ###
    for feature in features:
        
        path = 'html/%s/DocumentFeatures.csv' % (info.data + '_' + info.identifier)
        model = ClassificationModel(path)
        
        model.droplist = ['DV', 'SA', 'File', 'id'] 
        

        model.getSelectedTopics(info.topicNr, selectedTopics)
        similarDocList = model.getSimilarDocs()
        relevantWords = model.getRelevantWords()

        model.droplist.extend(similarDocList + relevantWords)
        #model.keeplist = model.topicList 

        model.targetFeature = feature

        ### PREPROCESSING ###
        categories = 'Documents/ICAAD/CategoryLists.csv'
        categories = pd.read_csv(categories).astype(str)
        categories = pd.unique(categories.values.ravel())

        for elem in categories:
            try:
                model.data[elem] = model.data[elem].divide(sum(model.data[elem]))
            except KeyError:
                pass

        column = dataFeatures[['id', model.targetFeature]]
        model.mergeDataset(column)
        model.data = model.data.set_index('Unnamed: 0')

        model.dropNANRows()
        model.createNumericFeature('court')

        ### SELECT TEST AND TRAINING DATA ###
        model.factorFalseCases = 2 
        model.balanceDataset(model.factorFalseCases)
        model.createTarget()
        model.dropFeatures()

        model.numberTrainingDocs = len(model.data)/3
        model.splitDataset(model.numberTrainingDocs)
        
        ### CLASSIFICATION ###
        model.buildClassifier(info.classifierType)
        model.trainClassifier()
        model.predict()

        ### EVALUATION ###
        model.evaluate()
        model.evaluation.confusionMatrix()
        
        if not info.classifierType=='NeuralNet':
            model.computeFeatureImportance()
        model.getTaggedDocs()

        html = Viewer(info)
        html.classificationResults(model)

   
if __name__ == "__main__":
    classification_ICAAD()

