from modelSelection import modelSelection
from validateModel import validateModel
from lda import Collection, FeatureAnalyser, Viewer
from lda.docLoader import loadTargets
import pdb

PATH = '../data/'


def classificationScript():

    name = 'ICAAD'
    targets = loadTargets(PATH + 'ProjectTargets.csv', 'HRC_topics')
    targets = loadTargets(PATH + 'ProjectTargets.csv', 'ICAAD')
    whitelist = None
    analyse = False
    features = ['tfidf']

    for target in targets[:2]:

        dataPath = PATH + 'ICAAD/ICAAD.pkl'
        #dataPath = PATH + 'RightDocs.csv'
        #modelPath = 'processedData/processedData_TF_binary'
        #modelPath = 'processedData/processedData'
        #modelPath = 'processedData/doc2vec'
        #modelPath = 'processedData/processedData_whitelist'
        #modelPath = 'processedData/SADV_whitelist'
        modelPath = PATH + 'processedData/SADV'
        #modelPath = 'processedData/RightDocs_topics'
        #modelPath = 'processedData/RightDocs_topics_5grams'
        categoryPath = PATH + name + '/CategoryLists.csv'

        collection = Collection()

        if not collection.existsProcessedData(modelPath):
            collection = Collection(dataPath)

            #pdb.set_trace()
            collection.name = name
            #collection.emptyDocs = 10111
            print 'Preprocessing'
            collection.cleanDataframe()
            collection.data['id'] = range(len(collection.data))
            collection.cleanTexts()
            print 'Extract Entities'
            #collection.extractEntities(categoryPath)
            print 'Vectorize'
            collection.vectorize('tfidf', whitelist, ngrams=(1,5), maxFeatures=10000)
            print 'Set Relevant Words'
            collection.setRelevantWords()
            collection.save(modelPath)

        collection = Collection().load(modelPath)
        collection.name = name
        #pdb.set_trace()
        viewer = Viewer(collection.name, target)

        #print 'Feature Extraction'
        #data = FeatureExtraction_ICAAD(collection.data[:5])

        print 'Feature Analysis'
        analyser = FeatureAnalyser()
        plotPath = 'results/' + collection.name + '/' + target + '/frequencyDistribution.jpg'
        analyser.frequencyPlots(collection, [target], plotPath)

        if analyse:
            analyser.frequencyPlots(collection)
            collection.correlation =  analyser.correlateVariables(collection)
            viewer.printCollection(collection)

        model  = modelSelection(collection, target, features, whitelist=whitelist)
        validateModel(model, features)

        print 'Display Results'
        displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age']
        #displayFeatures = ['predictedLabel', 'probability', 'tag', 'Year', 'entities', 'DocType', 'Type1', 'Type2', 'Session', 'Date', 'agenda', 'is_last', 'order', 'favour_count', 'agains_count', 'topics', 'sponsors', 'relevantWords']
        viewer.printDocuments(model.testData, displayFeatures, target)
        viewer.classificationResults(model, normalized=False)



if __name__=='__main__':
    classificationScript()
