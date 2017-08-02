from modelSelection import modelSelection 
from buildClassificationModel import buildClassificationModel
from validateModel import validateModel
from lda import Collection, FeatureAnalyser, Viewer
import pdb
from lda.docLoader import loadTargets


def classificationScript():

    targets = loadTargets('Documents/ProjectTargets.csv', 'HRC_topics')
    whitelist = None
    analyse = False
    features = ['tfidf']

    for target in targets: 

        #dataPath = 'Documents/ICAAD/ICAAD.pkl'
        dataPath = 'Documents/RightDocs.csv'
        #modelPath = 'processedData/processedData_TF_binary'
        #modelPath = 'processedData/processedData'
        #modelPath = 'processedData/doc2vec'
        #modelPath = 'processedData/processedData_whitelist'
        #modelPath = 'processedData/SADV_whitelist'
        #modelPath = 'processedData/SADV'
        modelPath = 'processedData/RightDocs_topics'
        #modelPath = 'processedData/RightDocs_topics_5grams'

        collection = Collection()
        pdb.set_trace()

        if not collection.existsProcessedData(modelPath):
            collection = Collection(dataPath)
            collection.emptyDocs = 10111 
            print 'Preprocessing'
            pdb.set_trace()
            collection.cleanDataframe()
            collection.data['id'] = range(len(collection.data))
            collection.cleanTexts()
            print 'Extract Entities'
            collection.extractEntities()
            print 'Vectorize'
            collection.vectorize('tfidf', whitelist, ngrams=(1,5), maxFeatures=10000)
            print 'Set Relevant Words'
            collection.setRelevantWords()
            collection.save(modelPath)
        
        collection = Collection().load(modelPath)

        viewer = Viewer(collection.name, target)
        print 'Feature Extraction' 
        data = FeatureExtraction(collection.data[:5])

        print 'Feature Analysis'
        analyser = FeatureAnalyser()
        plotPath = 'results/' + collection.name + '/' + target + '/frequencyDistribution.jpg'
        analyser.frequencyPlots(collection, [target], plotPath)
        
        if analyse:
            analyser.frequencyPlots(collection)
            collection.correlation =  analyser.correlateVariables(collection)
            viewer = Viewer(collection.name)
            viewer.printCollection(collection)

        model  = modelSelection(collection, target, features, whitelist=whitelist)
        validateModel(model, features) 

        
        print 'Display Results'
        #displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age']
        displayFeatures = ['predictedLabel', 'probability', 'tag', 'Year', 'entities', 'DocType', 'Type1', 'Type2', 'Session', 'Date', 'agenda', 'is_last', 'order', 'favour_count', 'agains_count', 'topics', 'sponsors', 'relevantWords']
       # viewer.printDocuments(model.testData, displayFeatures, target)
        viewer.printDocuments(collection.data, displayFeatures, target)
        viewer.classificationResults(model, normalized=False)
        


if __name__=='__main__':
    classificationScript()
