from modelSelection import modelSelection 
from buildClassificationModel import buildClassificationModel
from validateModel import validateModel
from lda import Collection, FeatureAnalyser, Viewer
import pdb
import csv
from lda.listUtils import flattenList, sortTupleList
from sklearn.feature_extraction.text import CountVectorizer

#targets = ['Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'Age', 'Family.Member.Victim', 'SGBV', 'Rape', 'DV.Restraining.Order', 'Penal.Code', 'Defilement', 'Reconciliation', 'Incest', 'Year']
targets = ['Year', 'DocType', 'Type1', 'Type2', 'agenda', 'is_last', 'favour', 'favour_count', 'against_count', 'order', 'sponsors_count']

with open('Documents/HRC_topics.csv', 'rb') as f:
    reader = csv.reader(f)
    targets = flattenList(list(reader))

#targets = ['OHCHR']


#whitelist = ['domestic violence', 'grievous harm', 'domestic', 'wife', 'wounding', 'bodily harm', 'batter', 'aggression', 'attack', 'protection order', 'woman']
whitelist = None

def tweetAnalyser():

    for target in targets: 

        features = ['tfidf']
        analyse = False 

        dataPath = 'Documents/gamergate_sample.csv'
        modelPath = 'processedData/gamerGate'

        #pdb.set_trace()
        
        collection = Collection(dataPath)
        #collection.cleanDataframe()
        collection.name = 'GamerGate'
        collection.emptyDocs = 0 
        collection.data['decodeTweet'] = collection.data.apply(lambda doc: doc['tweet'].encode('utf8'), axis=1)
        collection.cleanTweets()
        
        collection.extractDate()
        collection.data['year'] = collection.data.apply(lambda doc: int(doc['date'].split('-')[0]), axis=1)
        collection.data['month'] = collection.data.apply(lambda doc: int(doc['date'].split('-')[1]), axis=1)

        years = collection.data.year.unique()
        years.sort()

        for year in years:
            print year
            collectionYear = collection.data[collection.data.year==year]
            months = collectionYear.month.unique()
            months.sort()
            for month in months:
                print month
                text = ' '.join(collectionYear[collectionYear.month==month].cleanTweets.tolist())
                text = text.lower()
                vectorizer = CountVectorizer(stop_words='english', ngram_range=(1,2))
                wordCounts = vectorizer.fit_transform([text]).toarray()[0]
                vocabulary = vectorizer.vocabulary_
                invVocabulary = dict([(vocabulary[key], key) for key in vocabulary.iterkeys()])
                wordFreq = [(invVocabulary[index], freq) for index, freq in enumerate(wordCounts)]
                wordFreq = sortTupleList(wordFreq)
                print wordFreq[:50]

        pdb.set_trace()
                
        print 'Vectorize'
        collection.vectorize('tfidf', field='cleanTweets')
        collection.data['id'] = range(len(collection.data))
        print 'Set Relevant Words'
        collection.setRelevantWords()
        collection.save(modelPath)
        
        collection = Collection().load(modelPath)
        #target = targets[-18]
        target = 'Test'
        collection.name = 'GamerGate'

        pdb.set_trace()
        viewer = Viewer(collection.name, target)
        #pdb.set_trace()
        
        #data = FeatureExtraction(collection.data[:5])

        analyser = FeatureAnalyser()
        plotPath = 'results/' + collection.name + '/' + target + '/frequencyDistribution.jpg'
        analyser.frequencyPlots(collection, [target], plotPath)
        
        if analyse:
            analyser = FeatureAnalyser()
            analyser.frequencyPlots(collection)
            collection.correlation =  analyser.correlateVariables(collection)
            viewer = Viewer(collection.name)
            viewer.printCollection(collection)

        model  = modelSelection(collection, target, features, whitelist=whitelist)
        validateModel(model, features) 

        
        print 'Display Results'
        #pdb.set_trace()
        #displayFeatures = ['Court', 'Year', 'Sexual.Assault.Manual', 'Domestic.Violence.Manual', 'predictedLabel', 'tag', 'Family.Member.Victim', 'probability', 'Age']
        #displayFeatures = ['predictedLabel', 'probability', 'tag', 'Year', 'entities', 'DocType', 'Type1', 'Type2', 'Session', 'Date', 'agenda', 'is_last', 'order', 'favour_count', 'agains_count', 'topics', 'sponsors', 'relevantWords']
        displayFeatures = ['user_screen_name', 'user_description', 'user_verified', 'user_location', 'retweeted', 'retweet_count']
        collection.data['title'] = collection.data['user_screen_name']
        #viewer.printDocuments(model.testData, displayFeatures, target)
        viewer.printDocuments(collection.data, displayFeatures, target)
        viewer.classificationResults(model, normalized=False)
        
        #pdb.set_trace()


if __name__=='__main__':
    tweetAnalyser()
