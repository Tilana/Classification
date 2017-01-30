import utils
import ImagePlotter 

class Topic:

    def __init__(self):
        self.number = None
        self.wordDistribution = []
        self.relatedDocuments = []

    def addTopic(self, topicTuple):
        self.number = topicTuple[0]
        self.wordDistribution = topicTuple[1]

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def setAttribute(self, name, value):
        setattr(self, name, value)

    def getTopicWords(self):
        return zip(*self.wordDistribution)[0][0:7]

    def labelTopic(self, word2vec, categories):
        topicWords = word2vec.filterList(self.getTopicWords()) 
        similarWords = word2vec.getSimilarWords(topicWords)
        meanSimilarity = word2vec.getMeanSimilarity(categories, similarWords)
        self.keywords = word2vec.sortCategories(meanSimilarity, categories)

    def findIntruder(self, word2vec):
       topicWords = word2vec.filterList(self.getTopicWords())
       if not topicWords:
           self.intruder = 'default'
       else:
           self.intruder = word2vec.net.doesnt_match(topicWords)
    
    
    def computeSimilarityScore(self, word2vec):
        topicWords = word2vec.filterList(self.getTopicWords())
        if not topicWords:
            self.pairwiseSimilarity = []
            self.medianSimilarity = 0
        else:
            similarityMatrix = [word2vec.wordToListSimilarity(word, topicWords) for word in topicWords]
            self.pairwiseSimilarity = utils.getUpperSymmetrixMatrix(similarityMatrix)
            self.medianSimilarity = utils.getMedian(self.pairwiseSimilarity)

    def getRelevanceHistogram(self, info):
        path = 'html/' + info.data + '_' + info.identifier + '/Images/documentRelevance_topic%d.jpg' % self.number
        if zip(*self.relatedDocuments)!=[]:
            self.relevanceScores = zip(*self.relatedDocuments)[0]
        else:
            self.relevanceScores = [0]

        title = 'Frequency of Relevant Documents for Topic %d' % self.number
        ImagePlotter.plotHistogram(self.relevanceScores, title, path, 'Relevance', 'Number of Documents', log=1, open=0)

        
            


