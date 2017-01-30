from gensim.models import word2vec 
import os
import utils

class Word2Vec:

    def __init__(self):
        if os.path.exists('Word2Vec/text8Net.bin'):
            print 'Load trained Word2Vec net'
            self.net = word2vec.Word2Vec.load_word2vec_format('Word2Vec/text8Net.bin', binary = True)
        else:
            print 'Train Word2Vec model with text8 corpus'
            sentences = word2vec.Text8Corpus('Word2Vec/text8')
            self.net = word2vec.Word2Vec(sentences, size=200)
            self.net.init_sims(replace=True)
            self.net.save_word2vec_format('Word2Vec/text8Net.bin', binary=True)

    def getSimilarWords(self, words, nr=5):
        if not words:
            return ['default', 'default', 'default', 'default', 'default']
        return zip(*self.net.most_similar(positive=words, topn=nr))[0]

    def wordSimilarity(self, w1, w2):
        return self.net.similarity(w1, w2)

    def filterList(self, wordList):
        return [word for word in wordList if word in self.net.vocab]

    def wordToListSimilarity(self, w1, wordList):
        return [self.wordSimilarity(w1, w2) for w2 in wordList]

    def getMeanSimilarity(self, categories, words):
        meanSimilarities = []
        for keyword in categories:
            keywordSimilarities = self.wordToListSimilarity(keyword, words)
            meanSimilarities.append(utils.getMean(keywordSimilarities))
        return meanSimilarities

    def sortCategories(self, similarity, categories):
        sortedIndices = utils.indicesOfReverseSorting(similarity)
        return [categories[ind] for ind in sortedIndices]



