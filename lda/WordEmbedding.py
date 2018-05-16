#!/usr/bin/env python
import Pyro.core
import fastText
import csv

#path = 'WordEmbedding/FastText_wiki-news-300d-1M-subword'
path = 'WordEmbedding/FastText_wiki-news-300d-40000-subword'
USE_DAEMON = False

class WordEmbedding(Pyro.core.ObjBase):

    def __init__(self):
        if USE_DAEMON:
            Pyro.core.ObjBase.__init__(self)
        self.wordEmbedding = fastText.load_model(path + '.bin')
        with open(path + '.vec', 'rb') as f:
            self.vocabulary = [line.split(' ')[0] for line in f.readlines()][1:]

    def getVocabulary(self, nTop=50000):
        return self.vocabulary[:nTop]

    def getWordVector(self, word):
        print word
        return self.wordEmbedding.get_word_vector(word)

    def toTSV(self, nTop=50000):
        self.vectorsToTSV(nTop)
        self.vocabularyToTSV(nTop)

    def vectorsToTSV(self, nTop=50000):
        with open(path + '.tsv', 'wb') as f:
            writer = csv.writer(f, delimiter='\t', lineterminator='\n')
            for word in self.vocabulary[:nTop]:
                writer.writerow(self.getWordVector(word))
        f.close()

    def vocabularyToTSV(self, nTop=50000):
        with open(path + '_metadata.tsv', 'wb') as f:
            vocabWithLineSeparator = [word + '\n' for ind, word in enumerate(self.vocabulary[:nTop]) if ind<nTop-1]
            f.writelines(vocabWithLineSeparator)
        f.close()

if USE_DAEMON:
    Pyro.core.initServer()
    daemon = Pyro.core.Daemon()
    uri = daemon.connect(WordEmbedding(), "wordEmbedding")

    print("The daemon runs on port: {port}".format(port=daemon))
    print("The object's uri is: {uri}".format(uri=uri))

    daemon.requestLoop()


