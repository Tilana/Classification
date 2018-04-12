#!/usr/bin/env python
import Pyro.core
import fastText

class WordEmbedding(Pyro.core.ObjBase):

    def __init__(self, path='WordEmbedding/wiki.en.bin'):
        Pyro.core.ObjBase.__init__(self)
        self.wordEmbedding = fastText.load_model(path)

    def getWordVector(self, word):
        print word
        return self.wordEmbedding.get_word_vector(word)

    def getLabels(self, topN=40000):
        return self.wordEmbedding.get_labels()[:topN]

Pyro.core.initServer()
daemon = Pyro.core.Daemon()
uri = daemon.connect(WordEmbedding(), "wordEmbedding")

print("The daemon runs on port: {port}".format(port=daemon))
print("The object's uri is: {uri}".format(uri=uri))

daemon.requestLoop()




