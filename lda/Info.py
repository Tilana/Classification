import json
import os

class Info:

    def __init__(self, path):
        self.path = path
        if os.path.exists(self.path):
            self.load()


    def setup(self, category='', preprocessing=False):
        self.category = category
        self.TOTAL_NR_TRAIN_SENTENCES = 0
        self.NR_TRAIN_SENTENCES_POS = 0
        self.NR_TRAIN_SENTENCES_NEG = 0
        self.global_step = 0
        self.OOV = []
        self.NEG_WORD_FREQUENCY = {}
        self.POS_WORD_FREQUENCY = {}
        self.preprocessing = preprocessing
        self.save()


    def save(self):
        json.dump(self.__dict__, open(self.path, 'wb'))


    def load(self):
        attributes = json.load(open(self.path))
        for name, value in attributes.iteritems():
            setattr(self, name, value)


    def updateWordFrequencyInSentence(self, tokens, wordFrequency):
        for word in tokens:
            if word not in self.OOV:
                if wordFrequency.get(word):
                    wordFrequency[word] += 1
                else:
                    wordFrequency[word] = 1

    def updateOOV(self, oov):
        self.OOV.extend(oov)
        self.OOV = list(set(self.OOV))


    def update(self, evidences):
        oov = sum(evidences.oov.tolist(),[])
        self.updateOOV(oov)

        groupedEvidences = evidences.groupby('label')
        for name, group in groupedEvidences:
            if name==True:
                wordFrequency = self.POS_WORD_FREQUENCY
                self.NR_TRAIN_SENTENCES_POS += len(group)
            else:
                wordFrequency = self.NEG_WORD_FREQUENCY
                self.NR_TRAIN_SENTENCES_NEG += len(group)
            group.tokens.apply(self.updateWordFrequencyInSentence, wordFrequency=wordFrequency)

        self.TOTAL_NR_TRAIN_SENTENCES += len(evidences)
        self.global_step += 1


