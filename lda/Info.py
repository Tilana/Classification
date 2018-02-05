import json

class Info:

    def __init__(self, path):
        self.path = path


    def setup(self):
        self.TOTAL_NR_TRAIN_SENTENCES = 0
        self.OOV = []
        self.NEG_WORD_FREQUENCY = {}
        self.POS_WORD_FREQUENCY = {}

        self.save()


    def save(self):
        json.dump(self.__dict__, open(self.path, 'wb'))


    def load(self):
        attributes = json.load(open(self.path))
        for name, value in attributes.iteritems():
            setattr(self, name, value)


    def updateWordFrequencyInSentence(self, sentence, wordFrequency, vocabulary):
        for word in sentence.split(' '):
            if word in vocabulary:
                if wordFrequency.get(word):
                    wordFrequency[word] += 1
                else:
                    wordFrequency[word] = 1
            else:
                self.OOV.append(word)
        self.OOV = list(set(self.OOV))


    def updateWordFrequency(self, groupedDataframe, vocabulary):
        for name, group in groupedDataframe:
            wordFrequency = self.NEG_WORD_FREQUENCY
            if name==True:
                wordFrequency = self.POS_WORD_FREQUENCY
            group.sentence.apply(self.updateWordFrequencyInSentence, wordFrequency=wordFrequency, vocabulary=vocabulary)


